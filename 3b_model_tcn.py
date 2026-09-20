"""One causal dilated TCN behind two interchangeable per-epoch encoders.
DATA_MODE='features' gives MLP-TCN, a small MLP over the engineered feature vector; DATA_MODE='raw'
gives CNN-TCN, a 1-D convolutional encoder over the waveform. Every fold holds out whole training
recordings for validation and scores the test recordings exactly once."""

import os
import glob
import json
import warnings
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr

from lib import feature_selection_core as fsc
from lib import folds as _folds
from lib.folds import N_SPLITS, SEED, subj_of

warnings.filterwarnings('ignore')

# Config
DATA_MODE = 'raw' # 'features' | 'raw' (additionally requires SAVE_RAW_EPOCHS=True in script 1.)
LABEL_MODE = 'smoothed' # 'event_locked' | 'smoothed'
EVAL_MODE = 'within_subject' # 'within_subject' (calibrated driver) | 'cross_subject' (unseen driver)
DROP_REDUNDANT_ENTROPY = False # features mode: drop svd/spec entropy (51 -> 31)
# 'all'      -- the 51 engineered features (original behaviour)
# 'selected' -- Greedy feature selection runs inside the fold loop,
FEATURE_SET = os.environ.get('CNN_FEATURE_SET', 'selected')
N_CORE = 8 # of 11 band-power features, ranked among themselves
N_EXTRA = 7 # added from entropy + wPLI; held-out test is flat from ~7 on
# Where the selection gets its data. This is a leakage question, not a convenience one. 'holdout' select
# once, on the 8 single-recording subjects only. Under within_subject those recordings can never be in a
# test set (only subjects with >=2 recordings are held out), so one fixed feature list is reused by every
# fold with no leakage. This is the default: it gives a single publishable feature set.
SELECTION_SOURCE = os.environ.get('CNN_SELECTION_SOURCE', 'holdout')
N_PARAMS = 0 # filled in by run(); recorded in the results JSON
fold_features = [] # per-fold selected feature names, recorded in the results
DATA_MODE = os.environ.get('CNN_DATA_MODE', DATA_MODE)
LABEL_MODE = os.environ.get('CNN_LABEL_MODE', LABEL_MODE)
EVAL_MODE = os.environ.get('CNN_EVAL_MODE', EVAL_MODE)
HERE = os.path.dirname(os.path.abspath(__file__))
FEATURE_DIR = os.path.join(HERE, 'eeg_data_local', 'extracted_features_labeled')
# SEQ_LEN: the GNN-LSTM used 20 (~70 s) because LSTMs degrade over long sequences (recency bias —
# the very thing its temporal self-attention was added to patch). A dilated causal TCN has stable
# gradients over long ranges, so inheriting that limit would handicap it.
SEQ_LEN = 60 # 60 epochs x 3.5 s = 3.5 min of context
STRIDE = 30 # keep 50% overlap
EMBED_DIM = 64 # CNN output per epoch
TCN_CHANNELS = 64
TCN_KERNEL = 3
# Dilations scale the reach exponentially for linear cost — the cheap way to buy long context
# (far cheaper than attention, and it keeps the model simple). Pair the RF with SEQ_LEN:
#     (1,2,4,8)         ->  61 steps ~  3.5 min   <- matches SEQ_LEN=60
#     (1,2,4,8,16)      -> 125 steps ~  7.3 min   <- use with SEQ_LEN~120
#     (1,2,4,8,16,32)   -> 253 steps ~ 14.8 min   <- use with SEQ_LEN~250
# There is no point making RF >> SEQ_LEN (the window bounds context) or << SEQ_LEN (wasted history).
# One caveat: longer windows mean fewer training windows per recording (~1,000–1,700 epochs each), so
# there's a data/context trade. The added dilations don't improve the model on this dataset.
TCN_DILATIONS = (1, 2, 4, 8)
DROPOUT = 0.3
BATCH_SIZE = 16
EPOCHS = 60
LR = 5e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
PATIENCE = 15
torch.manual_seed(SEED)
np.random.seed(SEED)

def pick_device():
    # CUDA, else Apple's GPU (MPS), else CPU.
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        # a few ops still lack MPS kernels; this lets them fall back to CPU instead of raising
        os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
        return torch.device('mps')
    return torch.device('cpu')

DEVICE = torch.device(os.environ.get('CNN_DEVICE') or pick_device())

# Data
def entropy_keep_mask(feature_names):
    if DROP_REDUNDANT_ENTROPY:
        return np.array([not n.startswith(('svd_ent_', 'spec_ent_')) for n in feature_names])
    return np.ones(len(feature_names), dtype=bool)

def load_selection_pool():
    # Preload the 141-feature pool once, for in-fold selection. None unless FEATURE_SET='selected'.
    if FEATURE_SET != 'selected':
        return None, None
    mask_key, rt_key = (('preonset_mask', 'preonset_rt') if LABEL_MODE == 'event_locked'
                        else ('smoothed_mask', 'smoothed_rt'))
    paths = sorted(glob.glob(os.path.join(FEATURE_DIR, '*.pt')))
    return fsc.load_pool(paths, mask_key, rt_key)

CORE_NAMES = ('TBR_', 'FMT_', 'ATR_', 'theta_')

def holdout_recordings(paths):
    # The single-recording subjects: permanently in train under within_subject, so selecting on them
    # leaks nothing into any test set. Under cross_subject they are tested, so that protocol should
    # use SELECTION_SOURCE='per_fold' instead.
    by_subj = defaultdict(list)
    for p in paths:
        by_subj[subj_of(p)].append(p)
    return sorted(f[0] for f in by_subj.values() if len(f) == 1)

def load_session(path, cols=None):
    # One recording as (X, y, valid), all indexed by epoch. X is whichever input the run needs: the
    # selected columns of the 141-candidate pool when cols is given, else the 51 engineered features
    # (n_ep, F) or the raw waveform (n_ep, 10, S) per DATA_MODE. y is log10 reaction time; valid marks
    # the epochs that carry a label, as most do not.
    d = torch.load(path, weights_only=False)
    if cols is not None:
        # 141 candidates: the 51 engineered features plus both wPLI triangles. Built by the
        # shared module so the saved column indices cannot drift from what was selected.
        X = fsc.build_pool(d)[0][:, cols].astype(np.float32)
    elif DATA_MODE == 'features':
        keep = entropy_keep_mask(d['feature_names'])
        X = d['features_norm'].numpy()[:, keep].astype(np.float32)
    elif DATA_MODE == 'raw':
        if 'raw_epochs' not in d:
            raise KeyError(f"{os.path.basename(path)}: no raw_epochs (set SAVE_RAW_EPOCHS=True)")
        raw = d['raw_epochs'].float().numpy() # (n_ep, 10, S), microvolts
        # Take the first 5 minutes of the drive (86 epochs), throw out the artifact-flagged ones, and
        # compute a mean and standard deviation per electrode across those epochs. Freeze those 10 pairs
        # of numbers, then use them to z-score every epoch in the recording.
        if 'norm_calib_epochs' not in d or d.get('raw_unit') != 'uV':
            raise RuntimeError(f"{os.path.basename(path)}: raw_epochs predate the microvolt fix — "
                               f"re-run scripts 1 and 2")
        calib = int(d['norm_calib_epochs'])
        clean = ~d['artifact_mask'].numpy().astype(bool)[:calib]
        win = raw[:calib][clean] if clean.sum() >= 20 else raw[:calib]
        mu = win.mean(axis=(0, 2), keepdims=True)
        sd = win.std(axis=(0, 2), keepdims=True) + 1e-3        # uV floor; negligible vs sd ~ 5 uV
        X = ((raw - mu) / sd).astype(np.float32)
    else:
        raise ValueError(DATA_MODE)

    if LABEL_MODE == 'event_locked':
        y = d['preonset_rt'].numpy().astype(np.float32)
        valid = d['preonset_mask'].numpy().astype(bool)
    elif LABEL_MODE == 'smoothed':
        y = d['smoothed_rt'].numpy().astype(np.float32)
        valid = d['smoothed_mask'].numpy().astype(bool)
    else:
        raise ValueError(LABEL_MODE)
    return X, y, valid

class SequenceDataset(Dataset):
    # Windows of SEQ_LEN consecutive epochs. Keeps windows with >=1 supervised step. Each window also
    # carries the identity of every epoch in it (which recording, which row). The windows overlap so
    # without an identity the pooled metrics score most epochs twice.
    def __init__(self, paths, cols=None):
        self.samples = []
        # si: session index — position of the recording in the paths list (0, then 1, then 2…)
        # p: path — the actual file (... / s01_051017m.set.pt)
        # n: number of epochs in this recording (1033)
        # s: start — index of the window's first epoch (0, 30, 60, …)
        # e: end — one past the last epoch, s + SEQ_LEN (60, 90, 120, …)
        # vw: valid window — valid[s:e], the 60 booleans
        for si, p in enumerate(paths):
            X, y, valid = load_session(p, cols=cols)
            n = X.shape[0]
            # one integer per (recording, epoch); 10**6 is far above any recording's epoch count
            keys = si * 10 ** 6 + np.arange(n, dtype=np.int64)
            for s in range(0, n - SEQ_LEN + 1, STRIDE):
                e = s + SEQ_LEN
                vw = valid[s:e]
                if vw.sum() == 0:
                    continue
                self.samples.append((X[s:e], y[s:e], vw, keys[s:e]))

    def __len__(self):
        return len(self.samples)

# X: the inputs, 60 × 15 features; y: the target, log₁₀ reaction time at each of the 60 steps
# valid: which of those 60 steps actually have a label; keys: the identity of each of those 60 epochs
    def __getitem__(self, i):
        X, y, v, k = self.samples[i]
        return (torch.from_numpy(X.copy()),
                torch.from_numpy(y.copy()),
                torch.from_numpy(v.copy().astype(np.float32)),
                torch.from_numpy(k.copy()))

# Model
class FeatureEncoder(nn.Module):
    # features mode: small MLP per epoch (features are an unordered vector, not a signal).
    def __init__(self, in_dim, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.LayerNorm(128), nn.ELU(), nn.Dropout(DROPOUT),
            nn.Linear(128, embed_dim), nn.LayerNorm(embed_dim), nn.ELU())

    def forward(self, x):           # x: (N, in_dim)
        return self.net(x)

class RawCNNEncoder(nn.Module):
    # raw mode: compact 1-D CNN over the 10-channel waveform ('imaging the brain').
    def __init__(self, n_ch, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_ch, 32, kernel_size=7, padding=3), nn.BatchNorm1d(32), nn.ELU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=7, padding=3), nn.BatchNorm1d(64), nn.ELU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 64, kernel_size=7, padding=3), nn.BatchNorm1d(64), nn.ELU(),
            nn.AdaptiveAvgPool1d(1))
        self.proj = nn.Sequential(nn.Linear(64, embed_dim), nn.ELU())

    def forward(self, x):           # x: (N, n_ch, S)
        h = self.net(x).squeeze(-1)  # (N, 64)
        return self.proj(h)

class CausalConv1d(nn.Module):
    # 1-D conv that only sees the present and past (left-padded: Pad both ends, then throw away the right end)
    def __init__(self, c_in, c_out, kernel, dilation):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(c_in, c_out, kernel, padding=self.pad, dilation=dilation)

    def forward(self, x):           # x: (B, C, T)
        out = self.conv(x)
        return out[:, :, :-self.pad] if self.pad > 0 else out

class TCNBlock(nn.Module):
    # Two causal convolutions at the same dilation, then the input added back. The residual
    # is what makes the stack trainable at depth: the identity path has derivative 1, so gradients
    # reach the earliest block undiminished and each block only learns a correction to its input.
    def __init__(self, c, kernel, dilation):
        super().__init__()
        self.c1 = CausalConv1d(c, c, kernel, dilation)
        self.b1 = nn.BatchNorm1d(c)
        self.c2 = CausalConv1d(c, c, kernel, dilation)
        self.b2 = nn.BatchNorm1d(c)
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x):
        h = self.drop(F.elu(self.b1(self.c1(x))))
        h = self.drop(F.elu(self.b2(self.c2(h))))
        return F.elu(x + h)          # residual

class TCNModel(nn.Module):
    # The whole model: a per-epoch encoder, a stack of dilated causal blocks over time, a per-step head,
    # ordinary linear layers applied at every timestep; a prediction comes out at every step.
    def __init__(self, mode, in_shape):
        super().__init__()
        if mode == 'features':
            self.encoder = FeatureEncoder(in_shape, EMBED_DIM)
        else:
            self.encoder = RawCNNEncoder(in_shape[0], EMBED_DIM)   # in_shape = (n_ch, S)
        self.in_proj = nn.Conv1d(EMBED_DIM, TCN_CHANNELS, 1)
        self.tcn = nn.Sequential(*[TCNBlock(TCN_CHANNELS, TCN_KERNEL, d) for d in TCN_DILATIONS])
        self.head = nn.Sequential(
            nn.Conv1d(TCN_CHANNELS, 32, 1), nn.ELU(), nn.Dropout(0.1), nn.Conv1d(32, 1, 1))

# B	batch — how many windows at once	T time — epochs per window
# F	features per epoch	    E embedding — encoder output per epoch
# C	channels inside the TCN	           n_ch, S electrodes, samples
    def forward(self, x):
        # x: features (B, T, F) or raw (B, T, n_ch, S)
        B, T = x.shape[0], x.shape[1]
        flat = x.reshape(B * T, *x.shape[2:])
        emb = self.encoder(flat).reshape(B, T, EMBED_DIM)        # (B, T, E)
        seq = emb.transpose(1, 2)                                # (B, E, T)
        h = self.in_proj(seq)
        h = self.tcn(h)                                          # (B, C, T) causal
        out = self.head(h).transpose(1, 2)                       # (B, T, 1)
        return out

def masked_mse(pred, target, valid):
    # pred,target,valid: (B, T[,1]); loss only where valid>0.5
    pred = pred.squeeze(-1)
    diff2 = (pred - target) ** 2 * valid
    denom = valid.sum().clamp(min=1.0)
    return diff2.sum() / denom

# Results i/o
def model_name():
    return 'MLP-TCN' if DATA_MODE == 'features' else 'CNN-TCN'

def results_path():
    tag = f"{model_name()}_{LABEL_MODE}_{EVAL_MODE}"
    # The 15-feature selected set is the default, so it needs no suffix; deviations get one.
    if FEATURE_SET != 'selected':
        tag += '_all51'
    elif (N_CORE, N_EXTRA) != (8, 7):
        tag += f"_sel{N_CORE}+{N_EXTRA}"
    if FEATURE_SET == 'selected' and SELECTION_SOURCE != 'holdout':
        tag += f"_{SELECTION_SOURCE}"
    return os.path.join(HERE, 'eeg_data_local', f'results_{tag}.json')

def save_results(all_results, complete):
    # Write the results file. Called after every fold so an interrupted run keeps what it earned.
    payload = {
        'config': {'data_mode': DATA_MODE, 'label_mode': LABEL_MODE, 'eval_mode': EVAL_MODE,
                   'seq_len': SEQ_LEN, 'stride': STRIDE, 'dilations': list(TCN_DILATIONS),
                   'n_splits': N_SPLITS, 'n_params': N_PARAMS, 'seed': SEED,
                   'feature_set': FEATURE_SET,
                   'n_core': N_CORE if FEATURE_SET == 'selected' else None,
                   'n_extra': N_EXTRA if FEATURE_SET == 'selected' else None,
                   'selection_source': SELECTION_SOURCE if FEATURE_SET == 'selected' else None,
                   'fold_features': fold_features or None,
                   'device': str(DEVICE),
                   'complete': bool(complete), 'folds_done': len(all_results)},
        'folds': all_results,
    }
    path = results_path()
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)
    if not complete:
        print(f"  [checkpoint] {len(all_results)}/{N_SPLITS} folds saved")

# Folds
def build_folds(paths):
    return _folds.build_folds(paths, EVAL_MODE)

def evaluate(model, loader, return_preds=False):
    # Pooled r and MAE over each supervised epoch, counted once. Overlapping windows give most epochs
    # more than one prediction. Where that happens the one kept is the prediction made from the latest
    # window position so with the most history behind it, as a live monitor would have at that moment.
    model.eval()
    # P: Predictions from the model	0.41, 0.38, … Tt: True targets, the actual log₁₀ RT	0.35, 0.44, …
    # K: Keys — which epoch this is 45, 46, … Pos: Position of that epoch inside its window (0–59)
    P, Tt, K, Pos = [], [], [], []
    with torch.no_grad():
        for X, y, v, k in loader:
            X = X.to(DEVICE)
            pred = model(X).squeeze(-1).cpu().numpy()
            y = y.numpy(); v = v.numpy() > 0.5; k = k.numpy()
            pos = np.broadcast_to(np.arange(pred.shape[1]), pred.shape)
            P.append(pred[v]); Tt.append(y[v]); K.append(k[v]); Pos.append(pos[v])
    P = np.concatenate(P) if P else np.array([])
    Tt = np.concatenate(Tt) if Tt else np.array([])
    if P.size:
        K = np.concatenate(K); Pos = np.concatenate(Pos)
        order = np.lexsort((Pos, K))              # by epoch, then window position ascending
        K, P, Tt = K[order], P[order], Tt[order]
        keep = np.ones(K.size, dtype=bool)
        keep[:-1] = K[1:] != K[:-1]               # last row of each epoch = deepest window position
        P, Tt = P[keep], Tt[keep]
    r = pearsonr(P, Tt)[0] if len(P) > 1 and np.std(P) > 0 else 0.0
    mae = float(np.mean(np.abs(P - Tt))) if len(P) else float('nan')
    out = {'pearson_r': float(r), 'mae': mae}
    # Off by default: this runs after every training epoch just to read one number, and the
    # arrays are ~7 MB. Switched on for the single end-of-fold test pass, whose predictions
    # feed the predicted-vs-true scatter and any later calibration analysis.
    if return_preds:
        out['pred'] = P.tolist(); out['true'] = Tt.tolist()
    return out

def per_session_r(model, test_paths, cols=None):
    rs = []
    for p in test_paths:
        ds = SequenceDataset([p], cols=cols)
        if len(ds) == 0:
            continue
        ld = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
        r = evaluate(model, ld)['pearson_r']
        if not np.isnan(r):
            rs.append(r)
    return rs

# Train / cv
def run():
    paths = sorted(glob.glob(os.path.join(FEATURE_DIR, '*.pt')))
    if not paths:
        raise SystemExit(f"No .pt in {FEATURE_DIR} — run 1_ and 2_ first.")
    # infer input shape
    if FEATURE_SET == 'selected' and DATA_MODE != 'features':
        raise SystemExit("FEATURE_SET='selected' only applies to DATA_MODE='features'")
    # Under cross_subject the single-recording subjects are tested too, so selecting on them
    # leaks. Refuse rather than silently produce an optimistic number.
    if (FEATURE_SET == 'selected' and EVAL_MODE == 'cross_subject'
            and SELECTION_SOURCE == 'holdout'):
        raise SystemExit("cross_subject requires CNN_SELECTION_SOURCE=per_fold: the holdout "
                         "subjects appear in test sets under this protocol")
    sel_store, sel_names = load_selection_pool()
    fixed_cols = None
    if FEATURE_SET == 'selected':
        in_shape = N_CORE + N_EXTRA
        # SELECTION_SOURCE decides which recordings the feature selection is allowed to look at when
        # picking the 15 features. It's a leakage question. 'holdout' means: run the greedy selection
        # once, on the 8 single-recording subjects only. Because they can never appear in a test set.
        # The alternative, 'per_fold', re-runs the selection inside each fold using only that fold's
        # train+val recordings.
        if SELECTION_SOURCE == 'holdout':
            hold = holdout_recordings(paths)
            fixed_cols, core_ranked = fsc.select_for_holdout(hold, sel_store, sel_names, N_CORE, N_EXTRA)
            core_sel = [sel_names[j] for j in core_ranked[:N_CORE]]
            added = [sel_names[j] for j in fixed_cols if sel_names[j] not in core_sel]
            print(f"FEATURE_SET=selected  source=holdout  "
                  f"({len(hold)} single-recording subjects, never in a within_subject test set)")
            print(f"  core  ({N_CORE}): {core_sel}")
            print(f"  added ({N_EXTRA}): {added}")
        else:
            print(f"FEATURE_SET=selected  source=per_fold  {N_CORE}+{N_EXTRA} features, "
                  f"re-selected inside every fold from its own train+val")
    else:
        X0, _, _ = load_session(paths[0])
        in_shape = X0.shape[1] if DATA_MODE == 'features' else (X0.shape[1], X0.shape[2])
    global N_PARAMS, fold_features
    # a throwaway model, built only to count parameters for the results file
    N_PARAMS = sum(p.numel() for p in TCNModel(DATA_MODE, in_shape).parameters() if p.requires_grad)
    print(f"DATA_MODE={DATA_MODE}  LABEL_MODE={LABEL_MODE}  EVAL_MODE={EVAL_MODE}")
    print(f"in_shape={in_shape}  params={N_PARAMS:,}  device={DEVICE}")
    print(f"{len(paths)} recordings\n")

    folds = build_folds(paths)          # the 5 train/val/test splits, deterministic given SEED
    all_results = []
    fold_features = []          # what each fold actually trained on, recorded in the results
    for fi, (train, val, test) in enumerate(folds):
        # which feature columns this fold trains on; None means the full vector
        cols = None
        if FEATURE_SET == 'selected':
            if SELECTION_SOURCE == 'holdout':
                cols = fixed_cols            # one list, chosen once, reused by every fold
            else:
                cols, core_ranked = fsc.select_for_fold(train, val, sel_store, sel_names, N_CORE, N_EXTRA)
                print(f"  re-selected for this fold: "
                      f"{[sel_names[j] for j in cols if not sel_names[j].startswith(CORE_NAMES)]}")
            fold_features.append([sel_names[j] for j in cols])
        print(f"--- Fold {fi + 1}/{N_SPLITS}: train={len(train)} val={len(val)} test={len(test)}"
              + (f" feats={len(cols)} ---" if cols else " ---"))
        # shuffle the training windows only; val and test keep recording order for scoring
        tr_loader = DataLoader(SequenceDataset(train, cols=cols), batch_size=BATCH_SIZE, shuffle=True)
        va_loader = DataLoader(SequenceDataset(val, cols=cols), batch_size=BATCH_SIZE, shuffle=False)
        te_loader = DataLoader(SequenceDataset(test, cols=cols), batch_size=BATCH_SIZE, shuffle=False)

        # a fresh model per fold: nothing carries over from the previous fold's training
        model = TCNModel(DATA_MODE, in_shape).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR * 0.01)

        # early-stopping bookkeeping: best score so far, the weights that scored it, epochs since
        best_val_r, best_state, stale, best_epoch = -np.inf, None, 0, 0
        history = []               # per-epoch curve, kept for the training-curve figure
        for ep in range(1, EPOCHS + 1):
            model.train()
            tot, nb = 0.0, 0
            for X, y, v, _ in tr_loader:      # epoch keys matter only at scoring time
                X, y, v = X.to(DEVICE), y.to(DEVICE), v.to(DEVICE)
                loss = masked_mse(model(X), y, v)     # unlabelled steps contribute nothing
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)   # cap exploding updates
                opt.step()
                tot += loss.item(); nb += 1
            sched.step()                              # cosine decay, once per epoch not per batch
            vm = evaluate(model, va_loader)            # score on validation, never on test
            history.append({'epoch': ep, 'train_loss': tot / max(nb, 1),
                            'val_r': vm['pearson_r'], 'lr': float(sched.get_last_lr()[0])})
            if vm['pearson_r'] > best_val_r + 1e-4:
                # stash a cpu copy of the best weights: validation usually peaks well before training
                # stops, and it is these weights that get tested, not the last epoch's
                best_val_r = vm['pearson_r']; best_epoch = ep; stale = 0
                best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
            else:
                stale += 1
            if ep % 5 == 0 or stale == 0:
                print(f"  ep {ep:3d} | loss {tot / max(nb, 1):.4f} | val r {vm['pearson_r']:.3f} "
                      f"| best {best_val_r:.3f}")
            if stale >= PATIENCE: # 15 epochs with no validation improvement: stop and keep best_state
                print(f"  early stop @ {ep} (best val r {best_val_r:.3f} @ {best_epoch})")
                break

        if best_state is not None:
            model.load_state_dict(best_state)            # restore best-on-validation weights
        # score test exactly once
        m = evaluate(model, te_loader, return_preds=True)
        m['fold'] = fi
        m['best_val_r'] = float(best_val_r)
        m['best_epoch'] = best_epoch
        m['history'] = history
        psr = per_session_r(model, test, cols=cols)
        if psr:
            m['per_session_r'] = [float(x) for x in psr]
            m['per_session_r_mean'] = float(np.mean(psr))
            m['per_session_r_median'] = float(np.median(psr))
        all_results.append(m)
        print(f"  TEST pooled r={m['pearson_r']:.3f}  MAE={m['mae']:.3f}  "
              f"per-session mean={m.get('per_session_r_mean', float('nan')):.3f}\n")
        # Save after every fold, not just at the end. A raw-mode run is long enough that losing it
        # to an interruption at fold 5 costs hours; the file is rewritten completely each time, so a
        # partial file is simply a run with fewer folds and downstream code handles it unchanged.
        save_results(all_results, complete=(fi + 1 == len(folds)))

    # summary
    pr = [r['pearson_r'] for r in all_results]
    psm = [r['per_session_r_mean'] for r in all_results if 'per_session_r_mean' in r]
    mae = [r['mae'] for r in all_results]
    print("=" * 56)
    print(f"{model_name()}  DATA_MODE={DATA_MODE}  LABEL_MODE={LABEL_MODE}  EVAL_MODE={EVAL_MODE}")
    print(f"  Pooled r        : {np.mean(pr):.3f} ± {np.std(pr):.3f}")
    print(f"  Per-session r   : {np.mean(psm):.3f}" if psm else "  Per-session r   : n/a")
    print(f"  MAE (log10 RT)  : {np.mean(mae):.3f}")
    print(f"  saved -> {results_path()}")
    return all_results

if __name__ == '__main__':
    run()
