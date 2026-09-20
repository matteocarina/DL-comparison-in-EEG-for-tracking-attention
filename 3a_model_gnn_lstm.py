"""Graph baseline: a GAT over the 10-electrode connectivity graph feeding an LSTM.
Each epoch becomes a graph whose nodes carry the engineered features and whose edges carry the theta
and alpha wPLI. Node-attention pooling gives one embedding per epoch, an LSTM with temporal
attention reads the sequence, and a head predicts log-RT at every step."""

# Two architectures, chosen by ARCHITECTURE:
#   'GNN'      GAT over one epoch's connectivity graph -> node-attention pooling -> regression head.
#              The purely spatial baseline: does pre-onset connectivity alone predict that trial's RT?
#   'GNN_LSTM' the same GAT encoder, then the sequence of graph embeddings through an LSTM with
#              temporal self-attention. Adds the slow attention cycle across epochs.
# Both share the GAT and pooling code, so the only difference between them is the temporal half.
# Node features are assigned anatomically: each node (electrode) receives only the spectral and
# entropy features measured at that electrode, so attention weights over nodes stay interpretable.
# A random-forest importance ranking picks the features, refit inside every fold.

import os
import glob
import numpy as np

from lib.folds import subj_of
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data, Batch
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
import json

DEBUG_TRAIN        = False
DEBUG_TRAIN_FOLDS  = 1
DEBUG_TRAIN_EPOCHS = 50

# Architecture
ARCHITECTURE = 'GNN_LSTM'
#   'cross_subject'  no subject appears in both train and test: generalisation to an unseen driver.
#   'within_subject' every test session belongs to a driver who also has a session in train: the
#                    per-driver calibration scenario. Split across whole sessions, never across
#                    epochs of one session, since adjacent 4 s epochs are near-duplicates.
# Single-session subjects cannot be held out within-subject, so they stay permanently in train.
EVAL_MODE = 'within_subject'  # 'cross_subject' or 'within_subject'
#   'event_locked' — the sparse per-trial target (preonset_mask / preonset_rt). The original.
#   'smoothed'     — the dense 60 s Gaussian-smoothed envelope (smoothed_mask / smoothed_rt),
#                    added to the .pt files by 2_label_extraction.py. Run that first.
LABEL_MODE = 'event_locked'   # 'event_locked' or 'smoothed'
EVAL_MODE = os.environ.get('GNN_EVAL_MODE', EVAL_MODE)
ARCHITECTURE = os.environ.get('GNN_ARCHITECTURE', ARCHITECTURE)
LABEL_MODE = os.environ.get('GNN_LABEL_MODE', LABEL_MODE)
LABEL_KEYS = {'event_locked': ('preonset_mask', 'preonset_rt'),
              'smoothed': ('smoothed_mask', 'smoothed_rt')}[LABEL_MODE]
# The extraction stores all 4 entropy families per channel (51 features). svd_ent and spec_ent
# correlate ~0.8 with samp_ent and add little, so this drops them to 31.
DROP_REDUNDANT_ENTROPY = False
FEATURE_DIR = os.environ.get('GNN_FEATURE_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)),
            'eeg_data_local', 'extracted_features_labeled'))
FRONTAL_CH = ['f3', 'f4', 'fz', 'fcz']  # node indices 0-3
PARIETAL_CH = ['pz', 'p3', 'p4']  # node indices 4-6
OCCIPITAL_CH = ['o1', 'oz', 'o2']  # node indices 7-9
ALL_INTEREST = FRONTAL_CH + PARIETAL_CH + OCCIPITAL_CH  # 10 GNN nodes
N_NODES = 10
EDGE_FEAT_DIM = 2  # [theta_wPLI, alpha_wPLI]
WPLI_THRESH = 0.10  # minimum wPLI to keep an edge
# GAT
N_HEADS = 4
GAT_HIDDEN = 32  # output features per head per layer
GAT_LAYERS = 2
GRAPH_EMBED = 64  # graph-level embedding size after pooling
# LSTM
LSTM_HIDDEN = 128
LSTM_LAYERS = 2
SEQ_LEN = 20  # consecutive epochs per sequence (~40 s)
LR = 5e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 60
# A GNN-LSTM batch of 16 sequences is 16 x SEQ_LEN graphs, while a GNN batch of 16 is 16
# graphs. Scaling the spatial-only batch by SEQ_LEN puts the same number of graphs behind
# each optimiser step in both modes, so they differ in architecture and nothing else.
BATCH_SIZE = 16
GNN_BATCH_SIZE = BATCH_SIZE * SEQ_LEN
GRAD_CLIP = 1.0
# Feature selection. Drop this fraction of features by RF importance (computed on training only)
RF_DROP_FRACTION = 0.2  # drop bottom 20% the least important features. 20% and not 70% because the
# GNN's features are assigned anatomically and some electrodes wouldn't have any features.
# Cross-validation. Number of grouped-by-subject folds.
N_SPLITS = 5

# anatomical node feature assignment
def build_feature_name_to_node_map(feature_names):
    # Map each feature to the node index of the electrode it was measured at, so attention weights
    # over nodes reflect real electrode contributions. A feature naming no electrode raises.
    ch_to_node = {ch.lower(): i for i, ch in enumerate(ALL_INTEREST)}
    assignments = np.zeros(len(feature_names), dtype=int)
    for fi, name in enumerate(feature_names):
        name_lower = name.lower()
        node = next((i for ch, i in ch_to_node.items() if ch in name_lower), None)
        if node is None and 'fmt' in name_lower:
            node = ch_to_node['fz']          # frontal-midline theta: averaged over fz and fcz
        if node is None:
            raise ValueError(f"{name!r} names no electrode, so it belongs to no node. Every "
                             f"feature must be attributable to one electrode for the attention "
                             f"weights over nodes to mean anything.")
        assignments[fi] = node
    return assignments

def assign_features_to_nodes(feature_vec, node_assignments, feat_mask, n_nodes=N_NODES):
    # Assign selected features to their anatomically correct nodes. Each node receives only the features
    # extracted from its electrode. Nodes are zero-padded to the same length so the GNN receives a
    # regular (N_NODES, max_features_per_node) tensor.
    node_feats = [[] for _ in range(n_nodes)]
    for fi, val in enumerate(feature_vec):
        node_idx = int(node_assignments[fi])
        node_feats[node_idx].append(float(val))
    # Pad to equal length across nodes; every node carries at least one feature
    max_len = max(len(f) for f in node_feats)
    x = np.zeros((n_nodes, max_len), dtype=np.float32)
    # The loop iterates over each node and its corresponding feature list. x[node_idx, :len(feats)] =
    # feats fills the first len(feats) elements of the row node_idx with the feature values from feats.
    for node_idx, feats in enumerate(node_feats):
        x[node_idx, :len(feats)] = feats
    return x

# fast RF feature selection
def rf_feature_selection(X_train, y_train, drop_fraction=RF_DROP_FRACTION):
    # Select features by Random Forest importance ranking. Trains one RF regressor on all training data,
    # ranks features by mean impurity decrease, and drops the bottom `drop_fraction`.
    # n_jobs=-1 uses all available CPU cores; random_state fixes reproducibility.
    clf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    importances = clf.feature_importances_
    threshold = np.percentile(importances, drop_fraction * 100)
    feat_mask = importances >= threshold
    n_kept = int(feat_mask.sum())
    print(f"  RF feature selection: kept {n_kept}/{len(feat_mask)} features "
          f"(dropped bottom {drop_fraction * 100:.0f}%)")
    return feat_mask

# graph construction
def build_graph(node_x, wpli_theta, wpli_alpha, threshold=WPLI_THRESH):
    # Build a PyG Data object for one epoch. node_x: ndarray (N_NODES, node_feat_dim);
    # wpli_theta/alpha: ndarray (N_NODES, N_NODES)
    n = node_x.shape[0]
    src, dst, attrs = [], [], [] # Source, destination, edge attributes
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            tw = float(wpli_theta[i, j])
            aw = float(wpli_alpha[i, j])
            if max(tw, aw) > threshold:
                src.append(i)
                dst.append(j)
                attrs.append([tw, aw])
    if not src:
        raise ValueError(f"no wPLI pair exceeds {threshold}: this epoch has no graph edges")
    return Data(x=torch.tensor(node_x, dtype=torch.float), edge_index=torch.tensor([src, dst], dtype=torch.long),
        edge_attr=torch.tensor(attrs, dtype=torch.float), )

# datasets
class EpochGraphDataset(Dataset):
    # GNN-only dataset. One sample = one epoch graph.
    def __init__(self, pt_paths, feat_mask, node_assignments):
        self.samples = []
        for path in pt_paths:
            d = torch.load(path, weights_only=False)
            feats = d['features_norm'].numpy()[:, feat_mask]
            wt = d['wpli_theta'].numpy()
            wa = d['wpli_alpha'].numpy()
            # aligned: the GNN-only baseline is purely event-locked. Non-pre-onset epochs carry no supervision
            # and are simply not turned into samples here (there is no temporal context in GNN-only mode.
            mask_key, rt_key = LABEL_KEYS
            if mask_key not in d or rt_key not in d:
                raise KeyError(f"{path} is missing {mask_key}/{rt_key} — run "
                               f"2_label_extraction.py first.")
            pre = d[mask_key].numpy().astype(bool)
            rt = d[rt_key].numpy()
            # Build one graph per pre-onset epoch only.
            for i in np.nonzero(pre)[0]:
                x = assign_features_to_nodes(feats[i], node_assignments, feat_mask)
                g = build_graph(x, wt[i], wa[i])
                # Each sample = one pre-onset epoch graph + that trial's true log-RT.
                self.samples.append((g, float(rt[i])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_epoch_graphs(batch):
    # A helper function used with PyTorch's DataLoader to combine individual samples from your
    # EpochGraphDataset into a batch for efficient processing. Batch.from_data_list is a PyTorch
    # Geometric function that combines a list of individual graphs into a single batched graph. The
    # function returns a tuple (graphs, scores, labels).
    graphs = Batch.from_data_list([b[0] for b in batch])
    scores = torch.tensor([b[1] for b in batch], dtype=torch.float)
    return graphs, scores

class SequenceGraphDataset(Dataset):
    # GNN+LSTM dataset. One sample = SEQ_LEN consecutive epoch graphs. Sequences are built with 50%
    # overlap (stride = SEQ_LEN // 2)
    def __init__(self, pt_paths, feat_mask, node_assignments, seq_len=SEQ_LEN):
        self.sequences = []
        for _si, path in enumerate(pt_paths):
            d = torch.load(path, weights_only=False)
            feats = d['features_norm'].numpy()[:, feat_mask]
            wt = d['wpli_theta'].numpy()
            wa = d['wpli_alpha'].numpy()
            # Every epoch becomes a graph so the LSTM gets full temporal context, but `valid` fires only
            # at supervised epochs, where the loss and metrics apply. The mask already excludes artifacts.
            mask_key, rt_key = LABEL_KEYS
            if mask_key not in d or rt_key not in d:
                raise KeyError(f"{path} is missing {mask_key}/{rt_key} — run "
                               f"2_label_extraction.py first.")
            scores = d[rt_key].numpy() # log-RT at the supervised epochs
            valid = d[mask_key].numpy().astype(np.float32)  # supervise only where the mask fires
            n_ep = len(feats) # = total number of epochs graphs
            stride = max(1, seq_len // 2) # seq_len set as 20 epochs for 40sec
            keys = _si * 10 ** 6 + np.arange(n_ep, dtype=np.int64)
            for start in range(0, n_ep - seq_len + 1, stride):
                end = start + seq_len
                graphs = []
                for i in range(start, end): # Iterates over every epoch of the stride
                    x = assign_features_to_nodes(feats[i], node_assignments, feat_mask)
                    graphs.append(build_graph(x, wt[i], wa[i]))
                # (list of graphs + per-epoch continuous targets + per-epoch validity mask) to self.sequences
                self.sequences.append((graphs, torch.tensor(scores[start:end], dtype=torch.float),
                torch.tensor(valid[start:end], dtype=torch.float), torch.from_numpy(keys[start:end])))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]

def collate_sequences(batch):
    # Now also carries the per-timestep validity mask so artifact epochs can be excluded
    # from the loss/metrics while the sequence itself stays time-contiguous.
    all_graphs, scores_list, valid_list, key_list, seq_lens = [], [], [], [], []
    for graphs, scores, valid, keys in batch:
        all_graphs.extend(graphs)
        scores_list.append(scores)
        valid_list.append(valid)
        key_list.append(keys)
        seq_lens.append(len(graphs)) # Each sequence can potentially have a different length in a more general case
    return (Batch.from_data_list(all_graphs), torch.stack(scores_list), torch.stack(valid_list),
            seq_lens, torch.stack(key_list))

# model components
class NodeAttentionPooling(nn.Module):
    # Learns a scalar gate per node so the pooled graph embedding dynamically weights electrode
    # contributions. Frontal theta nodes should dominate during high cognitive load; parietal alpha nodes
    # during relaxed alertness. This mechanism learns that weighting automatically from the training data.
    def __init__(self, in_dim, hidden=32):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh(), nn.Linear(hidden, 1),)

    def forward(self, x, batch):
        # A numerical stability trick done to prevent numerical overflow in the next step.
        gates = self.gate(x) - self.gate(x).max()
        exp_g = gates.exp()
        # batch is a tensor that indicates which graph each node belongs to. batch.max() finds the
        # maximum value in the batch tensor, .item() converts this element tensor to a Python integer.
        n_graphs = int(batch.max().item()) + 1
        # This initializes a tensor to store the sum of the exponentiated gates for each graph.
        sum_exp = torch.zeros(n_graphs, 1, device=x.device)
        sum_exp.scatter_add_(0, batch.unsqueeze(1), exp_g)
        # sum_exp[batch] selects the sum of exponentiated gates for the graph that each node belongs to.
        weights = exp_g / (sum_exp[batch] + 1e-9)
        out = torch.zeros(n_graphs, x.shape[1], device=x.device)
        # Computes the weighted sum of node features for each graph, using the attention weights.
        # batch.unsqueeze(1).expand_as(x)  creates a tensor of the same shape as x, where each element
        # is the graph index for the corresponding node.
        out.scatter_add_(0, batch.unsqueeze(1).expand_as(x), weights * x)
        return out  # (n_graphs, in_dim)
# Now we have batches of graphs, where each graph represents an epoch and has node features that have
# been assigned to their anatomically correct nodes and weighted.

class GATEncoder(nn.Module):
    # Input: PyG Batch; Output: (n_graphs, GRAPH_EMBED)
    def __init__(self, node_feat_dim):
        super().__init__()
        gat_in = GAT_HIDDEN * N_HEADS # input dimension for the first GAT layer.
        self.node_proj = nn.Sequential(nn.Linear(node_feat_dim, gat_in), nn.LayerNorm(gat_in), nn.ELU(),)
        self.gat_layers = nn.ModuleList()
        self.bns = nn.ModuleList()  # Track batch norms per layer
        # Creates our 2 GAT layers
        for layer in range(GAT_LAYERS):
            # For all layers except the last one, concat is True, meaning the output features from each
            # head are concatenated. For the last layer, the output features from each head are averaged.
            concat = (layer < GAT_LAYERS - 1)
            out_each = GAT_HIDDEN # number of output features per head in the GAT layer
            self.gat_layers.append(GATv2Conv(gat_in, out_each, heads=N_HEADS, concat=concat, edge_dim=EDGE_FEAT_DIM, dropout=0.2))
            out_dim = out_each * N_HEADS if concat else out_each
            self.bns.append(nn.BatchNorm1d(out_dim))  # Track layer output shape
            # Updates gat_in for the next layer. If concat is True, the output dimension for the next
            # layer is out_each * N_HEADS. Otherwise, it's out_each (32).
            gat_in = out_each * N_HEADS if concat else out_each
        self.pool = NodeAttentionPooling(gat_in, hidden=32)
        self.proj = nn.Sequential(nn.Linear(gat_in, GRAPH_EMBED), nn.LayerNorm(GRAPH_EMBED), nn.ELU(), nn.Dropout(0.2),)

    def forward(self, batch_graph):
        x = batch_graph.x # Extracts the node features from the input batch of graphs.
        ei = batch_graph.edge_index
        ea = batch_graph.edge_attr
        bv = batch_graph.batch # Extracts the batch vector, indicating which graph each node belongs to
        x = self.node_proj(x) # Projects the input node features into a higher-dimensional space.
        for gat, bn in zip(self.gat_layers, self.bns): # Processes the node features through each GAT layer.
            x = gat(x, ei, edge_attr=ea)
            x = bn(x)
            x = F.elu(x)
        return self.proj(self.pool(x, bv))

# Summary: After processing node features through GAT layers, GATEncoder passes the updated node features
# (x) and batch indices (bv) to NodeAttentionPooling which computes attention weights for nodes and
# aggregates them into graph-level embeddings. These embeddings are then projected into the final
# embedding space (GRAPH_EMBED). Workflow: Node features → GAT layers → NodeAttentionPooling →
# Graph embeddings → Final projection → Used as input by reg and cls heads:

def make_head(in_dim):
    # Single regression head -> continuous log-RT target. The target is log10(RT) in real units (can be
    # negative), so the output must be linear/unbounded. A Sigmoid would crush it into [0,1].
    return nn.Sequential(nn.Linear(in_dim, 32), nn.ELU(), nn.Dropout(0.1), nn.Linear(32, 1))

class GNNModel(nn.Module): # Defines a model for single-epoch classification/regression.
    def __init__(self, node_feat_dim):
        super().__init__()
        self.encoder = GATEncoder(node_feat_dim)
        self.reg_head = make_head(GRAPH_EMBED)

    def forward(self, batch_graph, seq_lengths=None):
        emb = self.encoder(batch_graph)
        return self.reg_head(emb)

class TemporalSelfAttention(nn.Module):
    # It asks "which moment in the past 40 seconds matters most for predicting attention right now".
    def __init__(self, embed_dim, n_heads=4, dropout=0.2):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        out, _ = self.attn(x, x, x)
        out = self.attn_dropout(out)
        # Adds the input x to the attention output (out) and returns the normalized output tensor.
        return self.norm(x + out)

class GNNLSTMModel(nn.Module): # Defines a spatiotemporal model that combines GAT, LSTM, and temporal self-attention.
    def __init__(self, node_feat_dim):
        super().__init__()
        self.encoder = GATEncoder(node_feat_dim)
        self.lstm = nn.LSTM(GRAPH_EMBED, LSTM_HIDDEN, LSTM_LAYERS, batch_first=True, dropout=0.3 if LSTM_LAYERS > 1 else 0.0)
        self.temp_attn = TemporalSelfAttention(LSTM_HIDDEN)
        self.post_attn_dropout = nn.Dropout(0.3)
        self.reg_head = make_head(LSTM_HIDDEN)

    def forward(self, batch_graph, seq_lengths):
        emb = self.encoder(batch_graph)
        T = seq_lengths[0]
        B = emb.shape[0] // T
        seq = emb.reshape(B, T, GRAPH_EMBED)
        lstm_out, _ = self.lstm(seq)
        attended = self.temp_attn(lstm_out)
        attended = self.post_attn_dropout(attended)
        return self.reg_head(attended)

# loss and evaluation
def compute_loss(score_pred, score_target, valid_mask=None):
    # Pure MSE on the continuous log-RT target (regression only). The GNN path passes no mask because
    # it already dropped artifact epochs at dataset-build time.
    if score_pred.dim() == 3:
        # B: Batch size (number of sequences); T: Sequence length; _: feature dim (1 for regression)
        B, T, _ = score_pred.shape
        sp = score_pred.squeeze(-1).reshape(B * T)
        st = score_target.reshape(B * T)
        vm = valid_mask.reshape(B * T) if valid_mask is not None else None
    else: # GNNModel case: predictions are per-graph (2D), no reshaping needed.
        sp = score_pred.squeeze(-1)
        st = score_target
        vm = valid_mask
    if vm is not None:
        keep = vm > 0.5
        if keep.sum() == 0:
            # Under event_locked supervision ~8% of sequences contain no supervised step at all,
            # and DataLoader does not drop the final partial batch so a batch of one can be
            # entirely unsupervised. mse_loss on an empty tensor returns NaN, which would propagate
            # through backward and destroy the weights. Return a real zero that still carries grad.
            return sp.sum() * 0.0
        return F.mse_loss(sp[keep], st[keep])
    return F.mse_loss(sp, st)

def evaluate(model, loader, device, is_sequence, return_preds=False):
    # Pooled r and MAE over each supervised epoch, counted once.
    model.eval()
    # sp_all: score predictions; st_all: score true; k_all:	epoch keys; pos_all: position within the sequence (0–19)
    sp_all, st_all, k_all, pos_all = [], [], [], []
    with torch.no_grad(): # Disables gradient computation to save memory and speed up evaluation.
        for batch in loader:
            if is_sequence: # Handles the case for GNNLSTMModel
                graphs, scores, valid, seq_lens, keys = batch
                graphs = graphs.to(device)
                sp = model(graphs, seq_lens)
                # Removes the last dimension (size 1), moves the tensor to CPU, converts to NumPy, and flattens it to 1D.
                sp = sp.squeeze(-1).cpu().numpy().ravel()
                st = scores.numpy().ravel()
                keep = valid.numpy().ravel() > 0.5
                kk = keys.numpy().ravel()[keep]
                pos = np.broadcast_to(np.arange(scores.shape[1]), scores.shape).ravel()[keep]
                sp, st = sp[keep], st[keep]
                k_all.append(kk); pos_all.append(pos)
            else:
                graphs, scores = batch
                graphs = graphs.to(device)
                sp = model(graphs)
                sp = sp.squeeze(-1).cpu().numpy()
                st = scores.numpy()
            sp_all.append(sp)
            st_all.append(st)
    sp = np.concatenate(sp_all)
    st = np.concatenate(st_all)
    if is_sequence and k_all:
        kk = np.concatenate(k_all); pos = np.concatenate(pos_all)
        order = np.lexsort((pos, kk))            # by epoch, then position within the sequence
        kk, sp, st = kk[order], sp[order], st[order]
        last = np.ones(kk.size, dtype=bool)
        last[:-1] = kk[1:] != kk[:-1]            # last row per epoch = deepest sequence position
        sp, st = sp[last], st[last]
    r = pearsonr(sp, st)[0] if np.std(sp) > 0 else 0.0
    out = {'pearson_r': float(r), 'mae': float(np.mean(np.abs(sp - st)))}
    if return_preds:
        out['pred'] = sp.tolist()
        out['true'] = st.tolist()
    return out

def build_eval_folds(mode, subjects, subj_to_files, pt_files, n_splits):
    # The train/test splits for one protocol which is what makes cross and within-subject scores comparable.
    folds = []
    if mode == 'within_subject':
        # Per-driver calibration: every test session belongs to a subject who also has a session in
        # train. Only subjects with >=2 sessions can be held out this way; the rest stay in train.
        multi = {s: sorted(f) for s, f in subj_to_files.items() if len(f) >= 2}
        if not multi:
            return [], 'within-subject (NO multi-session subjects — impossible)', 0
        test_by_fold = [[] for _ in range(n_splits)]
        for s, files in sorted(multi.items()):
            for k in range(n_splits):
                test_by_fold[k].append((s, files[k % len(files)]))
        for k in range(n_splits):
            test_pairs = test_by_fold[k]
            if not test_pairs:
                continue
            test_paths = [p for (_, p) in test_pairs]
            train_paths = [p for p in pt_files if p not in test_paths]
            tested = sorted({s for (s, _) in test_pairs})
            folds.append((train_paths, test_paths, f'{len(tested)} subj / {len(test_paths)} sess'))
        cv_name = (f'{len(folds)}-fold WITHIN-SUBJECT (calibration; test subjects also in train; '
                   f'{len(multi)}/{len(subjects)} subjects have >=2 sessions)')
        return folds, cv_name, len(multi)
    # default: Cross-subject grouped k-fold — no subject ever appears on both sides.
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for _, test_idx in kf.split(subjects):
        test_subjects = [subjects[i] for i in test_idx]
        test_paths = [p for s in test_subjects for p in subj_to_files[s]]
        train_paths = [p for p in pt_files if p not in test_paths]
        folds.append((train_paths, test_paths, ','.join(test_subjects)))
    cv_name = 'LOSO' if n_splits == len(subjects) else f'{n_splits}-fold grouped-by-subject (cross-subject)'
    return folds, cv_name, len(subjects)

def run_cross_validation(feature_dir=FEATURE_DIR, architecture=ARCHITECTURE, device_str='cpu'):
    # The whole experiment: build the folds, then per fold select features, train, and score the
    # held-out sessions. Writes one results JSON at the end, under the TCN's naming scheme. The best
    # checkpoint per fold is saved.
    device = torch.device(device_str)
    pt_files = sorted(glob.glob(os.path.join(feature_dir, '*.pt')))
    is_seq = (architecture == 'GNN_LSTM')
    epochs_to_run = DEBUG_TRAIN_EPOCHS if DEBUG_TRAIN else EPOCHS
    # Checkpoint filenames carry the protocol and label mode, so runs of different configurations
    # cannot overwrite each other's weights.
    mode_tag = '' if EVAL_MODE == 'cross_subject' else f'_{EVAL_MODE}'
    mode_tag = (f'_smoothed{mode_tag}' if LABEL_MODE == 'smoothed' else mode_tag)
    print(f"Architecture : {architecture}")
    print(f"Sessions     : {len(pt_files)}")
    print(f"Device       : {device_str}")
    if len(pt_files) == 0:
        print("\nERROR: No .pt files found.\n" f"Looked in: {feature_dir}\n" "Check that FEATURE_DIR is correct and the feature " "extraction script has been run.")
        return []
    # Group every session by its subject. Filenames look like 's31_061020m.pt', so the subject id is
    # the leading token. Folds are built over these groups, never over individual sessions.
    subj_to_files = {}
    for p in pt_files:
        subj_to_files.setdefault(subj_of(p), []).append(p)
    subjects = sorted(subj_to_files)
    # Build the train/test folds for the selected EVAL_MODE 
    n_splits = min(N_SPLITS, len(subjects))
    fold_specs, cv_name, n_eval_units = build_eval_folds(EVAL_MODE, subjects, subj_to_files, pt_files, n_splits)
    if not fold_specs:
        print(f"\nERROR: EVAL_MODE='{EVAL_MODE}' produced no usable folds "
              f"(within-subject needs subjects with >=2 sessions).")
        return []
    print(f"Subjects     : {len(subjects)}  |  EVAL_MODE: {EVAL_MODE}  |  CV: {cv_name}")
    # Load feature names from the first file to build the node map
    sample_d = torch.load(pt_files[0], weights_only=False)
    all_feat_names = sample_d.get('feature_names', []) # If 'feature_names' doesn't exist, defaults to an empty list.
    print(f"Total features per epoch: {len(all_feat_names)}")
    # Optionally drop the svd/spec entropy columns before the RF sees them
    if DROP_REDUNDANT_ENTROPY:
        entropy_keep = np.array([not n.startswith(('svd_ent_', 'spec_ent_')) for n in all_feat_names])
    else:
        entropy_keep = np.ones(len(all_feat_names), dtype=bool)
    kept_idx = np.where(entropy_keep)[0]                          # positions in the stored feature axis we keep
    n_dropped = int((~entropy_keep).sum())
    if n_dropped:
        print(f"  Entropy pruning: dropped {n_dropped} svd/spec features -> {int(entropy_keep.sum())} fed to RF")
    all_results = []
    for fold_idx, (train_paths, test_paths, test_name) in enumerate(fold_specs):
        print(f"\n{'─' * 60}")
        print(f"Fold {fold_idx + 1}/{len(fold_specs)}  [{EVAL_MODE}]  test: {test_name}  "
              f"({len(test_paths)} session(s) held out, {len(train_paths)} train session(s))")
        if DEBUG_TRAIN and fold_idx >= DEBUG_TRAIN_FOLDS:
            print(f"\n[DEBUG] Reached {DEBUG_TRAIN_FOLDS} fold(s) — stopping.")
            print("Set DEBUG_TRAIN = False for the full run.")
            break

        # Load training features
        X_parts, y_parts = [], []
        for path in train_paths:
            d = torch.load(path, weights_only=False)
            Xf = d['features_norm'].numpy()[:, entropy_keep]
            # Rank features against the signal the model will actually be trained on: keep only the
            # epochs labelled under this LABEL_MODE and use their log-RT as the RF target.
            mask_key, rt_key = LABEL_KEYS
            keep = d[mask_key].numpy().astype(bool)
            Xf, yf = Xf[keep], d[rt_key].numpy()[keep]
            X_parts.append(Xf)
            y_parts.append(yf)
        X_all = np.vstack(X_parts)
        y_all = np.concatenate(y_parts).astype(np.float32)
        print(f"  Training epochs: {len(X_all)}")
        feat_mask_reduced = rf_feature_selection(X_all, y_all)
        # The RF ran on the entropy-pruned columns, so widen its mask back to the full stored feature axis.
        feat_mask = np.zeros(len(all_feat_names), dtype=bool)
        feat_mask[kept_idx[feat_mask_reduced]] = True
        # Build anatomical node assignment for selected features
        selected_names = [all_feat_names[i] for i in range(len(all_feat_names)) if feat_mask[i]]
        node_assignments = build_feature_name_to_node_map(selected_names)
        # For each node n, it counts how many features are assigned to it then takes the maximum.
        node_feat_dim = max(int(np.sum(node_assignments == n)) for n in range(N_NODES))
        node_feat_dim = max(node_feat_dim, 1)
        print(f"  node_feat_dim (max features per node): {node_feat_dim}")
        for ni, ch in enumerate(ALL_INTEREST):
            count = int(np.sum(node_assignments == ni))
            print(f"    node {ni} ({ch:4s}): {count} features")

        # Build datasets. The test set is all held-out-subject sessions.
        if is_seq: # gnnlstm mode
            train_ds = SequenceGraphDataset(train_paths, feat_mask, node_assignments)
            test_ds = SequenceGraphDataset(test_paths, feat_mask, node_assignments)
            collate = collate_sequences
        else:
            train_ds = EpochGraphDataset(train_paths, feat_mask, node_assignments)
            test_ds = EpochGraphDataset(test_paths, feat_mask, node_assignments)
            collate = collate_epoch_graphs
        if len(train_ds) == 0 or len(test_ds) == 0:
            print("  Skipping fold — empty dataset.")
            continue
        # see GNN_BATCH_SIZE: equal graphs per optimiser step in both architectures
        bs = BATCH_SIZE if is_seq else GNN_BATCH_SIZE
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=collate)
        test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, collate_fn=collate)

        # Build model
        ModelClass = GNNLSTMModel if is_seq else GNNModel
        model = ModelClass(node_feat_dim).to(device)
        # Initializes the AdamW optimizer with decoupled weight decay (L2 regularization) and LR Learning rate
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        # Initializes a cosine annealing learning rate scheduler with: T_max=EPOCHS: Maximum number of
        # iterations (matches the total epochs). eta_min= Minimum learning rate (1% of the initial LR).
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40, eta_min=LR * 0.01)
        best_r = -np.inf # Initializes the best Pearson correlation to negative infinity
        # best_GNN or best_GNN_LSTM depending on the architecture.
        checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'eeg_data_local',
                                      f'best_{architecture}_aligned')
        os.makedirs(checkpoint_dir, exist_ok=True)
        best_path = os.path.join(checkpoint_dir, f'best_{architecture}{mode_tag}_fold{fold_idx}.pt')
        no_improve_counter = 0
        PATIENCE_LIMIT = 15  # evaluations, not epochs -- see eval_freq below
        history = []  # per-epoch (epoch, train_loss, test_r, lr) for figures
        best_epoch = 0

        # Training loop: an epoch here means one complete pass through the entire training dataset. A
        # batch is a small subset of the data processed in one forward+backward pass.
        for epoch in range(1, epochs_to_run + 1):
            model.train() # Sets the model to training mode (enables dropout, batch normalization, etc.).
            epoch_loss, n_batches = 0.0, 0
            for batch in train_loader:
                # Everything we want the GPU to compute must be explicitly transferred to it first.
                # The model weights are moved once with model.to(device). The data tensors need to be
                # moved every batch because they come from the DataLoader which lives on CPU.
                if is_seq:
                    graphs, scores, valid, seq_lens, _keys = batch   # keys used only at scoring
                    graphs = graphs.to(device)
                    scores = scores.to(device)
                    valid = valid.to(device)
                    sp = model(graphs, seq_lens)
                    # Masked MSE: artifact timesteps (valid=0) are excluded from the loss.
                    loss = compute_loss(sp, scores, valid)
                else:
                    graphs, scores = batch
                    graphs = graphs.to(device)
                    scores = scores.to(device)
                    sp = model(graphs)
                    # GNN already dropped artifact epochs at dataset build, so no mask needed.
                    loss = compute_loss(sp, scores)
                opt.zero_grad() # Clears the gradients from the previous iteration.
                loss.backward() # Computes gradients of the loss with respect to the model parameters.
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP) # Clips the gradients to a maximum norm to prevent exploding gradients.
                opt.step() # Updates the model weights using the optimizer.
                epoch_loss += loss.item()
                n_batches += 1
            sched.step() # Updates the learning rate scheduler (cosine annealing) at the end of each epoch.
            # This scores the TEST sessions, not a validation split: the checkpoint kept is the one
            # that peaked on test. That is the peak-on-test optimism the TCN avoids, and it is why
            # the graph numbers are reported as an upper bound.
            eval_freq = 1 if DEBUG_TRAIN else 2
            if epoch % eval_freq == 0:
                m = evaluate(model, test_loader, device, is_seq)
                current_r = m['pearson_r']
                print(f"  ep {epoch:3d} | loss={epoch_loss / n_batches:.4f} | r={current_r:.3f} | MAE={m['mae']:.3f}")
                # the per-epoch curve, kept for the training-curve figure
                history.append({'epoch': epoch, 'train_loss': epoch_loss / max(n_batches, 1),
                                'test_r': float(current_r), 'lr': float(sched.get_last_lr()[0])})
                if current_r > best_r + 1e-4:
                    best_r = current_r
                    best_epoch = epoch
                    no_improve_counter = 0
                    torch.save(model.state_dict(), best_path)
                    print(f"  [Save] New best Pearson r: {best_r:.3f}")
                else:
                    no_improve_counter += 1
                    if no_improve_counter >= 3:
                        print(f"  [Wait] No improvement for {no_improve_counter}/{PATIENCE_LIMIT} epochs")
                if no_improve_counter >= PATIENCE_LIMIT:
                    print(f"  [Stop] Early stopping triggered at epoch {epoch} (Peak performance was r={best_r:.3f})")
                    break

        if os.path.exists(best_path):
            model.load_state_dict(torch.load(best_path, map_location=device))
            print(f"  [Load] Restored peak weights from checkpoint (r={best_r:.3f})")
        else:
            print("  [Warning] Checkpoint file not found. Using final epoch weights.")

        # Final evaluation of the model after training on the test set. Pooled r/MAE over all held-out
        # sessions' pre-onset epochs (predicted vs true per-trial log-RT).
        m = evaluate(model, test_loader, device, is_seq, return_preds=True)
        m['fold'] = fold_idx
        m['subject'] = test_name
        m['history'] = history          # per-epoch loss / test-r / lr (slide-10 curves)
        m['best_epoch'] = best_epoch     # epoch of peak r, for the gold-star marker
        # Pooled r can be inflated by between-session RT offsets (the baseline-recall effect); computing
        # r inside each session removes those offsets, so this is "does the model track RT moment-to-
        # moment within one drive". Build a single-session loader per test session and reuse evaluate().
        per_session_r = []
        DS = SequenceGraphDataset if is_seq else EpochGraphDataset
        for tp in test_paths:
            ds1 = DS([tp], feat_mask, node_assignments)
            if len(ds1) == 0:
                continue
            ld1 = DataLoader(ds1, batch_size=bs, shuffle=False, collate_fn=collate)
            r1 = evaluate(model, ld1, device, is_seq)['pearson_r']
            if not np.isnan(r1):
                per_session_r.append(r1)
        if per_session_r:
            m['per_session_r_mean'] = float(np.mean(per_session_r))
            m['per_session_r_median'] = float(np.median(per_session_r))
            m['n_sessions_scored'] = len(per_session_r)
            m['per_session_r'] = [float(x) for x in per_session_r]
        all_results.append(m)
        print(f"  ★  POOLED r={m['pearson_r']:.3f}  MAE={m['mae']:.3f}")
        if per_session_r:
            print(f"  ★  PER-SESSION r: mean={np.mean(per_session_r):+.3f}  "
                  f"median={np.median(per_session_r):+.3f}  ({len(per_session_r)} sessions)")

    if all_results:
        pr = [r['pearson_r'] for r in all_results]
        mae = [r['mae'] for r in all_results]
        psm = [r['per_session_r_mean'] for r in all_results if 'per_session_r_mean' in r]
        print("=" * 56)
        print(f"{architecture}  LABEL_MODE={LABEL_MODE}  EVAL_MODE={EVAL_MODE}")
        print(f"  Pooled r        : {np.mean(pr):.3f} ± {np.std(pr):.3f}")
        print(f"  Per-session r   : {np.mean(psm):.3f}" if psm else "  Per-session r   : n/a")
        print(f"  MAE (log10 RT)  : {np.mean(mae):.3f}")
        # results_{MODEL}_{label}_{eval}.json, the same scheme the TCN uses, so one rule parses
        # every result file in eeg_data_local/.
        model_tag = 'GNN-LSTM' if architecture == 'GNN_LSTM' else 'GNN'
        results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'eeg_data_local',
                                    f'results_{model_tag}_{LABEL_MODE}_{EVAL_MODE}.json')
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"  saved -> {results_path}")
    return all_results

if __name__ == '__main__':
    device_str = os.environ.get('GNN_DEVICE') or (
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available()
        else 'cpu')
    if device_str == 'mps':
        # a few ops still lack MPS kernels; fall back to CPU for those rather than crashing
        os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
    results = run_cross_validation(feature_dir=FEATURE_DIR, architecture=ARCHITECTURE, device_str=device_str)
