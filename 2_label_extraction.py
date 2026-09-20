"""Derive the supervision signal from the task markers and write labelled copies.
Two label modes are saved side by side: 'event_locked' puts each trial's log10 reaction time on the
last epoch ending before the deviation onset, and 'smoothed' spreads every trial's log-RT into a
continuous curve with a Gaussian kernel. The model chooses which to train on."""

import os
import glob
import warnings
import numpy as np
import torch
import mne

from lib import sadt_io

warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')

# config
LABEL_MODE = 'both'          # 'event_locked' | 'smoothed' | 'both'
HERE = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(HERE, 'eeg_data_local', 'extracted_features')
OUT_DIR = os.path.join(HERE, 'eeg_data_local', 'extracted_features_labeled')
SET_ROOT = os.environ.get('SADT_DATA', os.path.expanduser('~/PycharmProjects/7666055'))
EPOCH_LENGTH = 4.0
OVERLAP_S = 0.5
MAX_RT_S = 10.0
MIN_RT_S = 0.15
EVENT_ID = {'251': 1, '252': 2, '253': 3, '254': 4}
DEV_ONSET_CODE, RESP_ONSET_CODE = (1, 2), 3   # 251/252 = deviation left/right, 253 = response onset
SMOOTH_BW_S = 60.0 # Gaussian kernel bandwidth (s) for the smoothed target (~1 min vigilance scale)
SMOOTH_EDGE_S = SMOOTH_BW_S  # don't label epochs more than this far outside the trial-covered span

def extract_rt_events(events, sfreq):
    # Returns each deviation onset time and its reaction time, both in seconds. A trial's RT is the
    # first response between MIN_RT_S and the earlier of MAX_RT_S or the next onset; nan if none.
    dev_times = np.sort(events[np.isin(events[:, 2], DEV_ONSET_CODE), 0] / sfreq)
    resp_times = np.sort(events[events[:, 2] == RESP_ONSET_CODE, 0] / sfreq)
    trial_times, rt_values = [], []
    n_dev = len(dev_times)
    for k, dt in enumerate(dev_times):
        next_dt = dev_times[k + 1] if k + 1 < n_dev else np.inf
        upper = min(dt + MAX_RT_S, next_dt)
        cand = resp_times[(resp_times > dt + MIN_RT_S) & (resp_times < upper)]
        rt_values.append(cand[0] - dt if len(cand) > 0 else np.nan)
        trial_times.append(dt)
    return np.array(trial_times), np.array(rt_values)

def build_event_locked(trial_times, rt_values, epoch_starts, artifact):
    # sparse: label the last epoch ending at/before each trial's deviation onset.
    n = len(epoch_starts)
    epoch_ends = epoch_starts + EPOCH_LENGTH
    mask = np.zeros(n, dtype=bool)
    rt_log = np.zeros(n, dtype=np.float32)
    gap_at = np.full(n, np.inf)
    for dt, rt in zip(trial_times, rt_values):
        if np.isnan(rt):
            continue
        i = int(np.searchsorted(epoch_ends, dt, side='right') - 1)
        if i < 0 or i >= n or not np.isfinite(epoch_ends[i]) or artifact[i]:
            continue
        gap = dt - epoch_ends[i]
        if gap < gap_at[i]:
            gap_at[i] = gap
            mask[i] = True
            rt_log[i] = np.log10(rt + 1e-3)
    return mask, rt_log

def build_smoothed(trial_times, rt_values, epoch_starts, artifact, bw=SMOOTH_BW_S):
    # dense: Gaussian smoothing of per-trial log-RT onto every epoch center.
    n = len(epoch_starts)
    mask = np.zeros(n, dtype=bool)
    smooth = np.zeros(n, dtype=np.float32)
    valid = ~np.isnan(rt_values)
    tt = trial_times[valid]
    yy = np.log10(rt_values[valid] + 1e-3)
    if len(tt) < 3:
        return mask, smooth
    lo, hi = tt.min(), tt.max()
    centers = epoch_starts + EPOCH_LENGTH / 2.0
    for i, t in enumerate(centers):
        if not np.isfinite(t) or t < lo - SMOOTH_EDGE_S or t > hi + SMOOTH_EDGE_S:
            continue
        w = np.exp(-0.5 * ((tt - t) / bw) ** 2)
        sw = w.sum()
        if sw < 1e-3:
            continue
        smooth[i] = float(np.sum(w * yy) / sw)
        if not artifact[i]:
            mask[i] = True
    return mask, smooth

# Main loop
assert LABEL_MODE in ('event_locked', 'smoothed', 'both')
os.makedirs(OUT_DIR, exist_ok=True)
pt_files = sorted(glob.glob(os.path.join(SRC_DIR, '*.pt')))
if not pt_files:
    raise SystemExit(f"No .pt files in {SRC_DIR} — run 1_preprocessing_and_features.py first.")
print(f"LABEL_MODE={LABEL_MODE}   {len(pt_files)} sessions   {SRC_DIR} -> {OUT_DIR}\n")

n_ok = 0
tot_ep = tot_pre = tot_smooth = 0
for p in pt_files:
    d = torch.load(p, weights_only=False)
    n_ep = d['features_raw'].shape[0]
    art = d.get('artifact_mask', None)
    artifact = art.numpy().astype(bool) if art is not None else np.zeros(n_ep, bool)
    session = os.path.basename(p)[:-3]
    try:
        # Unzips the session on demand and removes it again (see sadt_io).
        with sadt_io.session_bundle(session, SET_ROOT) as set_path:
            raw = mne.io.read_raw_eeglab(set_path, preload=False, verbose=False)
            sfreq = raw.info['sfreq']
            present = {str(x) for x in raw.annotations.description}
            known = {k: v for k, v in EVENT_ID.items() if k in present}
            if not known:
                print(f"  {session:28s}  SKIP (no recognised SADT markers: {sorted(present)[:6]})")
                continue
            events, _ = mne.events_from_annotations(raw, event_id=known, verbose=False)
            fle = mne.make_fixed_length_events(raw, start=0, duration=EPOCH_LENGTH,
                                               stop=raw.times[-1] - EPOCH_LENGTH + 1.0 / sfreq,
                                               overlap=OVERLAP_S)
            epoch_starts = fle[:, 0] / sfreq
            del raw          # release the memmap before the extracted copy is deleted
    except FileNotFoundError:
        print(f"  {session:28s}  SKIP (no .set or .set.zip)")
        continue
    if not np.any(np.isin(events[:, 2], DEV_ONSET_CODE)):
        print(f"  {session:28s}  SKIP (no deviation events)")
        continue

    # Align reconstructed grid to stored feature rows
    epoch_starts = epoch_starts[:n_ep]
    if len(epoch_starts) < n_ep:
        epoch_starts = np.concatenate([epoch_starts, np.full(n_ep - len(epoch_starts), np.inf)])

    trial_times, rt_values = extract_rt_events(events, sfreq)
    n_resp = int(np.sum(~np.isnan(rt_values)))

    n_pre = n_sm = 0
    if LABEL_MODE in ('event_locked', 'both'):
        m, rt = build_event_locked(trial_times, rt_values, epoch_starts, artifact)
        d['preonset_mask'] = torch.from_numpy(m)
        d['preonset_rt'] = torch.from_numpy(rt)
        n_pre = int(m.sum())
    if LABEL_MODE in ('smoothed', 'both'):
        m, sm = build_smoothed(trial_times, rt_values, epoch_starts, artifact)
        d['smoothed_mask'] = torch.from_numpy(m)
        d['smoothed_rt'] = torch.from_numpy(sm)
        n_sm = int(m.sum())

    torch.save(d, os.path.join(OUT_DIR, os.path.basename(p)))
    n_ok += 1
    tot_ep += n_ep
    tot_pre += n_pre
    tot_smooth += n_sm
    print(f"  {session:28s}  epochs={n_ep:5d}  responded={n_resp:4d}  "
          f"event_locked={n_pre:4d}  smoothed={n_sm:5d}")

print(f"\nDone. Wrote {n_ok} sessions to {OUT_DIR}.")
if n_ok:
    if tot_pre:
        print(f"  event_locked density: {tot_pre}/{tot_ep} = {100 * tot_pre / tot_ep:.1f}% (sparse)")
    if tot_smooth:
        print(f"  smoothed density:     {tot_smooth}/{tot_ep} = {100 * tot_smooth / tot_ep:.1f}% (dense)")
