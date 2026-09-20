"""Control baseline: predict reaction time from elapsed time alone, using no EEG.

Uses the same folds, masks and metrics as the models, so any EEG model that cannot beat this is
only tracking time-on-task."""

import glob
import json
import os
# run directly (python analysis/x.py) as well as from the repo root, so make the project
# importable either way
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import numpy as np

from lib import folds
import torch
from scipy.stats import pearsonr

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURE_DIR = os.path.join(HERE, 'eeg_data_local', 'extracted_features_labeled')
OUT_DIR = os.path.join(HERE, 'eeg_data_local')

EPOCH_LENGTH, HOP = 4.0, 3.5
LABEL_KEYS = {'event_locked': ('preonset_mask', 'preonset_rt'),
              'smoothed': ('smoothed_mask', 'smoothed_rt')}

def load_labels(path, label_mode):
    """(t, y, valid) for one recording: epoch-centre time in seconds, target, supervision mask."""
    d = torch.load(path, weights_only=False)
    mask_key, rt_key = LABEL_KEYS[label_mode]
    y = d[rt_key].numpy().astype(np.float64)
    valid = d[mask_key].numpy().astype(bool)
    t = np.arange(len(y)) * HOP + EPOCH_LENGTH / 2.0
    return t, y, valid

def fit_line(paths, label_mode):
    """Least-squares fit of y = a + b*t pooled over the training recordings."""
    T, Y = [], []
    for p in paths:
        t, y, v = load_labels(p, label_mode)
        T.append(t[v]); Y.append(y[v])
    T = np.concatenate(T); Y = np.concatenate(Y)
    if len(T) < 2 or np.std(T) == 0:
        return float(np.mean(Y)) if len(Y) else 0.0, 0.0
    b, a = np.polyfit(T, Y, 1)
    return float(a), float(b)

def score(paths, label_mode, a, b):
    """Pooled r, MAE and per-session r on held-out recordings."""
    P, Tt, per_session = [], [], []
    for p in paths:
        t, y, v = load_labels(p, label_mode)
        if v.sum() < 3:
            continue
        pred = a + b * t[v]
        true = y[v]
        P.append(pred); Tt.append(true)
        if np.std(pred) > 0 and np.std(true) > 0:
            per_session.append(float(pearsonr(pred, true)[0]))
    if not P:
        return None
    P = np.concatenate(P); Tt = np.concatenate(Tt)
    pooled = float(pearsonr(P, Tt)[0]) if np.std(P) > 0 and np.std(Tt) > 0 else 0.0
    return {
        'pearson_r': pooled,
        'mae': float(np.mean(np.abs(P - Tt))),
        'per_session_r': per_session,
        'per_session_r_mean': float(np.mean(per_session)) if per_session else float('nan'),
        'per_session_r_median': float(np.median(per_session)) if per_session else float('nan'),
        'pred': P.tolist(), 'true': Tt.tolist(),
    }

def build_folds_for(eval_mode, paths):
    # The models' own folds, so the comparison is exact.
    return folds.build_folds(paths, eval_mode)

def main():
    paths = sorted(glob.glob(os.path.join(FEATURE_DIR, '*.pt')))
    if not paths:
        raise SystemExit(f"no .pt in {FEATURE_DIR}")

    for eval_mode in ('within_subject', 'cross_subject'):
        cv = build_folds_for(eval_mode, paths)
        for label_mode in ('event_locked', 'smoothed'):
            fold_results = []
            for fi, (train, val, test) in enumerate(cv):
                a, b = fit_line(train, label_mode)      # train recordings only
                m = score(test, label_mode, a, b)       # held-out test recordings
                if m is None:
                    continue
                m['fold'] = fi
                m['intercept'] = a
                m['slope_per_hour'] = b * 3600.0
                fold_results.append(m)

            out = os.path.join(OUT_DIR, f'results_TIME_{label_mode}_{eval_mode}.json')
            payload = {
                'config': {'data_mode': 'time_only', 'label_mode': label_mode,
                           'eval_mode': eval_mode, 'n_splits': len(fold_results),
                           'n_params': 2, 'complete': True, 'folds_done': len(fold_results)},
                'folds': fold_results,
            }
            with open(out, 'w') as f:
                json.dump(payload, f, indent=2)

            psr = [r for fr in fold_results for r in fr['per_session_r']]
            pooled = np.mean([fr['pearson_r'] for fr in fold_results])
            mae = np.mean([fr['mae'] for fr in fold_results])
            slope = np.mean([fr['slope_per_hour'] for fr in fold_results])
            print(f"  {label_mode:13s} {eval_mode:15s} "
                  f"pooled r={pooled:+.3f}  per-session r={np.mean(psr):+.3f}  "
                  f"MAE={mae:.3f}  slope={slope:+.3f} log10RT/hour")

if __name__ == '__main__':
    main()
