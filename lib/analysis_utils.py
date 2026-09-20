"""Map fold results back to subjects and summarise spread correctly.

Within-subject scoring produces 95 session scores from only 19 subjects, so treating them as
independent makes any confidence interval far too narrow. Everything inferential here resamples
subjects, not sessions."""

import glob
import json
import os
from collections import defaultdict

import numpy as np

from lib import folds

# the project root: this module lives in lib/, so go up one level. Getting this wrong
# silently emptied every results path and dropped the confidence intervals from the table.
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELED = os.path.join(HERE, 'eeg_data_local', 'extracted_features_labeled')
RESULTS = os.path.join(HERE, 'eeg_data_local')

def subject_of(path):
    return os.path.basename(path).split('_')[0]

def per_session_by_subject(model, label_mode, eval_mode, variant=''):
    """{subject: [r, ...]} for one TCN configuration, or None if the results cannot be aligned.

    `variant` is the filename suffix that distinguishes an ablation from the default run (e.g.
    'all51'). Omitting it silently returned the default run's scores for an ablation row, so the
    ablation was reported with the wrong confidence interval -- identical to the row above it.
    """
    suffix = f'_{variant}' if variant else ''
    path = os.path.join(RESULTS, f'results_{model}_{label_mode}_{eval_mode}{suffix}.json')
    if not os.path.exists(path):
        return None
    payload = json.load(open(path))
    fold_results = payload['folds'] if isinstance(payload, dict) else payload
    paths = sorted(glob.glob(os.path.join(LABELED, '*.pt')))
    splits = folds.build_folds(paths, eval_mode)

    out = defaultdict(list)
    for (tr, va, te), fr in zip(splits, fold_results):
        v = fr.get('per_session_r', [])
        if len(te) != len(v):          # a session was skipped; alignment is not trustworthy
            return None
        for p, r in zip(te, v):
            out[subject_of(p)].append(r)
    return dict(out)

def bootstrap_ci_by_subject(by_subject, n_boot=4000, alpha=0.05, seed=0):
    """Percentile CI for the mean per-session r, resampling subjects with replacement.

    Sessions within a subject are kept together, so the interval reflects the fact that the real
    sample size is the number of drivers, not the number of drives.
    """
    subs = sorted(by_subject)
    if len(subs) < 3:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(len(subs), size=len(subs), replace=True)
        vals = [r for i in pick for r in by_subject[subs[i]]]
        means[b] = np.mean(vals)
    return (float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))

def spread(values):
    """Descriptive spread of per-session r: what a reader wants to know about variability."""
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return {}
    q1, q3 = np.percentile(v, [25, 75])
    return {
        'mean': float(v.mean()),
        'sd': float(v.std(ddof=1)) if v.size > 1 else np.nan,
        'q1': float(q1), 'q3': float(q3), 'iqr': float(q3 - q1),
        'min': float(v.min()), 'max': float(v.max()),
        'frac_positive': float((v > 0).mean()),
        'n': int(v.size),
    }

# GNN alignment
def gnn_per_session_by_subject(label_mode, eval_mode, model='GNN-LSTM'):
    """{subject: [r, ...]} for a graph-model run ('GNN-LSTM' or the spatial-only 'GNN').

    Its results store per_session_r without session identity, but build_eval_folds in
    3a_model_gnn_lstm.py is fully deterministic -- within_subject cycles each multi-session
    subject's sorted recordings (files[k % len(files)]) and cross_subject uses a seeded KFold over
    sorted subjects -- so the test sets can be rebuilt exactly and each score mapped back. Returns
    None rather than guessing if the rebuilt fold sizes do not match the stored scores.

    """
    from sklearn.model_selection import KFold

    gnn_dir = RESULTS
    path = os.path.join(gnn_dir, f'results_{model}_{label_mode}_{eval_mode}.json')
    if not os.path.exists(path):
        return None
    payload = json.load(open(path))
    fold_results = payload['folds'] if isinstance(payload, dict) else payload

    pt = sorted(glob.glob(os.path.join(LABELED, '*.pt')))   # the shared extraction
    if not pt:
        return None
    subj_to_files = defaultdict(list)
    for p in pt:
        subj_to_files[subject_of(p)].append(p)
    subjects = sorted(subj_to_files)

    test_sets = []
    if eval_mode == 'within_subject':
        multi = {s: sorted(f) for s, f in subj_to_files.items() if len(f) >= 2}
        for k in range(len(fold_results)):
            test_sets.append([files[k % len(files)] for _, files in sorted(multi.items())])
    else:
        kf = KFold(n_splits=max(len(fold_results), 2), shuffle=True, random_state=42)
        for _, idx in kf.split(subjects):
            test_sets.append([p for i in idx for p in sorted(subj_to_files[subjects[i]])])

    out = defaultdict(list)
    for test_paths, fr in zip(test_sets, fold_results):
        v = fr.get('per_session_r', [])
        if len(test_paths) != len(v):        # cannot be trusted; refuse rather than mis-attribute
            return None
        for p, r in zip(test_paths, v):
            out[subject_of(p)].append(r)
    return dict(out)

# freshness
def data_mtime():
    """When the current labelled extraction was written."""
    pts = glob.glob(os.path.join(LABELED, '*.pt'))
    return max(os.stat(p).st_mtime for p in pts) if pts else 0.0

def stale_results(paths):
    """Result files older than the extraction they claim to describe.

    Re-running script 1 changes every feature value, so a result file written before the current
    extraction describes different inputs. Mixing the two in one table produced exactly that
    silent error once already -- old GNN numbers sitting beside new TCN ones under a different
    reference and a different normalisation.

    """
    cutoff = data_mtime()
    return [p for p in paths if os.path.exists(p) and os.stat(p).st_mtime < cutoff]

def warn_if_stale(paths, label='result'):
    bad = stale_results(paths)
    if bad:
        names = ', '.join(os.path.basename(p) for p in bad)
        print(f"  warning: {len(bad)} {label} file(s) predate the current extraction: {names}")
    return bad
