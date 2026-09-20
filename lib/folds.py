"""Cross-validation folds, shared by the model and every analysis script.

Defined once here because five scripts previously read 3b_model_tcn.py as text and exec'd it just
to borrow these functions. The split is deterministic given SEED, so any script can rebuild exactly
the folds a run used and map its per-session scores back to subjects."""

import os
from collections import defaultdict

import numpy as np

N_SPLITS = 5
SEED = 42
VAL_N_RECORDINGS = 6              # whole training recordings held out for validation (per fold)

def subj_of(path):
    return os.path.basename(path).split('_')[0]

def build_folds_cross_subject(paths):
    """Grouped K-fold by subject: no subject appears in both train and test.

    Answers a different question from within_subject: not "can it track a driver it has calibrated
    on", but "does it transfer to a driver it has never seen". Expect markedly lower numbers -- the
    per-recording z-score removes amplitude offsets but not individual EEG morphology.
    """
    subj = defaultdict(list)
    for p in paths:
        subj[subj_of(p)].append(p)
    subjects = sorted(subj)
    rng = np.random.default_rng(SEED)
    order = list(subjects)
    rng.shuffle(order)
    chunks = [order[i::N_SPLITS] for i in range(N_SPLITS)]      # round-robin -> even subject counts
    folds = []
    for k in range(N_SPLITS):
        test_subj = set(chunks[k])
        val_subj = set(chunks[(k + 1) % N_SPLITS])              # a neighbouring chunk validates
        test = [p for p in paths if subj_of(p) in test_subj]
        val = [p for p in paths if subj_of(p) in val_subj]
        train = [p for p in paths if subj_of(p) not in test_subj | val_subj]
        folds.append((train, val, test))
    return folds

def build_folds_within_subject(paths):
    subj = defaultdict(list)
    for p in paths:
        subj[subj_of(p)].append(p)
    multi = {s: sorted(f) for s, f in subj.items() if len(f) >= 2}
    rng = np.random.default_rng(SEED)
    folds = []
    for k in range(N_SPLITS):
        test = [files[k % len(files)] for s, files in sorted(multi.items())]   # one rec per subject, cycled
        test_set = set(test)
        train_all = [p for p in paths if p not in test_set]

        # Carve a validation set from the training recordings (whole recordings, seeded).
        # Never take a recording that is the last remaining training recording of a subject who is
        # in test: that leaves the subject unseen during training, which silently turns their test
        # recording into a cross-subject evaluation and breaks the within-subject premise
        # (measured: 1-2 subjects per fold were being orphaned this way). The 8 single-recording
        # subjects are never tested, so they are always safe validation candidates.
        test_subj = {subj_of(p) for p in test}
        remaining = defaultdict(int)
        for p in train_all:
            remaining[subj_of(p)] += 1
        tr = list(train_all)
        rng.shuffle(tr)
        n_val = min(VAL_N_RECORDINGS, max(1, len(tr) // 6))
        val = []
        for p in tr:
            if len(val) >= n_val:
                break
            s = subj_of(p)
            if s in test_subj and remaining[s] <= 1:
                continue                      # would orphan a test subject -> skip
            val.append(p)
            remaining[s] -= 1
        val_set = set(val)
        train = [p for p in train_all if p not in val_set]
        folds.append((train, val, test))
    return folds

def build_folds(paths, eval_mode):
    if eval_mode == 'within_subject':
        return build_folds_within_subject(paths)
    if eval_mode == 'cross_subject':
        return build_folds_cross_subject(paths)
    raise ValueError(f"eval_mode must be 'within_subject' or 'cross_subject', got {eval_mode!r}")
