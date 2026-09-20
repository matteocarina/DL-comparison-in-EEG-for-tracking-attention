"""Compare feature budgets to justify keeping 8 core features and adding 7.
Every budget is selected on the training and validation recordings inside each fold, then scored
once on that fold's held-out test recordings."""
import json, os, sys, time
# run directly (python analysis/x.py) as well as from the repo root, so make the project
# importable either way
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from lib import feature_selection_core as fsc
from lib import folds

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELED = os.path.join(HERE, 'eeg_data_local', 'extracted_features_labeled')
RESULTS = os.path.join(HERE, 'eeg_data_local')

LABEL_KEYS = {'event_locked': ('preonset_mask', 'preonset_rt'),
              'smoothed': ('smoothed_mask', 'smoothed_rt')}

LABEL_MODE = sys.argv[1] if len(sys.argv) > 1 else 'event_locked'
EVAL_MODE = sys.argv[2] if len(sys.argv) > 2 else 'within_subject'
assert LABEL_MODE in LABEL_KEYS, LABEL_MODE
BUDGETS = [(11, 10), (10, 10), (8, 7), (11, 7), (8, 10), (11, 0), (8, 0)]

def main():
    import glob
    paths = sorted(glob.glob(os.path.join(LABELED, '*.pt')))
    if not paths:
        raise SystemExit(f"no .pt in {LABELED} — run 1_ and 2_ first")
    store, names = fsc.load_pool(paths, *LABEL_KEYS[LABEL_MODE])
    core = fsc.fixed_indices(names)
    pool = [j for j in range(len(names)) if j not in core]
    print(f"{len(paths)} recordings | core {len(core)} | pool {len(pool)} "
          f"| labels={LABEL_MODE} protocol={EVAL_MODE}\n")

    cv = folds.build_folds(paths, EVAL_MODE)
    results = {b: [] for b in BUDGETS}
    core_orders = []
    for fi, (train, val, test) in enumerate(cv, 1):
        t0 = time.time()
        # fsc.greedy, not a local copy: this script kept its own duplicate, which is exactly the
        # drift feature_selection_core exists to prevent.
        core_order = fsc.greedy(train, val, store, core, [], len(core))
        core_orders.append(core_order)
        for (n_core, n_extra) in BUDGETS:
            cols = fsc.greedy(train, val, store, pool, core_order[:n_core], n_extra)
            model = fsc.ridge_fit(*fsc.stack(train, store, cols))
            results[(n_core, n_extra)].append(fsc.ridge_score(model, test, store, cols))
        print(f"  fold {fi} done ({time.time() - t0:.0f}s)   core order: "
              f"{[names[j] for j in core_order[:4]]} ...")

    pos = {j: float(np.mean([o.index(j) for o in core_orders])) for j in core}
    ranked = sorted(core, key=lambda j: pos[j])
    print(f"\ncore features ranked by usefulness (mean greedy position across folds):")
    for i, j in enumerate(ranked, 1):
        tail = '   <- dropped at n_core=8' if i > 8 else ''
        print(f"  {i:2d}. {names[j]:12s} {pos[j]:5.1f}{tail}")

    print(f"\nheld-out TEST per-session r by budget:")
    print(f"  {'core':>5s} {'added':>6s} {'total':>6s} {'test r':>9s} {'sd':>7s}")
    for (n_core, n_extra) in BUDGETS:
        v = np.array(results[(n_core, n_extra)], dtype=float)
        print(f"  {n_core:5d} {n_extra:6d} {n_core + n_extra:6d} "
              f"{np.nanmean(v):9.3f} {np.nanstd(v, ddof=1):7.3f}")

    out = os.path.join(RESULTS, f'feature_budgets_{LABEL_MODE}_{EVAL_MODE}.json')
    json.dump({'label_mode': LABEL_MODE, 'eval_mode': EVAL_MODE, 'names': names,
               'core_ranked': [int(j) for j in ranked],
               'budgets': {f'{a}+{b}': results[(a, b)] for a, b in BUDGETS}},
              open(out, 'w'), indent=2)
    print(f"\nsaved -> {os.path.relpath(out, HERE)}")

if __name__ == '__main__':
    main()
