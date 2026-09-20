"""Sweep the number of added features to justify the 15-feature budget.
Ranks the band-power core among itself, then adds candidates one at a time, recording validation and
held-out test curves for every fold. The validation curve chooses; the test curve reports."""

import glob
import json
import os
import sys
# run directly (python analysis/x.py) as well as from the repo root, so make the project
# importable either way
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lib import feature_selection_core as fsc
from lib import folds
from lib.plot_style import NAVY, STEEL, GOLD, CRIMSON, MUTED, apply_style, grid, despine, save

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELED = os.path.join(HERE, 'eeg_data_local', 'extracted_features_labeled')
OUT_DIR = os.path.join(HERE, 'figures')
RESULTS = os.path.join(HERE, 'eeg_data_local')

# The budget the model actually uses. Keep these in step with 3b_model_tcn.py.
N_CORE = int(os.environ.get('N_CORE', 8))      # of the 11 band-power features
N_EXTRA = int(os.environ.get('N_EXTRA', 7))    # added from the 130 entropy/connectivity candidates
MAX_K = 25                                     # how far the curve explores past N_EXTRA

LABEL_KEYS = {'event_locked': ('preonset_mask', 'preonset_rt'),
              'smoothed': ('smoothed_mask', 'smoothed_rt')}

FAM_COLOR = {'band power': NAVY, 'entropy': GOLD, 'wPLI theta': STEEL, 'wPLI alpha': CRIMSON}

def family(name):
    if name.startswith('wpli_t'):
        return 'wPLI theta'
    if name.startswith('wpli_a'):
        return 'wPLI alpha'
    if name.startswith(('samp_ent', 'perm_ent', 'svd_ent', 'spec_ent')):
        return 'entropy'
    return 'band power'

def run_fold(train, val, test, store, names, max_k):
    """One fold: rank the core, keep N_CORE, then add features one at a time.
    Returns the added-feature order plus validation and test curves indexed by how many extra
    features have been added (index 0 = the core alone).
    """
    core = fsc.fixed_indices(names)
    pool = [j for j in range(len(names)) if j not in core]

    core_ranked = fsc.greedy(train, val, store, core, [], len(core))
    chosen = core_ranked[:N_CORE]

    def score(cols, on):
        return fsc.ridge_score(fsc.ridge_fit(*fsc.stack(train, store, cols)), on, store, cols)

    order, val_curve, test_curve = [], [score(chosen, val)], [score(chosen, test)]
    remaining = list(pool)
    for _ in range(max_k):
        best_j, best_r = None, -np.inf
        for j in remaining:
            r = score(chosen + [j], val)
            if np.isfinite(r) and r > best_r:
                best_j, best_r = j, r
        if best_j is None:
            break
        chosen.append(best_j); remaining.remove(best_j)
        order.append(best_j)
        val_curve.append(best_r)
        test_curve.append(score(chosen, test))
    return core_ranked, order, val_curve, test_curve

def main():
    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    label_mode = args[0] if args else 'event_locked'
    eval_mode = args[1] if len(args) > 1 else 'within_subject'
    assert label_mode in LABEL_KEYS, label_mode

    paths = sorted(glob.glob(os.path.join(LABELED, '*.pt')))
    if not paths:
        raise SystemExit(f"no .pt in {LABELED} — run 1_ and 2_ first")
    store, names = fsc.load_pool(paths, *LABEL_KEYS[label_mode])
    core = fsc.fixed_indices(names)

    print(f"greedy forward selection   labels={label_mode}   protocol={eval_mode}")
    print(f"  {len(paths)} recordings, {len(names)} candidate features")
    print(f"  band-power core : {len(core)}  -> keeping the best {N_CORE}")
    print(f"  other candidates: {len(names) - len(core)}  -> adding {N_EXTRA}")
    print(f"  target budget   : {N_CORE} + {N_EXTRA} = {N_CORE + N_EXTRA} features\n")
    cv = folds.build_folds(paths, eval_mode)

    core_orders, orders, vcurves, tcurves = [], [], [], []
    for fi, (train, val, test) in enumerate(cv, 1):
        t0 = time.time()
        cr, order, vc, tc = run_fold(train, val, test, store, names, MAX_K)
        core_orders.append(cr); orders.append(order); vcurves.append(vc); tcurves.append(tc)
        print(f"  fold {fi}: core kept {[names[j] for j in cr[:N_CORE]][:3]}... | "
              f"test at {N_CORE}+{N_EXTRA} = {tc[min(N_EXTRA, len(tc) - 1)]:.3f}  "
              f"({time.time() - t0:.0f}s)")

    L = min(min(len(c) for c in vcurves), min(len(c) for c in tcurves))
    V = np.array([c[:L] for c in vcurves]); T = np.array([c[:L] for c in tcurves])
    vmean, vsd = V.mean(0), V.std(0, ddof=1)
    tmean, tsd = T.mean(0), T.std(0, ddof=1)
    k = min(N_EXTRA, L - 1)

    print(f"\n  core alone ({N_CORE} features)        test r = {tmean[0]:.3f} ± {tsd[0]:.3f}")
    print(f"  CHOSEN {N_CORE}+{N_EXTRA} = {N_CORE + N_EXTRA} features     test r = {tmean[k]:.3f} ± {tsd[k]:.3f}")
    print(f"  validation at the same point         {vmean[k]:.3f}  (biased — the search objective)")
    if L - 1 > N_EXTRA:
        print(f"  for reference, {N_CORE}+{L - 1} features      test r = {tmean[-1]:.3f}")

    # which core features survive, and what gets added
    cpos = {j: float(np.mean([o.index(j) for o in core_orders])) for j in core}
    cranked = sorted(core, key=lambda j: cpos[j])
    print(f"\n  band-power core ranked among itself:")
    for i, j in enumerate(cranked, 1):
        print(f"    {i:2d}. {names[j]:12s} {cpos[j]:4.1f}" + ('' if i <= N_CORE else '   <- dropped'))

    apos = {}
    for j in {x for o in orders for x in o}:
        apos[j] = float(np.mean([o.index(j) if j in o else MAX_K + 5 for o in orders]))
    added = sorted(apos, key=lambda j: apos[j])[:N_EXTRA]
    print(f"\n  most consistently added ({N_EXTRA}):")
    for i, j in enumerate(added, 1):
        print(f"    {i:2d}. {names[j]:22s} [{family(names[j])}]")
    fams = [family(names[j]) for j in added]
    print(f"    composition: " + ", ".join(f"{f} {fams.count(f)}" for f in FAM_COLOR if fams.count(f)))

    out = os.path.join(RESULTS, f'feature_selection_{label_mode}_{eval_mode}.json')
    json.dump({'label_mode': label_mode, 'eval_mode': eval_mode, 'names': names,
               'n_core': N_CORE, 'n_extra': N_EXTRA,
               'core_ranked': [int(j) for j in cranked],
               'consensus_added': [int(j) for j in added],
               'orders': [[int(j) for j in o] for o in orders],
               'core_orders': [[int(j) for j in o] for o in core_orders],
               'val_curve_mean': vmean.tolist(), 'val_curve_sd': vsd.tolist(),
               'test_curve_mean': tmean.tolist(), 'test_curve_sd': tsd.tolist()},
              open(out, 'w'), indent=2)
    print(f"\n  saved -> {os.path.basename(out)}")

    figure(names, vmean, vsd, tmean, tsd, added, k, label_mode, eval_mode)

def figure(names, vmean, vsd, tmean, tsd, added, k, label_mode, eval_mode):
    apply_style()
    x = np.arange(len(vmean))          # 0 = core alone, then one per added feature
    fig, ax = plt.subplots(figsize=(13, 6.4))

    ax.fill_between(x, vmean - vsd, vmean + vsd, color=NAVY, alpha=.12, lw=0)
    ax.plot(x, vmean, color=NAVY, lw=2.1, label='validation $r$ — the search objective (biased)')
    ax.fill_between(x, tmean - tsd, tmean + tsd, color=GOLD, alpha=.14, lw=0)
    ax.plot(x, tmean, color=GOLD, lw=2.4, label='held-out TEST $r$ — the honest curve')

    ax.axvline(k, color=MUTED, ls='--', lw=1.2)
    ax.plot(k, tmean[k], '*', ms=18, color=GOLD, mec='#6E5410', mew=1.2, zorder=6)
    ax.annotate(f'chosen: {N_CORE} core + {N_EXTRA} added\ntest $r$ = {tmean[k]:.3f}',
                xy=(k, tmean[k]), xytext=(k + 2.2, tmean[k] - .035),
                fontsize=10.5, color='#6E5410', fontweight='semibold', linespacing=1.4,
                arrowprops=dict(arrowstyle='-|>', color=GOLD, lw=1.3))

    lo = ax.get_ylim()[0]
    for i, j in enumerate(added, 1):
        if i < len(vmean):
            ax.annotate(names[j], xy=(i, lo), xytext=(i, lo + (vmean.max() - lo) * .04),
                        fontsize=7.8, rotation=90, ha='center', va='bottom',
                        color=FAM_COLOR[family(names[j])])

    ax.set_xlabel(f'features added to the {N_CORE}-feature band-power core')
    ax.set_ylabel('per-session Pearson $r$')
    ax.set_title(f'Feature selection — {label_mode.replace("_", "-")}, '
                 f'{eval_mode.replace("_", "-")}',
                 loc='left', fontsize=13.5, fontweight='semibold', pad=28)
    ax.text(0, 1.01, f'core ranked among itself, best {N_CORE} kept · additions chosen on '
                     f'validation inside each fold · ridge surrogate',
            transform=ax.transAxes, fontsize=9.5, color=MUTED, va='bottom')
    ax.legend(loc='lower right')
    grid(ax); despine(ax)
    save(fig, os.path.join(OUT_DIR, f'fig3_feature_selection_{label_mode}.png'))

if __name__ == '__main__':
    main()
