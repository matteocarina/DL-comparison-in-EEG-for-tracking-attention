"""Build the explanatory figures for the write-up.
Draws figures 1, 2 and 5. Figure 3 comes from analysis/feature_selection.py and figure 4 from
4_compare_models.py, which needs the whole results grid. Pass figure numbers to build only those."""

import glob
import json
import os
import sys
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from lib.plot_style import NAVY, STEEL, GOLD, CRIMSON, INK, MUTED, apply_style, grid, despine, save
matplotlib.use('Agg')
HERE = os.path.dirname(os.path.abspath(__file__))
LABELED = os.path.join(HERE, 'eeg_data_local', 'extracted_features_labeled')
RESULTS = os.path.join(HERE, 'eeg_data_local')
OUT = os.path.join(HERE, 'figures')
EPOCH_LENGTH, HOP = 4.0, 3.5
SEQ_LEN, DILATIONS, KERNEL = 60, (1, 2, 4, 8), 3

# helpers
def pick_session(prefer_many_trials=True):
    # A representative labeled recording: the one with the most supervised epochs.
    paths = sorted(glob.glob(os.path.join(LABELED, '*.pt')))
    if not paths:
        return None, None
    best, best_n = None, -1
    for p in paths:
        d = torch.load(p, weights_only=False)
        n = int(d['preonset_mask'].sum()) if 'preonset_mask' in d else 0
        if n > best_n:
            best, best_n = p, n
        if not prefer_many_trials:
            break
    return best, torch.load(best, weights_only=False)

def load_results(prefer=None):
    # The CNN run to illustrate. Defaults to the best configuration by per-session r.
    if prefer:
        cand = os.path.join(RESULTS, f'results_{prefer}.json')
        if os.path.exists(cand):
            return _read(cand)
    best, best_score = None, -np.inf
    paths = (glob.glob(os.path.join(RESULTS, 'results_MLP-TCN_*.json'))
             + glob.glob(os.path.join(RESULTS, 'results_CNN-TCN_*.json')))
    for path in sorted(paths):
        folds, cfg = _read(path)
        if not folds:
            continue
        psr = [r for fr in folds for r in fr.get('per_session_r', [])]
        score = float(np.mean(psr)) if psr else -np.inf
        if score > best_score:
            best, best_score = (folds, cfg), score
    return best if best else (None, None)

def _read(path):
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, dict) and 'folds' in raw:
        return raw['folds'], raw.get('config', {})
    return raw, {}

# Fig 05
def fig1():
    # Two representations of one 4 s epoch: the 51-feature vector vs the raw waveform.
    path, d = pick_session()
    if d is None:
        print("  fig1 skipped — no labeled .pt"); return
    if 'raw_epochs' not in d:
        print("  fig1 skipped — no raw_epochs (re-run script 1 with SAVE_RAW_EPOCHS=True)"); return
    names = d['feature_names']
    mask = d.get('preonset_mask')
    ei = int(np.argmax(mask.numpy())) if mask is not None and mask.sum() else len(names)
    feats = d['features_norm'].numpy()[ei]
    raw = d['raw_epochs'][ei].float().numpy()
    chs = d.get('raw_ch_names', [f'ch{i}' for i in range(raw.shape[0])])

    # entropy features are the tail of the vector; band-power/ratios lead it
    is_ent = np.array([n.startswith(('samp_ent', 'perm_ent', 'svd_ent', 'spec_ent')) for n in names])

    apply_style()
    fig = plt.figure(figsize=(13, 7.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.32], hspace=.34,
                          left=.085, right=.985, top=.9, bottom=.075)
    axA = fig.add_subplot(gs[0])
    colors = [GOLD if e else NAVY for e in is_ent]
    axA.bar(np.arange(len(feats)), feats, color=colors, width=.78)
    axA.axhline(0, color=MUTED, lw=.8)
    grid(axA); despine(axA)
    axA.set_xlim(-.8, len(feats) - .2)
    axA.set_ylabel('z-score')
    axA.set_title('A · 51 engineered features — the vector the MLP sees',
                  loc='left', fontsize=12.5, fontweight='semibold', pad=8)
    nb = int((~is_ent).sum())
    axA.axvspan(-.8, nb - .5, color=NAVY, alpha=.045, zorder=0)
    axA.text(nb / 2 - .5, axA.get_ylim()[1] * .93, f'band power & ratios ({nb})',
             ha='center', fontsize=9.5, color=NAVY)
    axA.text((nb + len(feats)) / 2, axA.get_ylim()[1] * .93,
             f'entropy ({int(is_ent.sum())} = 4 families × 10 electrodes)',
             ha='center', fontsize=9.5, color='#7A6224')
    axA.set_xticks([]); axA.tick_params(axis='x', length=0)
    axB = fig.add_subplot(gs[1])
    t = np.arange(raw.shape[1]) / 125.0
    off = 2.9
    for i in range(raw.shape[0]):
        s = raw[i] / (np.abs(raw[i]).max() + 1e-9)
        axB.plot(t, s + (raw.shape[0] - 1 - i) * off, color=STEEL, lw=1.0)
    axB.set_yticks([(raw.shape[0] - 1 - i) * off for i in range(raw.shape[0])])
    axB.set_yticklabels(chs, fontsize=9.5)
    axB.set_xlim(0, EPOCH_LENGTH); axB.set_xlabel('time within epoch (s)')
    despine(axB)
    axB.set_title('B · the same epoch as signal — what the 1-D CNN convolves',
                  loc='left', fontsize=12.5, fontweight='semibold', pad=8)
    axB.text(.995, .015, '10 electrodes × 500 samples · 125 Hz · fp16',
             transform=axB.transAxes, ha='right', va='bottom', fontsize=9.5, color=MUTED)

    fig.text(.5, .968, f'One 4-second epoch, two representations   ·   {os.path.basename(path)[:-7]}'
                       f'   ·   epoch {ei}',
             ha='center', fontsize=13.5, fontweight='semibold', color=INK)
    fig.text(.014, .5, 'same epoch', rotation=90, va='center', ha='center',
             fontsize=10.5, color=GOLD, fontweight='semibold')
    fig.patches.append(plt.Rectangle((.031, .09), .004, .82, transform=fig.transFigure,
                                     color=GOLD, alpha=.5, zorder=5))
    save(fig, os.path.join(OUT, 'fig1_two_representations.png'))

# Fig 06
def fig2():
    # The two label modalities over one drive.
    path, d = pick_session()
    if d is None:
        print("  fig2 skipped — no labeled .pt"); return
    need = ('preonset_mask', 'preonset_rt', 'smoothed_mask', 'smoothed_rt')
    if any(k not in d for k in need):
        print("  fig2 skipped — run 2_label_extraction.py with LABEL_MODE='both'"); return

    pm = d['preonset_mask'].numpy().astype(bool)
    pr = d['preonset_rt'].numpy()
    sm = d['smoothed_mask'].numpy().astype(bool)
    sr = d['smoothed_rt'].numpy()
    n = len(pm)
    t = (np.arange(n) * HOP + EPOCH_LENGTH / 2) / 60.0

    apply_style()
    fig, ax = plt.subplots(figsize=(13, 6.0))
    ax.scatter(t[pm], pr[pm], s=17, color=STEEL, alpha=.75, zorder=3,
               label='per-trial log₁₀ RT  (A: event-locked truth)')
    ax.plot(t[sm], sr[sm], color=GOLD, lw=2.3, zorder=4,
            label='60 s Gaussian-smoothed target  (B: dense envelope)')

    ymin = min(pr[pm].min(), sr[sm].min()) if pm.sum() and sm.sum() else 0
    ymax = max(pr[pm].max(), sr[sm].max()) if pm.sum() and sm.sum() else 1
    pad = (ymax - ymin) * .10
    ax.set_ylim(ymin - pad, ymax + pad)

    grid(ax); despine(ax)
    ax.set_xlabel('time into drive (min)')
    ax.set_ylabel('log₁₀ reaction time')
    ax.set_title(f'Two supervision signals from one drive   ·   {os.path.basename(path)[:-7]}',
                 loc='left', fontsize=13, fontweight='semibold', pad=10)
    # below the axes: anywhere inside them overlaps either the scatter or the kernel inset
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.13), ncol=2, frameon=False, fontsize=10)
    ax.text(.995, 1.012, f'A: {int(pm.sum())} labels ({100 * pm.mean():.0f}% of epochs)'
                         f'   ·   B: {int(sm.sum())} ({100 * sm.mean():.0f}%)',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=10, color=MUTED)

    # Gaussian kernel inset
    ins = ax.inset_axes([.79, .60, .19, .30])
    g = np.linspace(-150, 150, 300)
    ins.plot(g, np.exp(-.5 * (g / 60.0) ** 2), color=GOLD, lw=1.8)
    ins.fill_between(g, np.exp(-.5 * (g / 60.0) ** 2), color=GOLD, alpha=.16)
    ins.set_title('kernel · bw 60 s', fontsize=8.5, color=MUTED, pad=3)
    ins.set_xticks([-120, 0, 120]); ins.set_xticklabels(['−2', '0', '+2'], fontsize=7.5)
    ins.set_yticks([]); despine(ins, keep=('bottom',))
    ins.set_xlabel('min', fontsize=7.5, labelpad=1)
    save(fig, os.path.join(OUT, 'fig2_label_modalities.png'))

# Fig 12
def fig5():
    # Predicted vs true, and the per-session correlation spread.
    folds, cfg = load_results()
    if not folds:
        print("  fig5 skipped — no results JSON"); return
    P, T, PS = [], [], []
    for fr in folds:
        P += fr.get('pred', []); T += fr.get('true', [])
        PS += fr.get('per_session_r', [])
    P, T, PS = np.array(P), np.array(T), np.array(PS)
    if P.size == 0:
        print("  fig5 skipped — results contain no predictions"); return

    from scipy.stats import pearsonr
    r = pearsonr(P, T)[0]
    mae = float(np.mean(np.abs(P - T)))

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.9),
                             gridspec_kw=dict(width_ratios=[1, 1.18], wspace=.22))

    ax = axes[0]
    ax.scatter(T, P, s=9, color=STEEL, alpha=.28, edgecolors='none', zorder=2)
    lo, hi = min(T.min(), P.min()), max(T.max(), P.max())
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1.1, ls='--', zorder=3, label='identity')
    b, a = np.polyfit(T, P, 1)
    xs = np.linspace(lo, hi, 50)
    ax.plot(xs, a + b * xs, color=GOLD, lw=2.2, zorder=4, label='fit')
    ax.set_xlabel('true log₁₀ RT'); ax.set_ylabel('predicted log₁₀ RT')
    ax.set_title(f'A · pooled predictions   ·   $r$ = {r:.3f}   ·   MAE = {mae:.3f}',
                 loc='left', fontsize=12.5, fontweight='semibold', pad=10)
    ax.legend(loc='upper left'); grid(ax, axis='both'); despine(ax)
    ax.set_aspect('equal', adjustable='box')

    ax = axes[1]
    order = np.argsort(PS)
    v = PS[order]
    ax.bar(np.arange(len(v)), v, color=[GOLD if x > 0 else CRIMSON for x in v], width=.8)
    ax.axhline(0, color=INK, lw=.9)
    ax.axhline(v.mean(), color=NAVY, lw=1.4, ls='--',
               label=f'mean {v.mean():.3f}')
    ax.axhline(np.median(v), color=STEEL, lw=1.2, ls=':',
               label=f'median {np.median(v):.3f}')
    ax.set_xlabel('held-out recording (sorted)')
    ax.set_ylabel('within-session Pearson $r$')
    ax.set_title(f'B · per-session correlation   ·   {100 * np.mean(v > 0):.0f}% positive '
                 f'({int(np.sum(v > 0))}/{len(v)})',
                 loc='left', fontsize=12.5, fontweight='semibold', pad=10)
    ax.legend(loc='upper left'); grid(ax); despine(ax)
    ax.set_xticks([])

    cf = cfg or {}
    fig.suptitle(f"Prediction quality   ·   {cf.get('data_mode', 'features')} · "
                 f"{cf.get('label_mode', 'event_locked')} · {cf.get('eval_mode', 'within_subject')}",
                 x=.048, ha='left', fontsize=13.5, fontweight='semibold', color=INK)
    fig.subplots_adjust(top=.84)
    save(fig, os.path.join(OUT, 'fig5_pred_vs_true.png'))


FIGURES = {1: fig1, 2: fig2, 5: fig5}

def main():
    want = [int(a) for a in sys.argv[1:] if a.isdigit()] or sorted(FIGURES)
    os.makedirs(OUT, exist_ok=True)
    print(f"building {len(want)} figure(s) -> {os.path.relpath(OUT, HERE)}/")
    for n in want:
        fn = FIGURES.get(n)
        if fn is None:
            print(f"  no figure {n}"); continue
        try:
            fn()
        except Exception as e:
            print(f"  fig{n:02d} FAILED: {type(e).__name__}: {e}")
    print("\nFigure 3 comes from analysis/feature_selection.py, figure 4 from 4_compare_models.py.")

if __name__ == '__main__':
    main()
