"""Collect every results file into one table and draw the headline comparison.
Per-session r is the headline metric: pooled r mixes tracking reaction time within a drive with
ranking whole drives against each other, while per-session r removes each drive's baseline.
Confidence intervals resample subjects, not sessions."""

import csv
import glob
import json
import os
import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from lib.plot_style import NAVY, STEEL, GOLD, GRAY, CRIMSON, INK, MUTED, apply_style, grid, despine, save
from lib import analysis_utils as au

matplotlib.use('Agg')
HERE = os.path.dirname(os.path.abspath(__file__))
CNN_DIR = os.path.join(HERE, 'eeg_data_local')
OUT_DIR = os.path.join(HERE, 'figures')
GNN_DIR = CNN_DIR
VARIANT_LABEL = {'all51': 'all 51 feats'}
# translating internal identifiers into reader-facing text for the results table
PRETTY_INPUT = {'features': '15 selected feats', 'raw': 'raw 10x1000', 'graph': 'feats + wPLI graph',
                'graph, no time': 'wPLI graph, one epoch', 'elapsed time': 'elapsed time (no EEG)'}
PRETTY_LABEL = {'event_locked': 'event-locked', 'smoothed': 'smoothed'}
PRETTY_EVAL = {'within_subject': 'within-subject', 'cross_subject': 'cross-subject'}
MODEL_CAVEAT = (
    "NOTE  GNN-LSTM selects its reported checkpoint on the TEST score (peak-on-test), while both\n"
    "      TCN rows select on a held-out VALIDATION set and touch test exactly once. The GNN's\n"
    "      numbers are therefore optimistically biased as an upper bound, not an equal comparison.")

def load_payload(path):
    # Read one results file, whichever of the two layouts it uses, as (folds, config). The TCN writes
    # {'config': {...}, 'folds': [...]}; the graph models write the bare list of folds with no config.
    # The per-fold dictionaries are identical either way, so normalising here is what lets everything
    # below treat all four producers the same and keeps the difference from leaking into the table, the
    # figures and the staleness check.
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, dict) and 'folds' in raw:
        return raw['folds'], raw.get('config', {})
    return raw, {}

def summarize(folds):
    # One run's 5 folds collapsed into the row the table prints. Pooled r and MAE are already one number
    # per fold, so they average over 5 values. Per-session r is a LIST per fold so concatenated into all
    # 95 drive scores before any statistic is taken.
    def col(key):
        # a metric stored once per fold; missing or null entries are skipped rather than crashing
        return [f[key] for f in folds if key in f and f[key] is not None]
    per_session = []
    for f in folds:
        per_session += f.get('per_session_r', [])   # flatten 5 lists into every drive's score
    pooled = col('pearson_r')
    mae = col('mae')
    sp = au.spread(per_session)                     # mean, sd, quartiles, min/max, % positive
    return {
        'pooled_r': float(np.mean(pooled)) if pooled else np.nan,
        'pooled_r_sd': float(np.std(pooled)) if pooled else np.nan,
        'per_session_r': sp.get('mean', np.nan),
        # SD across recordings is the spread a reader actually cares about: it says how much
        # performance varies from drive to drive, and here it dwarfs every between-model gap.
        'per_session_r_sd': sp.get('sd', np.nan),
        'per_session_r_iqr': sp.get('iqr', np.nan),
        'per_session_r_q1': sp.get('q1', np.nan),
        'per_session_r_q3': sp.get('q3', np.nan),
        'per_session_r_min': sp.get('min', np.nan),
        'per_session_r_max': sp.get('max', np.nan),
        'per_session_pos': sp.get('frac_positive', np.nan),
        # the interval needs each drive mapped back to its subject, which only collect() can do
        'ci_lo': np.nan, 'ci_hi': np.nan,
        'mae': float(np.mean(mae)) if mae else np.nan,
        'n_folds': len(folds),
        'n_sessions': len(per_session), }

def collect():
    # Every result file in eeg_data_local/, parsed by one rule.
    rows = []
    # It reads the whole experimental configuration out of the filename. Every results file is named
    # results_{model}_{labels}_{protocol}[_suffix].json, and this pattern takes it apart into four pieces
    pat = re.compile(r'results_(MLP-TCN|CNN-TCN|GNN-LSTM|GNN|TIME)_'
                     r'(event_locked|smoothed)_(within_subject|cross_subject)(_.+)?$')
    INPUT_OF = {'MLP-TCN': 'features', 'CNN-TCN': 'raw', 'GNN-LSTM': 'graph',
                'GNN': 'graph, no time', 'TIME': 'elapsed time'}
    for path in sorted(glob.glob(os.path.join(CNN_DIR, 'results_*.json'))):
        m = pat.match(os.path.basename(path)[:-len('.json')])
        if not m:
            continue                       # stale or foreign file: ignore rather than guess
        model, label, ev, suffix = m.groups()
        folds, cfg = load_payload(path)
        if not folds:
            continue
        row = summarize(folds)
        row.update(model=model, input=INPUT_OF[model], label=label, eval=ev,
                   variant=(suffix or '').lstrip('_'),
                   n_params=cfg.get('n_params') or (373538 if model == 'GNN-LSTM' else None),
                   file=os.path.basename(path))
        by_subj = (au.gnn_per_session_by_subject(label, ev, model) if model.startswith('GNN')
                   else au.per_session_by_subject(model, label, ev, row['variant'])
                   if model != 'TIME' else None)
        if by_subj:
            lo, hi = au.bootstrap_ci_by_subject(by_subj)
            row['ci_lo'], row['ci_hi'] = lo, hi
            row['n_subjects'] = len(by_subj)
        rows.append(row)
    order = {'within_subject': 0, 'cross_subject': 1}
    rank = {'MLP-TCN': 0, 'CNN-TCN': 1, 'GNN': 2, 'GNN-LSTM': 3, 'TIME': 4}
    # Group by configuration, then protocol -- so a model's enrolled and unseen rows sit next to
    # each other and the transfer gap is readable across two lines instead of two blocks.
    # 'per_fold' marks where the selection ran, not which features were used, so it must not
    # separate a configuration from its own other protocol.
    config = lambda r: r['variant'].replace('per_fold', '').strip('_')
    rows.sort(key=lambda r: (r['label'], rank.get(r['model'], 9), config(r),
                             order.get(r['eval'], 9)))
    return rows

def fmt(v, nd=3): # Rounding to 3 decimals
    return '—' if v is None or (isinstance(v, float) and np.isnan(v)) else f'{v:.{nd}f}'

def input_label(r):
    # The text for the row's 'input' column. The type check is what makes the suffix mean the right thing.
    base = PRETTY_INPUT.get(r['input'], r['input'])
    if r['input'] != 'features':
        return base
    return VARIANT_LABEL.get(r.get('variant', ''), base)

def print_table(rows): # Shows results in the terminal
    hdr = (f"{'model':10s} {'input':22s} {'labels':14s} {'protocol':15s} "
           f"{'pooled r':>9s} {'per-sess':>9s} {'SD':>6s} {'95% CI (subj)':>16s} "
           f"{'MAE':>7s} {'params':>9s}")
    print('\n' + '=' * len(hdr))
    print('RESULTS' + ' ' * (len(hdr) - 7))
    print('=' * len(hdr))
    print(hdr)
    print('-' * len(hdr))
    last_label = None
    for r in rows:
        if last_label is not None and r['label'] != last_label:
            print('-' * len(hdr))
        last_label = r['label']
        params = f"{r['n_params']:,}" if r.get('n_params') else '—'
        ci = (f"[{r['ci_lo']:.3f},{r['ci_hi']:.3f}]"
              if np.isfinite(r.get('ci_lo', np.nan)) else '—')
        print(f"{r['model']:10s} {input_label(r):22s} "
              f"{PRETTY_LABEL.get(r['label'], r['label']):14s} "
              f"{PRETTY_EVAL.get(r['eval'], r['eval']):15s} "
              f"{fmt(r['pooled_r']):>9s} {fmt(r['per_session_r']):>9s} "
              f"{fmt(r['per_session_r_sd']):>6s} {ci:>16s} "
              f"{fmt(r['mae']):>7s} {params:>9s}")
    print('=' * len(hdr))

    if any(r['model'] == 'GNN-LSTM' for r in rows):
        print(MODEL_CAVEAT)

    missing = expected_missing()
    if missing:
        print("\nnot yet run:")
        for model, variant, l, e in missing:
            print(f"   {model:9s} {variant:20s} {l} / {e}      -> python run_matrix.py")

def expected_missing():
    # Tells which configurations still have no results. It asks run_matrix.grid() for every run the
    # experiment defines, checks whether each one's file exists and is complete, and returns the gaps.
    import run_matrix
    missing = []
    for r in run_matrix.grid():
        if not run_matrix.is_complete(r['path']):
            missing.append((r['model'], r.get('variant', ''), r['label'], r['eval']))
    return missing

def write_table_files(rows): # to file
    os.makedirs(OUT_DIR, exist_ok=True)
    cols = ['model', 'input', 'label', 'eval', 'pooled_r', 'pooled_r_sd',
            'per_session_r', 'per_session_r_sd', 'per_session_r_iqr',
            'per_session_r_q1', 'per_session_r_q3', 'per_session_r_min',
            'per_session_r_max', 'ci_lo', 'ci_hi', 'per_session_pos', 'mae',
            'n_folds', 'n_sessions', 'n_subjects', 'n_params']
    csv_path = os.path.join(OUT_DIR, 'results_table.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)
    md_path = os.path.join(OUT_DIR, 'results_table.md')
    with open(md_path, 'w') as f:
        f.write('| Model | Input | Labels | Protocol | Pooled r | Per-session r | MAE | Params |\n')
        f.write('|---|---|---|---|---|---|---|---|\n')
        for r in rows:
            params = f"{r['n_params']:,}" if r.get('n_params') else '—'
            f.write(f"| {r['model']} | {input_label(r)} "
                    f"| {PRETTY_LABEL.get(r['label'], r['label'])} "
                    f"| {PRETTY_EVAL.get(r['eval'], r['eval'])} "
                    f"| {fmt(r['pooled_r'])} | {fmt(r['per_session_r'])} "
                    f"| {fmt(r['mae'])} | {params} |\n")
        # The caveat travels with the table: this file is meant to be pasted on its own.
        if any(r['model'] == 'GNN-LSTM' for r in rows):
            f.write('\n> **GNN-LSTM selects its checkpoint on the test score** (peak-on-test), while\n'
                    '> both TCN rows select on a held-out validation set and touch test exactly once.\n'
                    '> The GNN numbers are an optimistic upper bound (~+0.026 per-session r), not a\n'
                    '> like-for-like comparison.\n')
    print(f"\nwrote {os.path.relpath(csv_path, HERE)}")
    print(f"wrote {os.path.relpath(md_path, HERE)}")

def figure_forest(rows):
    """Every configuration on one axis: mean per-session r, with a bar spanning +-1 SD.

    The SD across drives, not a confidence interval. A CI describes how precisely the mean is
    known and narrows with more subjects; a reader wants to know how much performance actually
    varies from drive to drive, which is three to five times wider and is what makes most of the
    between-model gaps here look smaller than they first appear.

    Both protocols are restricted to the 19 subjects with two or more recordings, since only those
    can be held out under within_subject -- otherwise the two columns would describe different
    populations.
    """
    import glob
    from collections import defaultdict
    paths = sorted(glob.glob(os.path.join(CNN_DIR, 'extracted_features_labeled', '*.pt')))
    cnt = defaultdict(int)
    for p in paths:
        cnt[au.subject_of(p)] += 1
    multi = {s for s, n in cnt.items() if n >= 2}

    def stat(by):
        if not by:
            return None, None
        v = np.array([r for s in by if s in multi for r in by[s]])
        return (v.mean(), v.std(ddof=1)) if v.size > 1 else (None, None)

    entries = []
    for r in rows:
        if r['model'] == 'TIME' or r['eval'] != 'within_subject':
            continue
        variant = r['variant']
        if r['model'].startswith('GNN'):
            w = au.gnn_per_session_by_subject(r['label'], 'within_subject', r['model'])
            c = au.gnn_per_session_by_subject(r['label'], 'cross_subject', r['model'])
        else:
            cvar = variant if variant else 'per_fold'
            w = au.per_session_by_subject(r['model'], r['label'], 'within_subject', variant)
            c = au.per_session_by_subject(r['model'], r['label'], 'cross_subject', cvar)
        wm, ws = stat(w)
        cm, cs = stat(c)
        if wm is None:
            continue
        entries.append(dict(name=f"{r['model']} · {input_label(r)}", label=r['label'],
                            wm=wm, ws=ws, cm=cm, cs=cs))
    if not entries:
        return
    entries.sort(key=lambda e: e['wm'])

    apply_style()
    fig, ax = plt.subplots(figsize=(11.5, 8.2))
    y = np.arange(len(entries), dtype=float)
    for i, e in enumerate(entries):
        for m, sd, col, off, lbl in ((e['wm'], e['ws'], NAVY, +.16, 'enrolled driver'),
                                     (e['cm'], e['cs'], CRIMSON, -.16, 'unseen driver')):
            if m is None:
                continue
            ax.plot([m - sd, m + sd], [y[i] + off] * 2, color=col, lw=2.4, alpha=.30,
                    solid_capstyle='round', zorder=2)
            ax.plot(m, y[i] + off, 'o', color=col, ms=7.5, zorder=3,
                    label=lbl if i == 0 else None)
    ax.axvline(0, color=MUTED, lw=1.0, ls=':', zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{e['name']}\n{'continuous' if e['label'] == 'smoothed' else 'event-locked'}"
                        for e in entries], fontsize=9.5)
    ax.set_xlabel('per-session r   (dot = mean across drives, bar = ±1 SD)')
    ax.set_xlim(-0.35, 0.92)
    grid(ax, axis='x'); despine(ax)
    ax.set_title('How well each model tracks reaction time within a drive',
                 loc='left', fontsize=13.5, fontweight='semibold', pad=12)
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.09), ncol=2, frameon=False, fontsize=10)
    save(fig, os.path.join(OUT_DIR, 'fig6_results_forest.png'))


def figure_headline(rows):
    # The headline chart: per-session r for every model, with subject-clustered error bars. Bars are
    # grouped TCN variants, then the graph models, then the no-EEG control, with parameter counts
    # annotated so the size/performance trade is visible at a glance.
    apply_style()
    # Both protocols now have a full grid, so both get a chart.
    for ev in ('within_subject', 'cross_subject'):
        sub = [r for r in rows if r['eval'] == ev and not np.isnan(r['per_session_r'])]
        if len(sub) < 2:
            continue
        cnn = [r for r in sub if r['model'].endswith('-TCN')]
        gnn = [r for r in sub if r['model'].startswith('GNN')]   # both graph models
        base = [r for r in sub if r['model'] == 'TIME']
        # the clock-only control belongs on this chart: without it the reader cannot tell how much
        # of any bar is EEG and how much is "drives get worse over time"
        ordered = cnn + gnn + base
        if not ordered:
            continue
        best = max(range(len(ordered)),
                   key=lambda i: (ordered[i]['model'] != 'TIME',
                                  ordered[i]['per_session_r']))
        labels, vals, errs, colors = [], [], [], []
        for i, r in enumerate(ordered):
            lab = 'event' if r['label'] == 'event_locked' else 'smooth'
            if r['model'] == 'TIME':
                labels.append(f'no EEG\nclock · {lab}')
            else:
                inp = ('feats' if r['input'] == 'features'
                       else 'raw' if r['input'] == 'raw' else 'graph')
                labels.append(f"{r['model']}\n{lab}")
            vals.append(r['per_session_r'])
            errs.append(r['per_session_r_sd'] / max(np.sqrt(r['n_sessions']), 1))
            if r['model'] == 'TIME':
                colors.append('#C9CFD8')
            elif i == best:
                colors.append(GOLD)
            elif r['model'] == 'GNN-LSTM':
                colors.append(GRAY)
            else:
                colors.append(NAVY if r['input'] == 'features' else STEEL)

        fig, ax = plt.subplots(figsize=(1.55 * len(ordered) + 2.4, 5.2))
        x = np.arange(len(ordered))
        bars = ax.bar(x, vals, yerr=errs, color=colors, width=.66,
                      error_kw=dict(ecolor=MUTED, lw=1.1, capsize=3))
        ax.axhline(0, color=INK, lw=.9)

        for i, (b, r) in enumerate(zip(bars, ordered)):
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, h + errs[i] + .012, f'{h:.3f}',
                    ha='center', va='bottom', fontsize=10.5,
                    fontweight='bold' if i == best else 'normal',
                    color=INK if i != best else '#6E5410')
            p = r.get('n_params')
            ax.text(b.get_x() + b.get_width() / 2, -.028,
                    (f"{p / 1000:.0f}k" if p and p >= 1000 else str(p) if p else '—'),
                    ha='center', va='top',
                    fontsize=9, color=MUTED)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel('per-session Pearson $r$')
        ax.set_title(f'Per-session correlation — {PRETTY_EVAL[ev]}, cross-recording',
                     loc='left', fontsize=13, fontweight='semibold', color=INK, pad=12)
        ax.set_xlabel('parameter count shown under each bar', fontsize=9.5,
                      color=MUTED, labelpad=20)
        top = max(v + e for v, e in zip(vals, errs))
        ax.set_ylim(min(-.06, min(vals) - .05), top * 1.24)
        ax.margins(x=.045)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis='x', length=0)
        save(fig, os.path.join(OUT_DIR, f'fig4_headline_{ev}.png'))

def main():
    au.warn_if_stale(sorted(glob.glob(os.path.join(CNN_DIR, 'results_*.json'))))
    rows = collect()
    if not rows:
        raise SystemExit(f"No result JSONs found in {CNN_DIR} — run 3b_model_tcn.py first.")
    print_table(rows)
    write_table_files(rows)
    figure_headline(rows)
    figure_forest(rows)

if __name__ == '__main__':
    main()
