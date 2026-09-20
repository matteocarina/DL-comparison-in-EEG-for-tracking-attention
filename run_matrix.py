"""Run every configuration in the experiment grid, each in a fresh process.
A separate interpreter per run means no state leaks between configurations, and every run writes its
own results file. This grid is the single source of truth for what the experiment contains."""

import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CNN_RESULTS = os.path.join(HERE, 'eeg_data_local')
CNN_SCRIPT = os.path.join(HERE, '3b_model_tcn.py')
GNN_HOME = HERE
GNN_RESULTS = CNN_RESULTS
GNN_SCRIPT = os.path.join(HERE, '3a_model_gnn_lstm.py')
LABEL_MODES = ('event_locked', 'smoothed')
# Within-subject only, testing on a driver absent from training entirely  is still supported by both
# models and can be swept by adding it back here, but it answers a different question.
# Both protocols. within_subject is the headline: a driver does a short calibration drive, then the
# system works for them. cross_subject is the harder question -- does it work for a driver never
# seen in training -- and is what published work on this dataset reports.
EVAL_MODES = ('within_subject', 'cross_subject')

def selection_source(data, ev, feature_set):
    """Where the feature selection may look. Only feature runs select at all.

    Under cross_subject the single-recording subjects appear in test sets, so the fixed holdout
    list would leak; selection has to be redone inside each fold instead.
    """
    if data != 'features' or feature_set != 'selected':
        return None
    return 'per_fold' if ev == 'cross_subject' else 'holdout'

def cnn_result_path(data, label, ev, feature_set='selected'):
    model = 'MLP-TCN' if data == 'features' else 'CNN-TCN'
    suffix = '' if feature_set == 'selected' else '_all51'
    if selection_source(data, ev, feature_set) == 'per_fold':
        suffix += '_per_fold'          # 3b_model_tcn.results_path() tags non-default sources
    return os.path.join(CNN_RESULTS, f'results_{model}_{label}_{ev}{suffix}.json')

def gnn_result_path(label, ev, arch='GNN-LSTM'):
    # Same scheme as the TCN: results_{model}_{label}_{eval}.json.
    return os.path.join(GNN_RESULTS, f'results_{arch}_{label}_{ev}.json')

def grid():
    # Every run as (model, input, label, eval, script, cwd, env, result_path).
    runs = []
    for data in ('features', 'raw'):
        fsets = ('selected', 'all') if data == 'features' else ('all',)
        for fset in fsets:
            for label in LABEL_MODES:
                for ev in EVAL_MODES:
                    runs.append(dict(
                        model=('MLP-TCN' if data == 'features' else 'CNN-TCN'),
                        input=data, label=label, eval=ev,
                        variant=('raw waveform' if data == 'raw'
                                 else '15 selected feats' if fset == 'selected'
                                 else 'all 51 feats'),
                        script=CNN_SCRIPT, cwd=HERE,
                        env={'CNN_DATA_MODE': data, 'CNN_LABEL_MODE': label,
                             'CNN_EVAL_MODE': ev, 'CNN_FEATURE_SET': fset,
                             **({'CNN_SELECTION_SOURCE': src}
                                if (src := selection_source(data, ev, fset)) else {})},
                        path=cnn_result_path(data, label, ev, fset)))
    for arch, tag, variant in (('GNN_LSTM', 'GNN-LSTM', '~41 feats (own RF)'),
                               ('GNN', 'GNN', 'one epoch, no time')):
        for label in LABEL_MODES:
            for ev in EVAL_MODES:
                runs.append(dict(
                    model=tag, input='graph', label=label, eval=ev, variant=variant,
                    script=GNN_SCRIPT, cwd=GNN_HOME,
                    env={'GNN_LABEL_MODE': label, 'GNN_EVAL_MODE': ev,
                         'GNN_ARCHITECTURE': arch},
                    path=gnn_result_path(label, ev, tag)))
    return runs

def is_complete(path):
    # The TCN checkpoints after each fold, so a results file can exist for an interrupted run.
    # Its payload carries a 'complete' flag, which is what gets checked.
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            payload = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    if isinstance(payload, list):                      # the graph models' shape: folds, no config
        return len(payload) > 0
    cfg = payload.get('config', {})
    if 'complete' in cfg:
        return bool(cfg['complete'])
    return len(payload.get('folds', [])) > 0

def status_of(path):
    if not os.path.exists(path):
        return 'missing'
    return 'done' if is_complete(path) else 'partial'

def matches(run, only):
    if only is None:
        return True
    o = only.lower()
    hay = f"{run['model']} {run['input']} {run['label']} {run['eval']}".lower()
    if o == 'gnn':                       # both graph models
        return run['model'].startswith('GNN')
    if o in ('gnn-lstm', 'lstm'):
        return run['model'] == 'GNN-LSTM'
    if o in ('gnn-only', 'spatial'):
        return run['model'] == 'GNN'
    if o == 'tcn':
        return run['model'].endswith('-TCN')
    if o == 'cnn':
        return run['model'] == 'CNN-TCN'
    if o == 'mlp':
        return run['model'] == 'MLP-TCN'
    return o in hay

def main():
    args = sys.argv[1:]
    force = '--force' in args
    only = args[args.index('--only') + 1] if '--only' in args else None
    runs = [r for r in grid() if matches(r, only)]

    if '--list' in args:
        print(f"{'model':10s} {'variant':20s} {'labels':14s} {'protocol':16s} status")
        print('-' * 74)
        last = None
        for r in grid():
            if last and r['model'] != last:
                print('-' * 74)
            last = r['model']
            print(f"{r['model']:10s} {r.get('variant',''):20s} {r['label']:14s} "
                  f"{r['eval']:16s} {status_of(r['path'])}")
        return

    todo = [r for r in runs if force or not is_complete(r['path'])]
    if not todo:
        print("Every requested configuration already has results. Use --force to re-run.")
        return

    print(f"{len(todo)} run(s) queued:")
    for r in todo:
        print(f"   {r['model']:10s} {r.get('variant',''):20s} {r['label']:14s} {r['eval']}")
    clobber = [r for r in todo if r['model'] == 'GNN-LSTM'
               and r['label'] == 'event_locked' and os.path.exists(r['path'])]
    if clobber:
        print("\n   NOTE: these overwrite the GNN's original result files:")
        for r in clobber:
            print(f"     {os.path.basename(r['path'])}")
    print()

    failed = []
    for i, r in enumerate(todo, 1):
        env = dict(os.environ, **r['env'])
        print('=' * 72)
        print(f"[{i}/{len(todo)}]  {r['model']} ({r.get('variant','')})  "
              f"{r['label']} / {r['eval']}")
        print('=' * 72, flush=True)
        t0 = time.time()
        rc = subprocess.run([sys.executable, r['script']], env=env, cwd=r['cwd']).returncode
        dt = (time.time() - t0) / 60
        if rc != 0:
            failed.append((r, rc))
            print(f"  FAILED (exit {rc}) after {dt:.1f} min\n")
        else:
            print(f"  done in {dt:.1f} min\n")

    print('=' * 72)
    print(f"finished: {len(todo) - len(failed)}/{len(todo)} succeeded")
    for r, rc in failed:
        print(f"  FAILED  {r['model']} {r['input']}/{r['label']}/{r['eval']}  (exit {rc})")
    print("\nNext: python 4_compare_models.py")

if __name__ == '__main__':
    main()
