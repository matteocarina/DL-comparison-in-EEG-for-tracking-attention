"""Greedy forward feature selection, shared by the model and the analysis scripts.

Selection never sees the recordings it will be judged on. Ridge regression stands in for the TCN
because a greedy sweep is thousands of fits: seconds with ridge, days with a neural network."""

import numpy as np
import torch
from scipy.stats import pearsonr

RIDGE_ALPHA = 1.0

# The hand-designed band-power core: theta/beta ratio, frontal-midline theta, alpha/theta ratio,
# posterior theta. These are ranked among themselves and the weakest are dropped, but they are
# never forced to compete against the 130 entropy/connectivity features on equal terms.
FIXED_PREFIXES = ('TBR_', 'FMT_', 'ATR_', 'theta_')

def build_pool(d):
    """The 141-feature pool for one recording: 51 engineered, then the two wPLI upper triangles.

    Column order is part of the contract -- saved feature indices are meaningless if it changes.
    """
    feats = d['features_norm'].numpy().astype(np.float64)
    names = list(d['feature_names'])
    chs = d.get('graph_ch_names', [f'n{i}' for i in range(d['wpli_theta'].shape[1])])
    iu = np.triu_indices(len(chs), k=1)
    blocks = [feats]
    for band in ('theta', 'alpha'):
        w = d[f'wpli_{band}'].numpy().astype(np.float64)
        blocks.append(w[:, iu[0], iu[1]])
        names = names + [f'wpli_{band[0]}_{chs[a]}-{chs[b]}' for a, b in zip(*iu)]
    return np.concatenate(blocks, axis=1), names

def load_pool(paths, mask_key, rt_key):
    """{path: (X, y, valid)} plus the feature names, for the given label keys."""
    store, names = {}, None
    for p in paths:
        d = torch.load(p, weights_only=False)
        X, nm = build_pool(d)
        names = names or nm
        store[p] = (X, d[rt_key].numpy().astype(np.float64),
                    d[mask_key].numpy().astype(bool))
    return store, names

def fixed_indices(names):
    return sorted(j for j, n in enumerate(names) if n.startswith(FIXED_PREFIXES))

def ridge_fit(X, y, alpha=RIDGE_ALPHA):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    ybar = y.mean()
    w = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (y - ybar))
    return mu, sd, w, ybar

def stack(paths, store, cols):
    Xs, ys = [], []
    for p in paths:
        X, y, v = store[p]
        if v.sum() == 0:
            continue
        Xs.append(X[v][:, cols]); ys.append(y[v])
    return np.concatenate(Xs), np.concatenate(ys)

def ridge_score(model, paths, store, cols):
    """Mean per-session Pearson r -- the same metric the TCN is judged on."""
    mu, sd, w, ybar = model
    rs = []
    for p in paths:
        X, y, v = store[p]
        if v.sum() < 5:
            continue
        pred = ((X[v][:, cols] - mu) / sd) @ w + ybar
        true = y[v]
        if np.std(pred) > 1e-12 and np.std(true) > 1e-12:
            rs.append(pearsonr(pred, true)[0])
    return float(np.mean(rs)) if rs else np.nan

def greedy(train, val, store, candidates, start, n_steps, scorer=None):
    """Add n_steps features from `candidates` to `start`, each chosen to maximise validation r.

    `scorer(cols) -> float` overrides the default fit-on-train/score-on-val, so the same search can
    be driven by a cross-validated score (see cv_scorer) instead of a single split.
    """
    if scorer is None:
        def scorer(cols):
            return ridge_score(ridge_fit(*stack(train, store, cols)), val, store, cols)
    chosen = list(start)
    remaining = [j for j in candidates if j not in chosen]
    for _ in range(n_steps):
        best_j, best_r = None, -np.inf
        for j in remaining:
            r = scorer(chosen + [j])
            if np.isfinite(r) and r > best_r:
                best_j, best_r = j, r
        if best_j is None:
            break
        chosen.append(best_j); remaining.remove(best_j)
    return chosen

def cv_scorer(paths, store, n_splits=4):
    """Score a feature subset by K-fold CV across recordings within `paths`.

    Needed for the holdout selection. Fitting and scoring on the same recordings -- which is what
    passing the holdout set as both train and val does -- makes the greedy search monotonically
    reward in-sample fit, so with 130 candidates it will happily add pure noise. Splitting the
    holdout into disjoint fit/score groups makes every candidate earn its place on unseen data.
    No leakage either way: these recordings are never in a test set.

    """
    groups = [paths[i::n_splits] for i in range(n_splits)]
    groups = [g for g in groups if g]

    def score(cols):
        rs = []
        for k, te in enumerate(groups):
            tr = [p for i, g in enumerate(groups) if i != k for p in g]
            if not tr:
                continue
            r = ridge_score(ridge_fit(*stack(tr, store, cols)), te, store, cols)
            if np.isfinite(r):
                rs.append(r)
        return float(np.mean(rs)) if rs else np.nan
    return score

def select_for_holdout(paths, store, names, n_core, n_extra, n_splits=4):
    """Feature indices chosen once on the holdout recordings, cross-validated among themselves.

    Same two-stage shape as select_for_fold -- rank the hand-designed core among itself, keep the
    best n_core, then greedily add n_extra from the rest -- but every candidate is scored by
    cv_scorer, so the choice is not just whichever features fit these recordings best.
    """
    scorer = cv_scorer(paths, store, n_splits)
    core = fixed_indices(names)
    pool = [j for j in range(len(names)) if j not in core]
    core_ranked = greedy(None, None, store, core, [], len(core), scorer=scorer)
    keep = core_ranked[:n_core]
    return sorted(greedy(None, None, store, pool, keep, n_extra, scorer=scorer)), core_ranked

def select_for_fold(train, val, store, names, n_core, n_extra):
    """Feature indices for one fold: the best n_core of the hand-designed core, plus n_extra more.

    Called from inside the cross-validation loop with that fold's train/val recordings only. The
    core is ranked among itself first so the domain features compete on their own terms rather
    than being crowded out by 130 entropy and connectivity candidates.
    """
    core = fixed_indices(names)
    pool = [j for j in range(len(names)) if j not in core]
    core_ranked = greedy(train, val, store, core, [], len(core))
    keep = core_ranked[:n_core]
    return sorted(greedy(train, val, store, pool, keep, n_extra)), core_ranked
