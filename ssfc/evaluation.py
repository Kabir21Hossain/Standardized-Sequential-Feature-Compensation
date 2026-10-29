"""Four-arm comparison used by every notebook.

Arms (all inside a scikit-learn Pipeline, so nothing leaks across CV folds):

  A_naive_all_features     all features, standardized                (baseline)
  B_drop_no_compensation   PCA-impact selects k features, drop them   (no compensation)
  C_compensate_then_drop   the SAME k features, compensate, then drop (our idea)
  D_pca_components         PCA keeps d-k components                   (classic reduction)

B and C remove exactly the same features, so C - B isolates the effect of
compensation. D is the standard dimensionality-reduction reference at the same
width as B and C.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .core import SSFC

ARM_A = "A_naive_all_features"
ARM_B = "B_drop_no_compensation"
ARM_C = "C_compensate_then_drop"
ARM_D = "D_pca_components"
ARMS = [ARM_A, ARM_B, ARM_C, ARM_D]


def n_to_remove(d, frac_remove):
    """Number of features to remove: at least 1, at most d-1."""
    return int(max(1, min(d - 1, round(frac_remove * d))))


def make_transformers(n_remove, n_features, alpha=1.0, beta=1.0, selection="pca_low", remove_idx=None):
    """The four feature-preparation steps (no model attached)."""
    return {
        ARM_A: StandardScaler(),
        ARM_B: SSFC(n_remove=n_remove, selection=selection, remove_idx=remove_idx, compensate=False),
        ARM_C: SSFC(n_remove=n_remove, selection=selection, remove_idx=remove_idx, compensate=True,
                    alpha=alpha, beta=beta),
        ARM_D: Pipeline([("scale", StandardScaler()),
                         ("pca", PCA(n_components=n_features - n_remove))]),
    }


def make_arms(model, n_remove, n_features, alpha=1.0, beta=1.0, selection="pca_low", remove_idx=None):
    """Pipelines = preparation step + a fresh copy of ``model``."""
    return {name: Pipeline([("prep", t), ("model", clone(model))])
            for name, t in make_transformers(n_remove, n_features, alpha, beta, selection, remove_idx).items()}


def _scoring(y):
    multi = len(np.unique(y)) > 2
    return {"auc": "roc_auc_ovr" if multi else "roc_auc", "acc": "accuracy"}


def run_comparison(X, y, model, n_remove, alpha=1.0, beta=1.0, selection="pca_low",
                   n_splits=5, n_repeats=2, random_state=42, extra_arms=None, n_jobs=1, remove_idx=None):
    """Cross-validate every arm on identical folds.

    Returns ``(table, folds)``: a per-arm summary DataFrame and a dict with the
    per-fold AUC arrays (used for paired comparisons).
    """
    d = X.shape[1]
    arms = make_arms(model, n_remove, d, alpha, beta, selection, remove_idx)
    if extra_arms:
        arms.update(extra_arms)
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    scoring = _scoring(y)
    rows, folds = [], {}
    for name, est in arms.items():
        res = cross_validate(est, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs, error_score="raise")
        folds[name] = res["test_auc"]
        rows.append(dict(arm=name, auc_mean=res["test_auc"].mean(), auc_std=res["test_auc"].std(),
                         acc_mean=res["test_acc"].mean(), acc_std=res["test_acc"].std(),
                         n_features_used=d if name == ARM_A else d - n_remove))
    return pd.DataFrame(rows), folds


def run_benchmark(datasets, models, frac_remove=0.25, alpha=1.0, beta=1.0, selection="pca_low",
                  n_splits=5, n_repeats=2, random_state=42, extra_arms_factory=None, verbose=True):
    """Run ``run_comparison`` for every (dataset, model).

    ``extra_arms_factory(model, n_remove)`` may return additional arms (used in version 4).
    Returns ``(results, folds)``; ``results`` is a tidy DataFrame.
    """
    all_rows, all_folds = [], {}
    for dname, (X, y) in datasets.items():
        k = n_to_remove(X.shape[1], frac_remove)
        for mname, model in models.items():
            extra = extra_arms_factory(model, k) if extra_arms_factory else None
            table, folds = run_comparison(X, y, model, k, alpha, beta, selection,
                                          n_splits, n_repeats, random_state, extra)
            table.insert(0, "model", mname)
            table.insert(0, "dataset", dname)
            table["k_removed"] = k
            table["d"] = X.shape[1]
            table["selection"] = selection
            all_rows.append(table)
            for arm, f in folds.items():
                all_folds[(dname, mname, arm)] = f
            if verbose:
                print(f"done {dname:13s} {mname:22s} k={k}")
    return pd.concat(all_rows, ignore_index=True), all_folds


def paired_summary(results, folds, extra_arm=None):
    """One row per (dataset, model): arm AUCs and paired fold differences.

    ``comp_minus_drop`` (arm C minus arm B) is the effect of compensation itself
    because both remove the same features. The ``+- se`` is the standard error of
    the paired fold differences (folds overlap, so treat it as a rough guide).
    """
    rows = []
    for (dname, mname), g in results.groupby(["dataset", "model"], sort=False):
        a = g[g.arm == ARM_A].iloc[0]
        b = g[g.arm == ARM_B].iloc[0]
        c = g[g.arm == ARM_C].iloc[0]
        dd = g[g.arm == ARM_D].iloc[0]
        diff = folds[(dname, mname, ARM_C)] - folds[(dname, mname, ARM_B)]
        row = dict(dataset=dname, model=mname, d=int(a.d), k_removed=int(a.k_removed),
                   auc_naive=a.auc_mean, auc_drop=b.auc_mean, auc_comp=c.auc_mean, auc_pca=dd.auc_mean,
                   comp_minus_drop=diff.mean(), se=diff.std(ddof=1) / np.sqrt(len(diff)),
                   comp_wins_folds=float((diff > 0).mean()))
        if extra_arm is not None:
            e = g[g.arm == extra_arm].iloc[0]
            ediff = folds[(dname, mname, extra_arm)] - folds[(dname, mname, ARM_B)]
            row.update(auc_extra=e.auc_mean, extra_minus_drop=ediff.mean(),
                       extra_se=ediff.std(ddof=1) / np.sqrt(len(ediff)))
        rows.append(row)
    return pd.DataFrame(rows)


def verdict_counts(summary, col="comp_minus_drop", se_col="se"):
    """Count cases where compensation is better / within noise / worse than dropping."""
    better = int(((summary[col] > 0) & (summary[col].abs() > summary[se_col])).sum())
    worse = int(((summary[col] < 0) & (summary[col].abs() > summary[se_col])).sum())
    tie = len(summary) - better - worse
    return dict(better=better, within_noise=tie, worse=worse, total=len(summary))


def plot_arms(results, title="", ylim=None):
    """Grouped bars: one panel per model, one group per dataset, one bar per arm."""
    import matplotlib.pyplot as plt

    models = list(results.model.unique())
    datasets = list(results.dataset.unique())
    arms = list(results.arm.unique())
    colors = {ARM_A: "#16a34a", ARM_B: "#ef4444", ARM_C: "#2563eb", ARM_D: "#f59e0b"}
    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 3.8), squeeze=False)
    w = 0.8 / len(arms)
    for ax, m in zip(axes[0], models):
        for i, arm in enumerate(arms):
            sub = results[(results.model == m) & (results.arm == arm)].set_index("dataset").loc[datasets]
            ax.bar(np.arange(len(datasets)) + i * w, sub.auc_mean, w, yerr=sub.auc_std,
                   label=arm, color=colors.get(arm, "#7c3aed"), capsize=2)
        ax.set_xticks(np.arange(len(datasets)) + w * (len(arms) - 1) / 2)
        ax.set_xticklabels(datasets, rotation=20, fontsize=8)
        ax.set_title(m, fontsize=10)
        ax.set_ylabel("ROC-AUC (mean ± std over folds)")
        if ylim:
            ax.set_ylim(*ylim)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0][0].legend(fontsize=7, frameon=False, loc="lower left")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


def relative_contrast(X, n_pairs=3000, seed=0):
    """(d_max - d_min) / d_min of distances from random anchors to random points.

    Small values mean distances have 'concentrated' (the curse of dimensionality):
    every point looks about equally far away, which hurts distance-based methods.
    """
    X = np.asarray(X, dtype=float)
    rng = np.random.default_rng(seed)
    contrasts = []
    for _ in range(50):
        anchor = X[rng.integers(len(X))]
        idx = rng.choice(len(X), size=min(n_pairs // 50 + 20, len(X)), replace=False)
        dist = np.linalg.norm(X[idx] - anchor, axis=1)
        dist = dist[dist > 1e-12]
        contrasts.append((dist.max() - dist.min()) / dist.min())
    return float(np.mean(contrasts))


def cluster_arms(X, y, n_clusters, n_remove, alpha=1.0, beta=1.0, selection="pca_low",
                 n_repeats=10, frac=0.8, random_state=0):
    """K-Means on each arm's representation; ARI / NMI against the true labels.

    Clustering is unsupervised, so the representation is fit on the (label-free)
    subsample it is evaluated on. Repeats use different random 80% subsamples.
    """
    rng = np.random.default_rng(random_state)
    d = X.shape[1]
    rows = []
    for r in range(n_repeats):
        idx = rng.choice(len(X), size=int(frac * len(X)), replace=False)
        Xs, ys = X.iloc[idx], y.iloc[idx]
        for name, t in make_transformers(n_remove, d, alpha, beta, selection).items():
            Z = clone(t).fit_transform(Xs)
            km = KMeans(n_clusters=n_clusters, n_init=10, random_state=r).fit(Z)
            rows.append(dict(arm=name, repeat=r,
                             ari=adjusted_rand_score(ys, km.labels_),
                             nmi=normalized_mutual_info_score(ys, km.labels_),
                             contrast=relative_contrast(Z, seed=r)))
    df = pd.DataFrame(rows)
    return df.groupby("arm").agg(ari_mean=("ari", "mean"), ari_std=("ari", "std"),
                                 nmi_mean=("nmi", "mean"), contrast_mean=("contrast", "mean")).reset_index()


def arm_table(results):
    """Readable table: rows = (dataset, model), columns = arms, cells = 'AUC ± std'."""
    r = results.copy()
    r["cell"] = r.auc_mean.map("{:.4f}".format) + " ± " + r.auc_std.map("{:.4f}".format)
    return r.pivot_table(index=["dataset", "model"], columns="arm", values="cell",
                         aggfunc="first", sort=False)
