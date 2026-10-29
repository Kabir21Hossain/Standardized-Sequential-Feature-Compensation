"""Core implementation of Standardized Sequential Feature Compensation (SSFC).

Idea
----
Before dropping a feature ``k``, fold the part of its information that is
*linearly shared* with the other features into them, then drop it:

    Z_j  <-  Z_j + alpha * p * Cov(Z_k, Z_j)          for every retained j
    p    =   sigmoid(beta * standardized Z_k)           (the "gate")

repeated one removed feature at a time (a "chain"). All statistics are
learned in ``fit`` (training data only), so the transformer is leak-free
inside a scikit-learn ``Pipeline`` / cross-validation.

Safety property (Theorem 1 in docs/research_log.md): with ``alpha = 0`` every
correction vanishes and the output equals plain dropping, so choosing alpha by
cross-validation over a grid that contains 0 can never score worse than plain
dropping *on the cross-validated score*.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.utils.metaestimators import available_if


def sigmoid(z):
    """Numerically safe logistic function."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))


def pca_feature_impact(Z, var_threshold=0.90):
    """Unsupervised 'impact' of each feature, from PCA on standardized data.

    impact_j = sum over the leading components (those that together explain
    ``var_threshold`` of the variance) of  explained_variance_ratio_c * loading_cj**2.
    Features that take part mostly in the dominant variance directions get a
    high impact; features that only live in the weak tail components get a low
    one. The result is normalised to sum to 1.

    NOTE: this never looks at the target ``y``. "Low impact" therefore means
    "low share of the dominant variance structure", not "low predictive value".
    """
    Z = np.asarray(Z, dtype=float)
    pca = PCA().fit(Z)
    evr = pca.explained_variance_ratio_
    n_comp = int(np.searchsorted(np.cumsum(evr), var_threshold) + 1)
    n_comp = max(1, min(n_comp, len(evr)))
    loadings = pca.components_[:n_comp]
    impact = (evr[:n_comp, None] * loadings ** 2).sum(axis=0)
    total = impact.sum()
    return impact / total if total > 0 else np.full(Z.shape[1], 1.0 / Z.shape[1])


class SSFC(TransformerMixin, BaseEstimator):
    """Select features to remove (PCA impact), optionally compensate, then drop.

    Parameters
    ----------
    n_remove : int
        How many features to remove.
    selection : {"pca_low", "pca_high"}
        "pca_low"  -> remove the least impactful features (the usual use case).
        "pca_high" -> remove the most impactful features (a stress test that
        creates a large performance gap for compensation to recover).
        Ignored when ``remove_idx`` is given.
    remove_idx : sequence of int or None
        Explicit column indices to remove (processed in the given order).
    compensate : bool
        False -> plain drop (the "no compensation" arm). True -> our idea.
    alpha : float
        Compensation strength. 0 reproduces plain dropping exactly.
    beta : float
        Gate sharpness: p = sigmoid(beta * z).
    var_threshold : float
        Variance share used by ``pca_feature_impact``.

    Attributes (after fit)
    ----------------------
    removed_idx_, keep_idx_ : lists of column indices
    removed_features_, kept_features_ : names (when X was a DataFrame)
    impact_ : PCA impact of every input column
    """

    def __init__(self, n_remove=1, selection="pca_low", remove_idx=None,
                 compensate=True, alpha=1.0, beta=1.0, var_threshold=0.90):
        self.n_remove = n_remove
        self.selection = selection
        self.remove_idx = remove_idx
        self.compensate = compensate
        self.alpha = alpha
        self.beta = beta
        self.var_threshold = var_threshold

    # ------------------------------------------------------------------ fit
    def fit(self, X, y=None):
        cols = list(X.columns) if hasattr(X, "columns") else None
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        self.n_features_in_ = d
        self.columns_ = cols

        self.mu_ = X.mean(axis=0)
        sd = X.std(axis=0, ddof=1)
        sd[sd == 0] = 1.0
        self.sd_ = sd
        Z = (X - self.mu_) / self.sd_

        self.impact_ = pca_feature_impact(Z, self.var_threshold)
        if self.remove_idx is not None:
            removed = [int(i) for i in self.remove_idx]
        else:
            k = int(min(self.n_remove, d - 1))
            if self.selection == "pca_low":
                removed = list(np.argsort(self.impact_)[:k])
            elif self.selection == "pca_high":
                removed = list(np.argsort(-self.impact_)[:k])
            else:
                raise ValueError("selection must be 'pca_low' or 'pca_high'")
        self.removed_idx_ = [int(i) for i in removed]

        cur = Z.copy()
        active = list(range(d))
        self.stages_ = []
        for k in self.removed_idx_:
            others = [j for j in active if j != k]
            if self.compensate:
                zk = cur[:, k]
                m_, s_ = zk.mean(), zk.std(ddof=1)
                s_ = s_ if s_ > 0 else 1.0
                zc = zk - zk.mean()
                oc = cur[:, others] - cur[:, others].mean(axis=0)
                cov = (oc * zc[:, None]).sum(axis=0) / (n - 1)
                self.stages_.append(dict(k=k, others=others, m=m_, s=s_, cov=cov))
                p = sigmoid(self.beta * (zk - m_) / s_)
                cur[:, others] += self.alpha * p[:, None] * cov[None, :]
            active.remove(k)
        self.keep_idx_ = active
        if cols is not None:
            self.kept_features_ = [cols[j] for j in self.keep_idx_]
            self.removed_features_ = [cols[j] for j in self.removed_idx_]
        return self

    # ------------------------------------------------------------ transform
    def transform(self, X):
        X = np.asarray(X, dtype=float)
        cur = (X - self.mu_) / self.sd_
        for st in self.stages_:
            zk = cur[:, st["k"]]
            p = sigmoid(self.beta * (zk - st["m"]) / st["s"])
            cur[:, st["others"]] += self.alpha * p[:, None] * st["cov"][None, :]
        return cur[:, self.keep_idx_]

    def get_feature_names_out(self, input_features=None):
        if self.columns_ is None:
            return np.array([f"x{j}" for j in self.keep_idx_])
        return np.array(self.kept_features_)


class AutoAlphaSSFC(ClassifierMixin, BaseEstimator):
    """Version 4: SSFC whose strength is chosen by inner cross-validation.

    The grid ``alphas`` must contain 0 (it is added if missing). Because
    alpha = 0 is plain dropping, the selected setting scores at least as well as
    plain dropping on the inner cross-validated score (Theorem 1). Ties go to the
    smaller alpha, and a non-zero alpha must beat alpha = 0 by more than
    ``min_gain`` to be chosen, which keeps the method conservative.

    Everything (PCA ranking, covariances, alpha) is learned inside ``fit`` from
    training data only, so wrapping this estimator in an outer cross-validation
    gives an honest (nested) estimate.
    """

    def __init__(self, estimator, n_remove=1, selection="pca_low",
                 alphas=(0.0, 0.5, 1.0, 2.0), betas=(1.0,), cv=3,
                 scoring="auto", min_gain=0.0, random_state=0, var_threshold=0.90):
        self.estimator = estimator
        self.n_remove = n_remove
        self.selection = selection
        self.alphas = alphas
        self.betas = betas
        self.cv = cv
        self.scoring = scoring
        self.min_gain = min_gain
        self.random_state = random_state
        self.var_threshold = var_threshold

    def _pipe(self, alpha, beta):
        return Pipeline([
            ("ssfc", SSFC(n_remove=self.n_remove, selection=self.selection,
                          compensate=True, alpha=alpha, beta=beta,
                          var_threshold=self.var_threshold)),
            ("model", clone(self.estimator)),
        ])

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        scoring = self.scoring
        if scoring == "auto":
            scoring = "roc_auc" if len(self.classes_) == 2 else "roc_auc_ovr"
        cv = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        alphas = sorted(set(float(a) for a in self.alphas) | {0.0})
        base = float(np.mean(cross_val_score(self._pipe(0.0, 1.0), X, y, cv=cv, scoring=scoring)))
        best = (0.0, 1.0, base)
        grid = []
        for beta in self.betas:
            for a in alphas:
                if a == 0.0:
                    continue
                s = float(np.mean(cross_val_score(self._pipe(a, beta), X, y, cv=cv, scoring=scoring)))
                grid.append((a, beta, s))
                if s > best[2] + self.min_gain and s > base + self.min_gain:
                    best = (a, beta, s)
        self.cv_grid_ = [(0.0, 1.0, base)] + grid
        self.best_alpha_, self.best_beta_, self.best_cv_score_ = best
        self.best_pipeline_ = self._pipe(self.best_alpha_, self.best_beta_).fit(X, y)
        return self

    def predict(self, X):
        return self.best_pipeline_.predict(X)

    @available_if(lambda self: hasattr(self.estimator, "predict_proba"))
    def predict_proba(self, X):
        return self.best_pipeline_.predict_proba(X)

    @available_if(lambda self: hasattr(self.estimator, "decision_function"))
    def decision_function(self, X):
        return self.best_pipeline_.decision_function(X)
