import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from ssfc import SSFC, AutoAlphaSSFC, pca_feature_impact


def make_data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    f1, f2 = rng.normal(size=(2, n))
    X = np.column_stack([f1, f1 + .3 * rng.normal(size=n), f1 + .3 * rng.normal(size=n),
                         f2, f2 + .3 * rng.normal(size=n), f2 + .3 * rng.normal(size=n),
                         rng.normal(size=n)])          # last column: pure noise
    y = (f1 + f2 + .5 * rng.normal(size=n) > 0).astype(int)
    return pd.DataFrame(X, columns=list("abcdefg")), pd.Series(y)


def test_output_width():
    X, y = make_data()
    out = SSFC(n_remove=3).fit_transform(X)
    assert out.shape == (len(X), X.shape[1] - 3)


def test_alpha_zero_equals_plain_drop():
    X, y = make_data()
    comp = SSFC(n_remove=2, compensate=True, alpha=0.0).fit(X)
    drop = SSFC(n_remove=2, compensate=False).fit(X)
    assert comp.removed_idx_ == drop.removed_idx_
    np.testing.assert_allclose(comp.transform(X), drop.transform(X))


def test_compensation_changes_values():
    X, y = make_data()
    comp = SSFC(n_remove=2, compensate=True, alpha=1.0).fit(X)
    drop = SSFC(n_remove=2, compensate=False).fit(X)
    assert not np.allclose(comp.transform(X), drop.transform(X))


def test_transform_is_deterministic_and_handles_new_data():
    X, y = make_data()
    t = SSFC(n_remove=2).fit(X)
    Xn, _ = make_data(n=50, seed=9)
    np.testing.assert_allclose(t.transform(Xn), t.transform(Xn))
    assert t.transform(Xn).shape == (50, X.shape[1] - 2)


def test_pca_low_removes_noise_column():
    X, y = make_data()
    t = SSFC(n_remove=1, selection="pca_low", var_threshold=0.80).fit(X)
    assert t.removed_features_ == ["g"]


def test_impact_sums_to_one():
    X, _ = make_data()
    Z = (X - X.mean()) / X.std()
    assert abs(pca_feature_impact(Z).sum() - 1.0) < 1e-9


def test_sklearn_clone_and_names():
    X, y = make_data()
    t = clone(SSFC(n_remove=2)).fit(X)
    assert len(t.get_feature_names_out()) == X.shape[1] - 2


def test_auto_alpha_never_worse_than_zero_on_inner_cv():
    X, y = make_data()
    m = AutoAlphaSSFC(LogisticRegression(max_iter=500), n_remove=2).fit(X, y)
    scores = {a: s for a, b, s in m.cv_grid_}
    assert m.best_cv_score_ >= scores[0.0] - 1e-12
    assert m.predict_proba(X).shape == (len(X), 2)


def test_auto_alpha_works_with_roc_auc_scorer_in_cv():
    from sklearn.model_selection import cross_val_score
    X, y = make_data()
    est = AutoAlphaSSFC(LogisticRegression(max_iter=500), n_remove=2)
    scores = cross_val_score(est, X, y, cv=3, scoring="roc_auc", error_score="raise")
    assert np.all(np.isfinite(scores))


def test_auto_alpha_exposes_only_methods_the_model_has():
    from sklearn.ensemble import RandomForestClassifier
    rf = AutoAlphaSSFC(RandomForestClassifier(n_estimators=5), n_remove=1)
    assert hasattr(rf, "predict_proba") and not hasattr(rf, "decision_function")
    lr = AutoAlphaSSFC(LogisticRegression(), n_remove=1)
    assert hasattr(lr, "predict_proba") and hasattr(lr, "decision_function")
