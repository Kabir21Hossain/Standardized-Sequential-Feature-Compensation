# Research log: how the idea evolved (including what was wrong)

This is the honest history, from the first experiment to this repository. Numbers quoted here
were produced by code in this repo or by the earlier experiments described in
`SSFC_Final_Research_Report.pdf`; where an earlier claim was later retracted, it says so.

## The idea in one paragraph

Before dropping a feature, fold the part of it that is linearly shared with the other features
into them (`Z_j <- Z_j + alpha * p * Cov(Z_k, Z_j)`, `p = sigmoid(beta * Z_k)`), so that the kept
features stay in their original, interpretable units. Repeat for each feature removed (a chain).

## Timeline

| Stage | What was done | What was learned |
|---|---|---|
| 0. First experiment (`notebooks/v0_original_synthetic_experiment.ipynb`) | Synthetic 40-feature data, remove "feature 31", compare drop / compensation / PCA with Logistic Regression and Random Forest | Random Forest: full 0.9836, drop 0.9823, fixed compensation 0.9835. Logistic Regression: no difference. Hard-step gate was worse than a smooth sigmoid gate. |
| 1. Make alpha adaptive | Treated strength `alpha` and gate sharpness as tunable | On that synthetic case `alpha = 0.5` closed the gap (0.9836). Motivated the idea of *learning* alpha. |
| 2. Chaining (k = 2, 3 removed features) | Applied the one-feature formula repeatedly; also a joint covariance-matrix variant | Simple chaining did as well as or better than the matrix variant. Single-split differences were small (~0.001) and partly noise. |
| 3. First real dataset (Breast Cancer) | Same formula, unchanged | **It hurt.** Raw covariance is scale dependent and real features have wildly different scales. **Standardizing first fixed it.** Standardization is now a required step. |
| 4. Six real datasets, single split | Breast Cancer, Wine, Digits, Sonar, Ionosphere, Pima | Mixed. On Digits/Sonar/Ionosphere the penalty for dropping was under 0.003 AUC (noise-sized), so there was nothing to recover. An "average correlation predicts recovery" hypothesis was tested and **rejected** (r = 0.034). |
| 5. Repeated CV, five model families (Pima n=768, Mammography n=11,183) | Removed the 3 features with the largest measured AUC drop; fixed alpha = 1; 5-fold x 4 repeats | Chain beat drop in **9 of 10** model x dataset cases (the exception, CatBoost on Mammography at -0.0017, was inside its own fold noise). Largest effect on Pima (+0.027 to +0.080 AUC). |
| 6. Theorem 1 | Safety argument for choosing alpha by CV with alpha = 0 in the grid | See below. This is what version 4 implements. |
| 7. This repository | PCA-guided removal of the *least* impactful features, four-arm comparison, versions 1 to 4 | See "Findings from the notebooks". |

## Theorem 1 (safety)

With `alpha = 0` every correction term vanishes, so the output equals plain dropping.
If `alpha*` maximizes the cross-validated score over a grid containing 0, then
`CV(alpha*) >= CV(0) = CV(plain drop)`.

It holds by construction. It is a statement about the **cross-validated score used to choose alpha**,
not about future test data, and the outer (nested) evaluation can still land slightly below plain
dropping by chance (it did in 2 of 16 cases in version 4).

## Corrections made along the way (kept visible on purpose)

1. **"Linear equivalence" claim retracted.** An early result said that for linear models the
   compensation is equivalent to adding one scalar gate column per removed feature, "and cheaper".
   That comparison was not width-matched: the gate-only variant adds the columns back, so its width
   is `d`, the same as the full model, not `d - k`. It was invisible on 30-40 feature datasets and obvious
   on Pima (d = 8). The valid comparison is always compensated vs. plain drop at the same width, which
   is what every table in this repo uses.
2. **Counting error.** A summary once said "14 of 15" cases; five models on two datasets is 10 cases,
   and the correct figure is 9 of 10.
3. **Raw covariance without standardization** was shown to hurt on real data (stage 3).

## Findings from the notebooks (this repo; seeds fixed, 5-fold CV, ROC-AUC)

Arms: A naive (all features), B PCA-guided drop, C compensate-then-drop (same features as B), D PCA components.
"Better / within noise / worse" compares C with B using the paired fold difference and one standard error.

| Version | Setting | Compensation (C) vs plain drop (B) |
|---|---|---|
| 1 | Logistic Regression + Random Forest, 4 datasets, remove least-impactful (PCA) features, alpha = 1 | better 3, within noise 4, worse 1 (of 8) |
| 1 | Pima, remove glucose + BMI + pedigree (features that matter) | LogReg 0.672 -> 0.699, Random Forest 0.706 -> 0.782 (PCA reference 0.799 / 0.780) |
| 2 | HistGradientBoosting, ExtraTrees, MLP, XGBoost, CatBoost, 4 datasets, alpha = 1 | better 3, within noise 11, worse 6 (of 20) |
| 3 | kNN, distance-weighted kNN, RBF-SVM, 5 datasets, alpha = 1 | better 5, within noise 1, worse 9 (of 15). **Sonar collapses (AUC down 0.18-0.26).** |
| 4 | LogReg, Random Forest, kNN, RBF-SVM, 4 datasets, alpha chosen by inner CV | fixed alpha = 1: better 5 / noise 6 / worse 5. Auto-alpha: better 5 / noise 9 / worse 2 (of 16). Worst case -0.0280 -> -0.0024. |

### What these results do and do not say

* **Removing low-impact features leaves little to recover.** A, B, C and D are usually within noise of
  each other, so most C - B differences are small.
* **PCA impact is unsupervised.** It ignores the target, so "low impact" is not "low predictive value".
  Its scores are also nearly flat on some datasets (Pima: 0.115 to 0.132).
* **A fixed alpha = 1 is not reliable.** It helped in some cases and hurt in others, and it can be
  catastrophic for distance-based models when many correlated features are removed (see next point).
* **Chain compounding.** Each stage recomputes covariances on already-modified data. With many removals on
  strongly correlated features the kept columns inflate (Sonar, 15 of 60 features removed, `alpha = 1`:
  median column std about 40; with `alpha = 0.5` it is about 1.2). The cross-validated alpha search in version 4
  avoids this by preferring small alpha when large alpha does not help. A cleaner fix (re-standardizing
  after each stage, or normalizing the injected term) is future work and has not been tested.
* **PCA (arm D) is a strong reference** at the same width and sometimes beats compensation. The case for SSFC is
  interpretability (kept features stay in original units) plus the safety property, not raw accuracy.
* **Evidence is limited**: small public datasets, binary or multi-class classification only, linear
  compensation only, removed features must still be measured to compute the gate.

## Ideas not yet done

Re-standardizing after each stage; non-linear or learned compensation; supervised ranking of features to remove;
per-feature alpha; regression targets; more and larger datasets; multiple seeds with a formal paired test.
