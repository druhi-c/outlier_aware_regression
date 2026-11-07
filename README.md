# Outlier-Aware Regression for House Price Prediction

A reliability-weighting framework for tabular regression, evaluated on the Ames Housing
dataset. Instead of deleting suspected outliers or trusting them equally, every training
row keeps a soft weight derived from anomaly and influence scores computed **inside the
training fold**, so no validation or test information leaks into the weighting.

The repository contains the full experiment notebook, the saved figures and tables it
produced, and a Streamlit app that explores those results and scores new houses.

---

## Research question

Standard regression workflows choose one of two extremes: ignore outliers entirely, or
delete them with a hard threshold (z-score, IQR fences, Cook's distance). Hard removal
discards genuine extreme values — a large luxury home is unusual but not erroneous — and
a threshold learned on the whole dataset leaks into cross-validation.

This project asks whether a middle ground helps: **can soft reliability weights, derived
from unsupervised anomaly scores and influence diagnostics computed only on the training
fold, improve accuracy and robustness over unweighted baselines, hard removal,
winsorizing, and robust loss functions?**

| Hypothesis | Claim |
|---|---|
| **H₁ Accuracy** | Weighted models beat their unweighted counterparts in mean CV RMSE (log space) by 1–5 %. |
| **H₂ Robustness** | Under synthetic contamination (5–15 % injected noise), weighted models degrade less than baselines. |
| **H₃ Stability** | Fold-to-fold RMSE standard deviation is lower for weighted models. |

---

## Data and evaluation protocol

- **Dataset:** Ames Housing (`SalePrice`), target modelled as `log1p(SalePrice)`.
- **Split:** 80 / 20, stratified by price quintile with `seed=42`. After cleaning, the
  development partition holds 1,168 rows and the locked hold-out holds 292 rows. The
  stratified split matched target distributions (KS statistic 0.029, p = 0.99).
- **Hold-out discipline:** the test partition is frozen to `report_assets/test_set.parquet`
  and is never used for EDA, imputation parameters, outlier scoring, or tuning.
- **Cross-validation:** repeated 5×3 CV (`RepeatedKFold`, 15 fold scores) on the development
  partition throughout, including the paired comparisons and ablations.
- **Metrics:** RMSE, MAE, MedAE, P90 absolute error, and R² in log space, plus
  RMSE/MAE in dollars. Robustness is reported as ΔRMSE = RMSE_contaminated − RMSE_clean.
- **Paired testing:** fold-level Wilcoxon signed-rank tests against the no-handling
  reference, with Holm correction and bootstrap intervals.

---

## Method: leakage-safe reliability weighting

For every fold, the pipeline is:

1. Fit the shared `ColumnTransformer` preprocessing on the training fold only.
2. Compute anomaly / influence signals on that fold: Isolation Forest, Local Outlier
   Factor, robust Mahalanobis distance, out-of-fold residuals, Cook's distance, leverage,
   and k-NN target distance.
3. Convert each signal to an empirical percentile rank on [0, 1].
4. Aggregate signals (mean, max, or rank average) and map the combined score to a weight
   in `[w_min, 1]` using a linear, exponential, or sigmoid decay, or a Huber/Tukey
   residual rule.
5. Fit the base model with `sample_weight=w`. Validation and test rows are always scored
   with uniform weight.

The tuned configurations were selected on development inner-validation records only and
then frozen before the single hold-out evaluation, e.g. Weighted Ridge used
`{scorer_type: rank_average, k: 3.0, w_min: 0.05}` and Weighted Gradient Boosting used
the same scorer with `learning_rate: 0.05` and 200 estimators.

Compared families: unweighted Ridge / Random Forest / HistGradientBoosting, hard removal,
winsorizing, robust losses (HuberRegressor, Huber and median-quantile boosting), and the
reliability-weighted variants of Ridge, Random Forest, and Gradient Boosting, plus an IRLS
reweighting variant.

---

## Repository contents

```
outlier_aware_regression.ipynb   Full study: protocol, EDA, baselines, scoring, weighting,
                                 refinements, hold-out, contamination lab, ablations
app.py                           Streamlit explorer for the saved results and predictions
requirements.txt                 App dependencies
README_STREAMLIT.md              Original app-specific notes
report_assets/
  figures/                       31 report figures (PNG) from stages 2–9
  tables/                        32 result tables in CSV and Markdown
  test_set.parquet               Locked 292-row hold-out partition
```

Inside the app, the notebook's stages map to these views: Overview, Price predictions,
Model comparison, Contamination lab, Influence & weights, Ablations & error analysis,
Figures & tables, and Protocol & caveats.

---

## Key results

Ranked development-CV results and the one-time hold-out evaluation (log-space RMSE):

| Method | Selection view | CV RMSE (SD) | Test RMSE (95 % bootstrap CI) | Interpretation |
|---|---|---:|---:|---|
| Hard removal — HistGBDT | Lowest mean development-CV RMSE | 0.13947 (0.01501) | 0.12070 ([0.10208, 0.14203]) | CV leader; paired Holm comparison not significant. |
| No handling — HistGBDT | No-handling reference | 0.14008 (0.01455) | 0.13043 ([0.10662, 0.15307]) | Strong standard tree baseline. |
| Weighted — Gradient Boosting | Preselected weighted strategy | 0.14268 (0.01453) | 0.12376 ([0.10754, 0.13958]) | 5.12 % lower test RMSE point estimate than no-handling HistGBDT; CIs overlap. |
| Hard removal — Ridge | Lowest one-time test RMSE (descriptive) | 0.14502 (0.04324) | 0.11350 ([0.09838, 0.12842]) | Post-hoc test point-estimate leader; not selected using test results. |
| Weighted — Ridge (tuned exponential) | Weighted linear-model comparison | 0.14416 (0.03465) | 0.11859 ([0.10190, 0.13589]) | Small CV point-estimate gain; relatively high fold variability. |

**The honest summary:** weighting produces consistent small improvements in point
estimates — notably a 5.12 % lower hold-out RMSE for the preselected weighted Gradient
Boosting than the unweighted HistGBDT baseline — but no comparison survived Holm
correction, and all bootstrap intervals overlap. Hard removal was narrowly the best CV
performer, which is itself evidence that the soft-weighting middle ground is not a clear
win on this dataset.

### Hypothesis outcomes

| Hypothesis | Status | Evidence |
|---|---|---|
| **H₁ Accuracy** | Qualified support | Preselected weighted Gradient Boosting test RMSE 0.12376 vs no-handling HistGBDT 0.13043 (5.12 % point reduction); Holm-adjusted CV tests were not significant and test intervals overlap. |
| **H₂ Robustness** | Partially supported | Detection was strong for leverage/mixed injections and weaker for target-only noise; predictive degradation varied by strategy and corruption type. |
| **H₃ Fold stability** | Partially supported overall; mixed in Stage 9 | Stage 7 showed a 1.2 % weighted-vs-baseline SD reduction, below its 20 % criterion; Stage 9 linear weighted Ridge SD was 0.04169 vs baseline 0.02908. |

Against the project's own success criteria (≥ 1 % RMSE gain, ≤ 50 % of baseline
degradation, or ≥ 20 % lower fold SD), none were fully met.

---

## Additional findings

**Detection works better than prediction improves.** The combined influence scorer
separated injected rows from clean ones with mean AUROC 0.939–0.947 for leverage
contamination and 0.973–0.981 for mixed contamination, but only 0.73–0.74 for
target-only noise. Precision@k rose from 0.51 to 0.86 for mixed contamination as its share
went from 2 % to 20 %.

**Weighting delayed degradation rather than preventing it.** Using a threshold of ≥ 10 %
above each strategy's clean RMSE, no-handling Ridge crossed it at 5 % mixed contamination
and 10 % target noise, and winsorizing Ridge crossed at 5 % mixed. The saved threshold
table flags no crossing within the tested 0–20 % range for weighted Ridge and weighted
Gradient Boosting under either corruption type. Hard removal held out until 20 % mixed
contamination, but its RMSE there (0.17822) was well above the weighted variants
(0.15623 for weighted Ridge, 0.15505 for weighted Gradient Boosting), so delayed crossing
did not mean lower error once corruption was severe.

**Signal and shape ablations.** Linear decay with an averaged score gave the lowest
ablation RMSE at 0.14451 versus 0.14519 for unweighted Ridge, and also lowered MAE and
P90 — but with *higher* fold variability (SD 0.04169 vs 0.02908). Stronger exponential
decay did not help: the sensitivity grid's best was 0.14426 at k = 1, while k = 5 with
`w_min = 0.01` gave 0.15126.

**Errors by price segment.** Weighting reduced MAE and P90 in Q2 (MAE 0.0922 → 0.0829,
P90 0.1745 → 0.1612) and Q3 (0.0794 → 0.0703, 0.1595 → 0.1437), with mixed effects in Q1
and Q4.

**Anomaly ≠ invalid.** Low-reliability flags included both unusual large, modestly priced
homes and expensive high-quality homes, so a high score should not be read as a data error.

**One cautionary result.** HuberRegressor showed a severe CV-to-test gap (CV RMSE 0.15528
vs test RMSE 1.97891), a reminder that a single split and an assumption that robust losses
generalize can both fail.

Full tables are in `report_assets/tables/`; the complete narrative is in the notebook's
*Key Findings* section.

---

## Running the Streamlit app

Requires Python 3 (the saved environment was CPython 3.13).

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Streamlit prints a local URL (typically `http://localhost:8501`). Keep `app.py` beside the
notebook and the `report_assets/` directory — the app reads the CSV tables and PNG figures
from there.

**Price predictions page.** The notebook archive has saved experiment tables and figures
but no deployable fitted estimator and no raw development dataset, so this page trains a
fresh model at request time. It accepts an uploaded Ames-style labeled CSV or fetches
`house_prices` from OpenML, then scores a single house via a form or a batch upload.

Important: those are **new fits**, not the frozen Stage 7 evaluation models, and the
notebook's locked test set is not used. The page does not rerun the cross-validation or
contamination sweeps. The weighted option estimates training-only influence scores and uses
them as sample weights.

The other pages are read-only views over the saved results, so they work with no dataset
and no network access.

---

## Reproducing the study

Open `outlier_aware_regression.ipynb` in Jupyter and run all cells top to bottom. The
notebook is self-contained: it downloads Ames Housing, performs the stratified split,
writes `report_assets/test_set.parquet`, and regenerates every figure and table in
`report_assets/`. Expect a long runtime — the notebook runs 5×3 repeated CV across roughly
17 strategies, plus contamination sweeps at 0/2/5/10/20 % over three corruption types and
repeated injection seeds.

Dependencies beyond the app's `requirements.txt` are the standard scientific stack plus
Jupyter (`jupyter`, `matplotlib`, `seaborn`); the notebook also uses `xgboost` or
`lightgbm` only if available.

Note that the notebook was executed under its original environment, so re-running it may
produce slightly different numbers from the saved assets depending on library versions.

---

## Caveats

- One dataset, one locked split: results are descriptive of Ames Housing, not evidence of a
  general accuracy win. The small test partition (292 rows) gives wide bootstrap intervals.
- Tuning used development inner-validation records; the hold-out was touched once, and the
  best test RMSE (hard removal Ridge) is reported descriptively rather than as a selection.
- Synthetic contamination is a controlled proxy for real data errors, and only leverage,
  mixed, and target-noise corruption were injected.
- Weighted fits estimate scores per fit, so predictions from the app are new models rather
  than the frozen evaluation artifacts.

---

## Notes

`README_STREAMLIT.md` contains the original app-focused notes. All figures and tables are
regenerated assets from the notebook run; treat the notebook as the source of truth for
methodology.
