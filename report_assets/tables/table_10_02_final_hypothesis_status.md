| Hypothesis | Status | Evidence |
|---|---|---|
| H1 — Accuracy | Qualified support | Preselected weighted Gradient Boosting test RMSE 0.12376 vs no-handling HistGBDT 0.13043 (5.12% point reduction); Holm-adjusted CV tests were not significant and test intervals overlap. |
| H2 — Contamination robustness | Partially supported | Detection was strong for leverage/mixed injections and weaker for target-only noise; predictive degradation varied by strategy and corruption type. |
| H3 — Fold stability | Partially supported overall; mixed in Stage 9 | Stage 7 showed a 1.2% weighted-vs-baseline SD reduction, below its 20% criterion; Stage 9 linear weighted Ridge SD was 0.04169 vs baseline 0.02908. |
