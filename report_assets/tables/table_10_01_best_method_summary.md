| Method | Selection view | CV RMSE (SD) | Test RMSE (95% bootstrap CI) | Interpretation |
|---|---|---:|---:|---|
| Hard removal — HistGBDT | Lowest mean development-CV RMSE | 0.13947 (0.01501) | 0.12070 ([0.10208, 0.14203]) | CV leader; paired Holm comparison not significant. |
| No handling — HistGBDT | No-handling reference | 0.14008 (0.01455) | 0.13043 ([0.10662, 0.15307]) | Strong standard tree baseline. |
| Weighted — Gradient Boosting | Preselected weighted strategy | 0.14268 (0.01453) | 0.12376 ([0.10754, 0.13958]) | 5.12% lower test RMSE point estimate than no-handling HistGBDT; CIs overlap. |
| Hard removal — Ridge | Lowest one-time test RMSE (descriptive) | 0.14502 (0.04324) | 0.11350 ([0.09838, 0.12842]) | Post-hoc test point-estimate leader; not selected using test results. |
| Weighted — Ridge (tuned exponential) | Weighted linear-model comparison | 0.14416 (0.03465) | 0.11859 ([0.10190, 0.13589]) | Small CV point-estimate gain; relatively high fold variability. |
