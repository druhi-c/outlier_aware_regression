| contamination_type   | strategy                             |   clean_RMSE |   sharp_degradation_at_pct | threshold                  |
|:---------------------|:-------------------------------------|-------------:|---------------------------:|:---------------------------|
| target noise         | No handling — Ridge                  |     0.145187 |                         10 | ≥10% above clean mean RMSE |
| target noise         | Hard removal — Ridge                 |     0.145021 |                         20 | ≥10% above clean mean RMSE |
| target noise         | Winsorizing — Ridge                  |     0.144142 |                         10 | ≥10% above clean mean RMSE |
| target noise         | Robust loss — HuberRegressor         |     0.155323 |                         10 | ≥10% above clean mean RMSE |
| target noise         | Weighted — Ridge (linear)            |     0.144509 |                         20 | ≥10% above clean mean RMSE |
| target noise         | Weighted — Ridge (tuned exponential) |     0.146658 |                        nan | ≥10% above clean mean RMSE |
| target noise         | Weighted — Ridge (sigmoid)           |     0.146833 |                        nan | ≥10% above clean mean RMSE |
| target noise         | Weighted — Random Forest             |     0.153963 |                        nan | ≥10% above clean mean RMSE |
| target noise         | Weighted — Gradient Boosting         |     0.142704 |                        nan | ≥10% above clean mean RMSE |
| target noise         | IRLS — Ridge                         |     0.14479  |                        nan | ≥10% above clean mean RMSE |
| leverage points      | No handling — Ridge                  |     0.145187 |                        nan | ≥10% above clean mean RMSE |
| leverage points      | Hard removal — Ridge                 |     0.145021 |                        nan | ≥10% above clean mean RMSE |
| leverage points      | Winsorizing — Ridge                  |     0.144142 |                        nan | ≥10% above clean mean RMSE |
| leverage points      | Robust loss — HuberRegressor         |     0.155323 |                        nan | ≥10% above clean mean RMSE |
| leverage points      | Weighted — Ridge (linear)            |     0.144509 |                        nan | ≥10% above clean mean RMSE |
| leverage points      | Weighted — Ridge (tuned exponential) |     0.146658 |                        nan | ≥10% above clean mean RMSE |
| leverage points      | Weighted — Ridge (sigmoid)           |     0.146833 |                        nan | ≥10% above clean mean RMSE |
| leverage points      | Weighted — Random Forest             |     0.153963 |                        nan | ≥10% above clean mean RMSE |
| leverage points      | Weighted — Gradient Boosting         |     0.142704 |                        nan | ≥10% above clean mean RMSE |
| leverage points      | IRLS — Ridge                         |     0.14479  |                        nan | ≥10% above clean mean RMSE |
| mixed                | No handling — Ridge                  |     0.145187 |                          5 | ≥10% above clean mean RMSE |
| mixed                | Hard removal — Ridge                 |     0.145021 |                         20 | ≥10% above clean mean RMSE |
| mixed                | Winsorizing — Ridge                  |     0.144142 |                          5 | ≥10% above clean mean RMSE |
| mixed                | Robust loss — HuberRegressor         |     0.155323 |                         10 | ≥10% above clean mean RMSE |
| mixed                | Weighted — Ridge (linear)            |     0.144509 |                        nan | ≥10% above clean mean RMSE |
| mixed                | Weighted — Ridge (tuned exponential) |     0.146658 |                        nan | ≥10% above clean mean RMSE |
| mixed                | Weighted — Ridge (sigmoid)           |     0.146833 |                        nan | ≥10% above clean mean RMSE |
| mixed                | Weighted — Random Forest             |     0.153963 |                        nan | ≥10% above clean mean RMSE |
| mixed                | Weighted — Gradient Boosting         |     0.142704 |                        nan | ≥10% above clean mean RMSE |
| mixed                | IRLS — Ridge                         |     0.14479  |                        nan | ≥10% above clean mean RMSE |