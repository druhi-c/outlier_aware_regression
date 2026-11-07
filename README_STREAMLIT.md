# Outlier-Aware Regression — Streamlit Results Explorer

This app presents the saved results from the Ames Housing outlier-aware regression notebook. It includes CV and hold-out comparisons, synthetic contamination results, influence-score and weighting diagnostics, Stage 9 ablations, error analysis, and a report-asset gallery.

The notebook archive contains saved experiment tables and figures but no deployable fitted estimator or raw development dataset. The Price predictions page therefore trains a fresh model from a user-uploaded Ames-style CSV or fetches the same Ames dataset from OpenML, then scores single-house or batch inputs. The app does not rerun the notebook’s cross-validation or contamination sweeps. The weighted model estimates training-only influence scores and uses them as sample weights; its predictions are new fits and are not the frozen Stage 7 evaluation models.

## Run locally

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Keep `app.py` beside the notebook and the `report_assets/` directory. The app reads the CSV tables and PNG figures in `report_assets/`.
