# ⚡ Household Electricity Power Forecast

**AIC354 Machine Learning Fundamentals – Assignment 3, Task 2**
COMSATS University Islamabad, Lahore Campus · Fall 2026

🔗 **Live app:** https://7r5qvtyj2jtu9cnqsaqm8q.streamlit.app/
*(free hosting: if the app has been idle, press the "wake up" button and wait a moment)*

A web app that estimates a household's **global active power 60 minutes ahead (in kW)** from the current meter
readings. The model was chosen by 10-fold cross-validation on the full UCI household power consumption dataset.

## Project summary
| Item | Details |
|---|---|
| Dataset | [UCI Individual Household Electric Power Consumption](https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set) – 2,075,259 per-minute rows (Dec 2006 – Nov 2010); 2,048,053 rows eligible after requiring real inputs and a real target |
| Task | Regression: predict `Global_active_power` 60 minutes ahead (kW, a *power*, not energy) |
| Algorithms compared | Ridge, Bayesian Ridge, Huber, SGD Regressor, Extra Trees |
| Validation | Required shuffled 10-fold CV on all eligible rows, plus a chronological 80/20 hold-out test |
| Best model | **Extra Trees** |

## Results
| Algorithm | 10-fold mean RMSE (kW) | Chronological test RMSE (kW) |
|---|---|---|
| **Extra Trees** | **0.730** | **0.718** |
| Ridge | 0.868 | 0.769 |
| Bayesian Ridge | 0.868 | 0.769 |
| SGD Regressor | 0.869 | 0.768 |
| Huber | 0.892 | 0.791 |

Baselines on the chronological test: persistence (current power) 0.951 kW, training mean 0.901 kW.
Extra Trees: CV R² = 0.52, chronological R² = 0.36. Predicting one hour ahead is inherently uncertain, so the output is an *estimate*.

## App inputs (measured at the current minute)
Date and time · current global active power (kW) · average power over the last 60 min (kW) · global reactive power (kVAR) ·
voltage (V) · global intensity (A) · sub-metering 1, 2, 3 (Wh).

## Repository files
| File | Purpose |
|---|---|
| `app.py` | Streamlit web app |
| `electricity_model.joblib` | Saved scikit-learn pipeline (preprocessing + Extra Trees) |
| `model_metadata.json` | Feature order, units, horizon, metrics, observed input ranges |
| `requirements.txt` | Python dependencies (scikit-learn 1.6.1, the version used for training) |

## Run locally
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
Then open http://localhost:8501. Use Python 3.11 or 3.12.

## Deploy on Streamlit Community Cloud
1. Upload the four files above to a public GitHub repository (`app.py` in the root).
2. At https://share.streamlit.io choose **Create app**, select the repo, branch `main` and `app.py`.
3. Under **Advanced settings** choose Python 3.12, then **Deploy**.

## Limitations
* One household in France (2006–2010); results may not transfer to other homes.
* Hyper-parameters were fixed, not tuned. Weather and holiday information are not used.
* The prediction is a statistical estimate, not a guaranteed measurement.
