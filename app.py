"""Streamlit app: 1-hour-ahead household electricity POWER estimate (kW).
Uses the pipeline saved by the notebook (electricity_model.joblib + model_metadata.json)."""
import json
from datetime import datetime, time as dtime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).parent


@st.cache_resource
def load_artifacts():
    model = joblib.load(BASE / "electricity_model.joblib")
    meta = json.loads((BASE / "model_metadata.json").read_text())
    return model, meta


def add_calendar_features(frame, timestamps):
    # identical to the notebook (Step 7)
    hour = timestamps.hour + timestamps.minute / 60.0
    frame["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    frame["month_sin"] = np.sin(2 * np.pi * (timestamps.month - 1) / 12)
    frame["month_cos"] = np.cos(2 * np.pi * (timestamps.month - 1) / 12)
    frame["is_weekend"] = (timestamps.dayofweek >= 5).astype(float)
    return frame


def build_feature_row(timestamp, current_power, reactive, voltage, intensity, sm1, sm2, sm3, mean_60, feature_names):
    ts = pd.DatetimeIndex([pd.Timestamp(timestamp)])
    row = pd.DataFrame({
        "Global_active_power": [current_power], "Global_reactive_power": [reactive], "Voltage": [voltage],
        "Global_intensity": [intensity], "Sub_metering_1": [sm1], "Sub_metering_2": [sm2],
        "Sub_metering_3": [sm3], "gap_mean_60min": [mean_60],
    })
    return add_calendar_features(row, ts)[feature_names]


def main():
    st.set_page_config(page_title="Electricity Power Forecast", page_icon="⚡")
    st.title("⚡ Household Electricity Power Forecast")
    st.caption("AIC354 Machine Learning Fundamentals – Assignment 3 · COMSATS University Islamabad, Lahore Campus")

    try:
        model, meta = load_artifacts()
    except Exception as e:  # missing / incompatible model file
        st.error(f"Could not load the trained model: {e}")
        st.stop()

    rng = meta["input_ranges_observed"]
    horizon = meta["horizon_minutes"]
    st.write(f"Enter the household's meter readings **now**. The model ({meta['best_algorithm']}) estimates the "
             f"**global active power {horizon} minutes later**, in **kW** (power, not energy).")

    c1, c2 = st.columns(2)
    d = c1.date_input("Date", value=datetime(2009, 1, 15).date())
    t = c2.time_input("Time (current)", value=dtime(19, 30))

    def num(label, key, default, help_):
        lo, hi = rng[key]
        return st.number_input(f"{label}  [observed range {lo:g} – {hi:g}]", value=float(default), step=0.01,
                               help=help_, format="%.3f")

    cur = num("Current global active power (kW)", "Global_active_power", 3.2, "Minute-averaged active power right now")
    mean60 = num("Average active power over the last 60 minutes (kW)", "gap_mean_60min", 2.8, "Mean of the past hour")
    rea = num("Global reactive power (kVAR)", "Global_reactive_power", 0.12, "Minute-averaged reactive power")
    volt = num("Voltage (V)", "Voltage", 237.5, "Minute-averaged voltage")
    inten = num("Global intensity (A)", "Global_intensity", 13.6, "Minute-averaged current")
    s1 = num("Sub-metering 1 – kitchen (Wh)", "Sub_metering_1", 0.0, "Energy in the last minute")
    s2 = num("Sub-metering 2 – laundry (Wh)", "Sub_metering_2", 1.0, "Energy in the last minute")
    s3 = num("Sub-metering 3 – water heater/AC (Wh)", "Sub_metering_3", 17.0, "Energy in the last minute")

    if st.button("Predict", type="primary"):
        values = {"Global_active_power": cur, "gap_mean_60min": mean60, "Global_reactive_power": rea,
                  "Voltage": volt, "Global_intensity": inten, "Sub_metering_1": s1, "Sub_metering_2": s2,
                  "Sub_metering_3": s3}
        problems = [f"{k} must be ≥ 0." for k, v in values.items() if v < 0]
        if not (100 <= volt <= 300):
            problems.append("Voltage should be between 100 V and 300 V.")
        if problems:
            for p in problems:
                st.error(p)
            return
        warn = [k for k, v in values.items() if not (rng[k][0] <= v <= rng[k][1])]
        if warn:
            st.warning("Outside the range seen in training (prediction may be unreliable): " + ", ".join(warn))
        try:
            row = build_feature_row(datetime.combine(d, t), cur, rea, volt, inten, s1, s2, s3, mean60,
                                    meta["feature_names_in_order"])
            pred = float(model.predict(row)[0])
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return
        pred = max(pred, 0.0)  # power cannot be negative
        target_time = datetime.combine(d, t) + pd.Timedelta(minutes=horizon)
        st.success(f"Estimated global active power at {target_time:%Y-%m-%d %H:%M}: **{pred:.2f} kW**")
        st.write(f"If that power were sustained for one hour it would correspond to ≈ **{pred:.2f} kWh** of energy.")
        st.info(f"This is a statistical **estimate**, not a guaranteed measurement. Typical error of this model on "
                f"unseen future data: about ±{meta['chronological_test_rmse_kw']:.2f} kW (RMSE).")

    with st.expander("About the model"):
        st.write({k: meta[k] for k in ["best_algorithm", "target", "target_unit", "training_rows",
                                       "cv_10fold_mean_rmse_kw", "chronological_test_rmse_kw",
                                       "persistence_baseline_test_rmse_kw"]})


main()
