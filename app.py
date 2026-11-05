import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).parent

st.set_page_config(
    page_title="California House Price Predictor",
    page_icon="🏡",
    layout="centered",
)

@st.cache_resource
def load_artifacts():
    with open(BASE_DIR / "model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    with open(BASE_DIR / "scaling_new.pkl", "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
    return model, scaler

st.title("🏡 California House Price Predictor")
st.write(
    "Enter the neighborhood and housing details below to estimate "
    "the median house value using the trained machine-learning model."
)

try:
    model, scaler = load_artifacts()
except Exception as exc:
    st.error(f"Could not load the model files: {exc}")
    st.stop()

with st.form("prediction_form"):
    st.subheader("Property and neighborhood details")
    col1, col2 = st.columns(2)

    with col1:
        med_inc = st.number_input("Median income (MedInc)", min_value=0.0, value=3.5, step=0.1)
        house_age = st.number_input("House age (years)", min_value=0.0, value=25.0, step=1.0)
        ave_rooms = st.number_input("Average rooms (AveRooms)", min_value=0.0, value=5.0, step=0.1)
        ave_bedrms = st.number_input("Average bedrooms (AveBedrms)", min_value=0.0, value=1.0, step=0.1)

    with col2:
        population = st.number_input("Population", min_value=0.0, value=1000.0, step=10.0)
        ave_occup = st.number_input("Average occupancy (AveOccup)", min_value=0.0, value=3.0, step=0.1)
        latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=34.0, step=0.1)
        longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=-118.0, step=0.1)

    submitted = st.form_submit_button("Predict house price", type="primary", use_container_width=True)

if submitted:
    if ave_rooms <= 0 or ave_bedrms <= 0 or ave_occup <= 0:
        st.error("Average rooms, average bedrooms, and average occupancy must be greater than zero.")
    else:
        feature_names = [
            "MedInc", "HouseAge", "AveRooms", "AveBedrms",
            "Population", "AveOccup", "Latitude", "Longitude"
        ]
        input_data = pd.DataFrame(
            [[med_inc, house_age, ave_rooms, ave_bedrms,
              population, ave_occup, latitude, longitude]],
            columns=feature_names,
        )
        try:
            scaled_data = scaler.transform(input_data)
            prediction = float(model.predict(scaled_data)[0])
            st.success(f"Estimated median house value: **${prediction * 100000:,.0f}**")
            st.caption(
                "The model predicts California housing values in units of $100,000. "
                "This is an estimate, not a property appraisal or guaranteed sale price."
            )
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

with st.expander("About this project"):
    st.write(
        "This application uses a saved Linear Regression model and StandardScaler. "
        "The model expects these eight California housing features: MedInc, HouseAge, "
        "AveRooms, AveBedrms, Population, AveOccup, Latitude, and Longitude."
    )
