"""
Heart Disease Prediction - Streamlit Web App
--------------------------------------------
Loads the Support Vector Classifier trained in "Heart_Disease_Prediction.ipynb"
and lets a user enter four clinical attributes to get a prediction.

The app reproduces the notebook's pipeline EXACTLY:
  * the same LabelEncoders (fit on the same fixed category lists),
  * the same fitted StandardScaler (loaded from scaler.pkl),
  * the same feature order: Age, Sex, ChestPain, MaxHeartRate.

Keep svc_trained_model.pkl and scaler.pkl in the same folder as this file.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "svc_trained_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"


@st.cache_resource
def load_artifacts():
    for path in (MODEL_PATH, SCALER_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"{path.name} was not found. Put it in the same folder as app.py."
            )
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


@st.cache_resource
def build_label_encoders():
    """
    Recreates the exact LabelEncoders used in the notebook. LabelEncoder assigns
    codes from the sorted category list, so fitting on the same fixed lists
    reproduces the same codes deterministically.
    """
    return {
        "Sex": LabelEncoder().fit(["Male", "Female"]),
        "ChestPain": LabelEncoder().fit(
            ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"]
        ),
    }


model, scaler = load_artifacts()
encoders = build_label_encoders()

# ---------------------------------------------------------------------------
# Page content
# ---------------------------------------------------------------------------

st.title("❤️ Heart Disease Prediction System")
st.markdown(
    """
    A teaching demo built from the **Heart Disease Prediction** notebook
    (UCI Heart Disease dataset). Enter a patient's details and the trained
    **Support Vector Classifier** will predict whether heart disease is present.

    *Model accuracy on held-out test data: **79%** (4 input attributes).*
    """
)

st.divider()

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age (years)", min_value=1, max_value=120, value=55, step=1)
    sex = st.selectbox("Sex", ["Male", "Female"])
with col2:
    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"],
    )
    max_hr = st.number_input(
        "Maximum Heart Rate Achieved (bpm)", min_value=60, max_value=220, value=130, step=1
    )

if st.button("Predict", type="primary", use_container_width=True):
    user_input = pd.DataFrame(
        {
            "Age": [float(age)],
            "Sex": [sex],
            "ChestPain": [chest_pain],
            "MaxHeartRate": [float(max_hr)],
        }
    )

    encoded = pd.DataFrame(
        {
            "Age": user_input["Age"],
            "Sex": encoders["Sex"].transform(user_input["Sex"]),
            "ChestPain": encoders["ChestPain"].transform(user_input["ChestPain"]),
            "MaxHeartRate": user_input["MaxHeartRate"],
        }
    )

    scaled = scaler.transform(encoded)
    prediction = model.predict(scaled)[0]

    st.divider()
    if prediction == 1:
        st.error("### ⚠️ HEART DISEASE DETECTED")
    else:
        st.success("### ✅ NO HEART DISEASE")

    with st.expander("See the encoded feature vector sent to the model"):
        st.dataframe(encoded, hide_index=True)
        st.caption("Scaled values:")
        st.dataframe(pd.DataFrame(np.round(scaled, 3), columns=encoded.columns), hide_index=True)

st.divider()
st.caption(
    "Educational demo using only 4 of the 13 available attributes from the UCI Heart "
    "Disease dataset. Not a medical device - do not use for real clinical decisions."
)
