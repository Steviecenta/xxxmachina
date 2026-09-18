import sys
from pathlib import Path

import joblib

# Saving the trained pipeline and metadata
joblib.dump(final_pipeline, "house_price_model_pipeline.pkl")
joblib.dump(metadata, "house_price_app_metadata.pkl")
import pandas as pd
import streamlit as st


# ==============================================================================
# 1. DEFINE YOUR CUSTOM FUNCTIONS / TRANSFORMERS HERE
# Any custom function or class defined in your notebook and used inside your
# pipeline must be defined in app.py before loading the model.
# ==============================================================================

# IMPORTANT:
# Replace the function below with the EXACT implementation used when the model
# was trained/saved. The placeholder is only sufficient for testing the app
# structure and for unpickling when the saved object references this function.
def handle_ames_missing_values(df):
    df = df.copy()
    # TODO: Put your actual missing-value handling logic here.
    return df


# MAP TO __main__ SCOPE (failsafe for joblib pickling references):
# If joblib saved the function under the notebook's __main__ scope, this makes
# it available under the same name during unpickling inside Streamlit.
sys.modules["__main__"].handle_ames_missing_values = handle_ames_missing_values


# ==============================================================================
# 2. CACHED ARTIFACT LOADING
# ==============================================================================

@st.cache_resource
def load_artifacts():
    base_dir = Path(__file__).resolve().parent
    model_path = base_dir / "model.joblib"
    metadata_path = base_dir / "metadata.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Place model.joblib in the same folder as app.py."
        )

    # Load model
    model = joblib.load(model_path)

    # Load metadata if available; otherwise extract feature names from model.
    if metadata_path.exists():
        metadata = joblib.load(metadata_path)
    else:
        metadata = {
            "features": list(getattr(model, "feature_names_in_", []))
        }

    return model, metadata


# ==============================================================================
# 3. STREAMLIT UI & PREDICTION LOGIC
# ==============================================================================

st.set_page_config(page_title="Real Estate Predictor", layout="wide")
st.title("🏡 Real Estate Price Prediction")

# Load model and metadata
try:
    model, metadata = load_artifacts()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


# App UI Inputs
st.sidebar.header("Input Features")

overall_qual = st.sidebar.slider(
    "Overall Quality (1-10)",
    min_value=1,
    max_value=10,
    value=5,
)

gr_liv_area = st.sidebar.number_input(
    "Above Ground Living Area (sq ft)",
    min_value=0,
    value=1500,
)

total_bsmt_sf = st.sidebar.number_input(
    "Total Basement (sq ft)",
    min_value=0,
    value=1000,
)


# Build DataFrame for prediction
input_df = pd.DataFrame(
    [
        {
            "Overall Qual": overall_qual,
            "Gr Liv Area": gr_liv_area,
            "Total Bsmt SF": total_bsmt_sf,
        }
    ]
)


st.subheader("Input Summary")
st.dataframe(input_df, use_container_width=True)


# Display model features when they are available.
model_features = list(getattr(model, "feature_names_in_", []))

if model_features:
    missing_features = [
        feature for feature in model_features
        if feature not in input_df.columns
    ]

    if missing_features:
        st.warning(
            "The loaded model expects additional input features that are not "
            "currently included in the app."
        )
        st.write("Expected features:")
        st.write(model_features)


# Prediction
if st.button("Predict House Price"):
    try:
        # Validate feature compatibility when the model exposes its feature names.
        if model_features:
            missing_features = [
                feature for feature in model_features
                if feature not in input_df.columns
            ]

            if missing_features:
                st.error(
                    "Prediction cannot run because the following model features "
                    f"are missing: {', '.join(missing_features)}"
                )
                st.stop()

        prediction = model.predict(input_df)[0]

        # Convert NumPy scalar / other numeric scalar types to a Python float.
        prediction = float(prediction)

        st.success(
            f"Estimated Market Value: **${prediction:,.2f}**"
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
