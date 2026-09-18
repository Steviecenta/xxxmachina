import sys
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

# ==============================================================================
# 1. CUSTOM FUNCTIONS (Failsafe for unpickling)
# ==============================================================================
def handle_ames_missing_values(df):
    df = df.copy()
    return df

sys.modules["__main__"].handle_ames_missing_values = handle_ames_missing_values

# ==============================================================================
# 2. CACHED ARTIFACT LOADING
# ==============================================================================
MODEL_PATH = "house_price_model_pipeline.pkl"
METADATA_PATH = "house_price_app_metadata.pkl"

@st.cache_resource
def load_artifacts():
    base_dir = Path(__file__).resolve().parent
    model_path = base_dir / MODEL_PATH
    metadata_path = base_dir / METADATA_PATH

    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "Model files not found. Make sure 'house_price_model_pipeline.pkl' "
            "and 'house_price_app_metadata.pkl' are in the same folder as app.py."
        )

    model = joblib.load(model_path)
    metadata = joblib.load(metadata_path)
    return model, metadata

# ==============================================================================
# 3. STREAMLIT UI & PREDICTION LOGIC
# ==============================================================================
st.set_page_config(page_title="HomeValue AI", layout="centered")
st.title("🏠 HomeValue AI")
st.subheader("House Price Prediction Demo App")

try:
    model, metadata = load_artifacts()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Extract metadata components
numeric_features = metadata.get("numeric_features", [])
categorical_features = metadata.get("categorical_features", [])
categorical_options = metadata.get("categorical_options", {})
numeric_defaults = metadata.get("numeric_defaults", {})

st.sidebar.header("Enter House Details")
user_input = {}

# Dynamically generate numeric inputs
for feature in numeric_features:
    default_value = numeric_defaults.get(feature, 0.0)
    if "Year" in feature:
        value = st.sidebar.number_input(feature, min_value=1800, max_value=2030, value=int(default_value), step=1)
    elif feature == "OverallQual":
        value = st.sidebar.slider(feature, min_value=1, max_value=10, value=int(round(default_value)))
    else:
        value = st.sidebar.number_input(feature, min_value=0.0, value=float(default_value), step=100.0)
    user_input[feature] = value

# Dynamically generate categorical inputs
for feature in categorical_features:
    options = categorical_options.get(feature, ["Unknown"])
    value = st.sidebar.selectbox(feature, options=options)
    user_input[feature] = value

input_df = pd.DataFrame([user_input])

st.write("### Input Summary")
st.dataframe(input_df)

if st.button("Predict House Price"):
    try:
        prediction = model.predict(input_df)[0]
        prediction = float(prediction)
        
        st.success(f"Estimated Sale Price: **${prediction:,.2f}**")
        
        lower_bound = prediction * 0.90
        upper_bound = prediction * 1.10
        st.info(f"Suggested range: ${lower_bound:,.2f} to ${upper_bound:,.2f}")
        
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.write("---")
st.caption("HomeValue AI Teaching Demo")
