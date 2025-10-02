import streamlit as st
import tensorflow as tf
import pandas as pd
import pickle
from pathlib import Path

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource
def load_model():
    """Load the trained ANN model."""
    model_path = Path("./models/model.keras")
    return tf.keras.models.load_model(model_path)


@st.cache_resource
def load_preprocessors():
    """Load all preprocessing objects."""
    with open("./models/label_encoder_gender.pkl", "rb") as f:
        label_encoder_gender = pickle.load(f)

    with open("./models/onehot_encoder_geo.pkl", "rb") as f:
        onehot_encoder_geo = pickle.load(f)

    with open("./models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return label_encoder_gender, onehot_encoder_geo, scaler


model = load_model()
label_encoder_gender, onehot_encoder_geo, scaler = load_preprocessors()

FEATURE_CONFIG = {
    "Geography": {
        "type": "selectbox",
        "options": onehot_encoder_geo.categories_[0].tolist(),
        "default": 0,
        "help": "Customer's country of residence",
    },
    "Gender": {
        "type": "selectbox",
        "options": label_encoder_gender.classes_.tolist(),
        "default": 0,
        "help": "Customer's gender",
    },
    "Age": {
        "type": "slider",
        "min": 18,
        "max": 92,
        "default": 35,
        "help": "Customer's age in years",
    },
    "CreditScore": {
        "type": "number",
        "min": 300,
        "max": 900,
        "default": 650,
        "step": 1,
        "help": "Customer's credit score",
    },
    "Balance": {
        "type": "number",
        "min": 0.0,
        "max": 250000.0,
        "default": 0.0,
        "step": 100.0,
        "help": "Account balance",
    },
    "EstimatedSalary": {
        "type": "number",
        "min": 0.0,
        "max": 200000.0,
        "default": 50000.0,
        "step": 1000.0,
        "help": "Customer's estimated annual salary",
    },
    "Tenure": {
        "type": "slider",
        "min": 0,
        "max": 10,
        "default": 5,
        "help": "Years as a customer",
    },
    "NumOfProducts": {
        "type": "slider",
        "min": 1,
        "max": 4,
        "default": 2,
        "help": "Number of bank products used",
    },
    "HasCrCard": {
        "type": "selectbox",
        "options": ["No", "Yes"],
        "default": 1,
        "help": "Does customer have a credit card?",
    },
    "IsActiveMember": {
        "type": "selectbox",
        "options": ["No", "Yes"],
        "default": 1,
        "help": "Is customer an active member?",
    },
}


def create_input_widget(feature_name, config):
    """Dynamically create input widget based on configuration."""
    if config["type"] == "selectbox":
        return st.selectbox(
            feature_name,
            options=config["options"],
            index=config["default"],
            help=config.get("help"),
        )
    elif config["type"] == "slider":
        return st.slider(
            feature_name,
            min_value=config["min"],
            max_value=config["max"],
            value=config["default"],
            help=config.get("help"),
        )
    elif config["type"] == "number":
        return st.number_input(
            feature_name,
            min_value=config["min"],
            max_value=config["max"],
            value=config["default"],
            step=config.get("step", 1),
            help=config.get("help"),
        )


def preprocess_input(user_inputs):
    """Preprocess user inputs for model prediction."""
    gender_encoded = label_encoder_gender.transform([user_inputs["Gender"]])[0]

    has_cr_card = 1 if user_inputs["HasCrCard"] == "Yes" else 0
    is_active_member = 1 if user_inputs["IsActiveMember"] == "Yes" else 0

    input_data = pd.DataFrame(
        {
            "CreditScore": [user_inputs["CreditScore"]],
            "Gender": [gender_encoded],
            "Age": [user_inputs["Age"]],
            "Tenure": [user_inputs["Tenure"]],
            "Balance": [user_inputs["Balance"]],
            "NumOfProducts": [user_inputs["NumOfProducts"]],
            "HasCrCard": [has_cr_card],
            "IsActiveMember": [is_active_member],
            "EstimatedSalary": [user_inputs["EstimatedSalary"]],
        }
    )

    geo_encoded = onehot_encoder_geo.transform([[user_inputs["Geography"]]]).toarray()
    geo_encoded_df = pd.DataFrame(
        geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(["Geography"])
    )

    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

    input_scaled = scaler.transform(input_data)

    return input_scaled


st.title("🤖 Customer Churn Prediction")
st.markdown("Predict customer churn probability using deep learning.")
st.divider()

left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.subheader("📋 Customer Information")

    user_inputs = {}

    input_col1, input_col2 = st.columns(2)

    features = list(FEATURE_CONFIG.keys())
    mid_point = len(features) // 2

    with input_col1:
        for feature in features[:mid_point]:
            user_inputs[feature] = create_input_widget(feature, FEATURE_CONFIG[feature])

    with input_col2:
        for feature in features[mid_point:]:
            user_inputs[feature] = create_input_widget(feature, FEATURE_CONFIG[feature])

    predict_button = st.button(
        "🚀 Predict Churn", type="primary", use_container_width=True
    )

with right_col:
    st.subheader("📊 Prediction Results")

    if predict_button:
        with st.spinner("Analyzing customer data..."):
            input_scaled = preprocess_input(user_inputs)

            prediction = model.predict(input_scaled, verbose=0)
            churn_probability = float(prediction[0][0])

        col_metric1, col_metric2 = st.columns(2)

        with col_metric1:
            st.metric("Churn Probability", f"{churn_probability * 100:.1f}%")

        with col_metric2:
            st.metric("Retention Probability", f"{(1 - churn_probability) * 100:.1f}%")

        risk_level = (
            "High"
            if churn_probability > 0.7
            else "Medium" if churn_probability > 0.5 else "Low"
        )
        st.metric("Risk Level", risk_level)

        st.progress(churn_probability, text="Churn Risk")

        st.divider()

        if churn_probability > 0.5:
            st.error(
                f"⚠️ **High Risk**: This customer has a {churn_probability * 100:.1f}% probability of churning. "
                "Consider immediate retention strategies."
            )
        else:
            st.success(
                f"✅ **Low Risk**: This customer has a {churn_probability * 100:.1f}% probability of churning. "
                "Customer retention is likely."
            )
    else:
        st.info(
            "👈 Enter customer information and click 'Predict Churn' to see results."
        )

st.divider()
st.caption("Powered by TensorFlow and Streamlit")
