import streamlit as st
import numpy as np
import joblib

# -------------------------------
# LOAD MODEL & SCALER
# -------------------------------
model = joblib.load("ridge_model.pkl")
scaler = joblib.load("scaler.pkl")

# -------------------------------
# COLUMN NAMES (ACTUAL)
# -------------------------------
columns = [
    "symboling",
    "wheelbase",
    "carlength",
    "carwidth",
    "curbweight",
    "enginesize",
    "horsepower",
    "citympg",
    "carbody_hardtop",
    "carbody_hatchback",
    "carbody_sedan",
    "carbody_wagon",
    "drivewheel_fwd",
    "drivewheel_rwd",
    "enginelocation_rear",
    "enginetype_dohcv",
    "enginetype_l",
    "enginetype_ohc",
    "enginetype_ohcf",
    "enginetype_ohcv",
    "enginetype_rotor",
    "cylindernumber_five",
    "cylindernumber_four",
    "cylindernumber_six",
    "cylindernumber_three",
    "cylindernumber_twelve",
    "cylindernumber_two"
]

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="wide")

# -------------------------------
# HEADER
# -------------------------------
st.markdown("<h1 style='text-align: center;'>🚗 Car Price Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Fill all features below to predict car price 💰</p>", unsafe_allow_html=True)

st.write("---")

# -------------------------------
# SHOW COLUMN NAMES
# -------------------------------
# st.subheader("📌 Model Column Names")

# st.code(columns, language="python")

# st.write("---")

# -------------------------------
# INPUT SECTION
# -------------------------------
st.subheader("📊 Enter Feature Values")

col1, col2 = st.columns(2)

inputs = []

for i, col_name in enumerate(columns):
    
    # Numerical features
    if i < 8:
        if i % 2 == 0:
            value = col1.number_input(col_name, value=0.0)
        else:
            value = col2.number_input(col_name, value=0.0)
    
    # Categorical (0/1)
    else:
        if i % 2 == 0:
            value = col1.selectbox(col_name, [0, 1])
        else:
            value = col2.selectbox(col_name, [0, 1])

    inputs.append(value)

# -------------------------------
# PREDICTION
# -------------------------------
st.write("")

if st.button("🔮 Predict Price", use_container_width=True):

    input_array = np.array(inputs).reshape(1, -1)
    scaled_input = scaler.transform(input_array)
    prediction = model.predict(scaled_input)

    st.success(f"💰 Predicted Car Price: {prediction[0]:,.2f}")

# -------------------------------
# FOOTER
# -------------------------------
st.write("---")
st.markdown("<p style='text-align: center;'>Made with ❤️ using Streamlit</p>", unsafe_allow_html=True)