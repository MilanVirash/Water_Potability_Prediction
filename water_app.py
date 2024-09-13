import streamlit as st
import numpy as np
from joblib import load

model = load("water_randomforest.joblib")

st.title("💧Water Potability Prediction App")

st.subheader("Enter The Values Of Each Parameter")

# Input fields for each parameter
pH = st.number_input("pH", min_value=0.000000, max_value=14.000000)
Hardness = st.number_input("Hardness", min_value=47.000000, max_value=324.000000)
Solids = st.number_input("Solids", min_value=320.000000, max_value=61228.000000)
Chloramines = st.number_input("Chloramines", min_value=0.000000, max_value=14.000000)
Sulfate = st.number_input("Sulfate", min_value=129.000000, max_value=482.000000)
Conductivity = st.number_input("Conductivity", min_value=181.000000, max_value=754.000000)
Organic_carbon = st.number_input("Organic Carbon", min_value=2.000000, max_value=29.000000)
Trihalomethanes = st.number_input("Trihalomethanes", min_value=0.000000, max_value=124.000000)
Turbidity = st.number_input("Turbidity", min_value=1.000000, max_value=7.000000)

# Prepare the input data for prediction
prediction_input = np.array([[pH, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity]])

# Prediction button
if st.button('🚰 Predict Water Potability'):
    with st.spinner('Analyzing water quality...'):
        prediction = model.predict(prediction_input)

    # Display the prediction result
    st.subheader("🚀 Prediction Result")
    
    if prediction[0] == 1:
        st.markdown("<div class='success-box'>✅ The water is Potable! It is safe to drink.</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='error-box'>❌ The water is Not Potable! It is not safe to drink.</div>", unsafe_allow_html=True)