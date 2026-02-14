import streamlit as st
import joblib
import numpy as np

page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at 20% 20%, #a1c4fd, #c2e9fb, #89f7fe, #66a6ff);
    background-size: 200% 200%;
    animation: pulseMove 6s ease-in-out infinite;
}

@keyframes pulseMove {
    0% {background-position: 0% 0%;}
    50% {background-position: 100% 100%;}
    100% {background-position: 0% 0%;}
}
</style>
"""

st.markdown(page_bg, unsafe_allow_html=True)
st.title("AD CLICK PREDICTION")
st.write("Fill the values below to predict whether the user will click the ad.")

daily_time = st.number_input("Daily Time Spent on Site", min_value=0.0)
age = st.number_input("Age", min_value=0)
area_income = st.number_input("Area Income", min_value=0.0)
daily_internet = st.number_input("Daily Internet Usage", min_value=0.0)
male = st.selectbox("Gender (Male=1, Female=0)", [0, 1])

if st.button("Predict"):
    X = np.array([[daily_time, age, area_income, daily_internet, male]])
    prediction = model.predict(X)[0]

    if prediction == 1:
        st.success("The user is likely to CLICK the ad.")
    else:
        st.error("The user is NOT likely to click the ad.")