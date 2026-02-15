import streamlit as st
import joblib as job
import numpy as np
import pandas as pd
from datetime import datetime
import plotly.express as px

df = pd.read_csv("class_marketing_advertising.csv")
country_city_map = df.groupby("Country")["City"].unique().apply(list).to_dict()
model = joblib.load("model.pkl")

page_bg = """
<style>

html, body, .stApp {
    background: linear-gradient(135deg, #F2EFE7, #9ACBD0, #48A6A7, #006A71);
    background-size: 400% 400%;
    animation: gradientMove 12s ease infinite;
}

@keyframes gradientMove {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

h1, h2 {
    color: #043c54 !important;
    text-align: center;
    font-weight: bold;
}

div.stSelectbox label, div.stNumberInput label {
    color: #043c54 !important;
    font-size: 18px;
    font-weight: bold;
}

label, p, span {
    color: #043c54 !important;
}

div.stButton > button {
    background-color: #48A6A7;
    color: #F2EFE7 !important;
    border-radius: 8px;
    height: 3em;
    width: 12em;
    font-size: 16px;
    display: block;
    margin-left: auto;
    margin-right: auto;
}

</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

st.image("PPC.png")
st.markdown("<h1>AD CLICK PREDICTION</h1>", unsafe_allow_html=True)

tabs = st.tabs(["🔮 Prediction", "📊 Analytics Dashboard"])

# ---------------------- PREDICTION TAB ----------------------
with tabs[0]:

    st.markdown("<p style='text-align:center; font-size:18px;'>Enter the user information to predict if they will click the ad</p>", unsafe_allow_html=True)

    daily_time = st.number_input("Daily Time Spent on Site (minutes)", min_value=0.0)
    age = st.number_input("Age", min_value=0)
    area_income = st.number_input("Area Income (USD)", min_value=0.0)
    daily_internet = st.number_input("Daily Internet Usage (minutes)", min_value=0.0)

    gender = st.selectbox("Gender", ["Female", "Male"])
    male = 1 if gender == "Male" else 0

    country = st.selectbox("Country", list(country_city_map.keys()))
    country_encoded = list(country_city_map.keys()).index(country)

    cities_list = country_city_map[country]
    city = st.selectbox("City", cities_list)
    city_encoded = cities_list.index(city)

    predict = st.button("Run Prediction")

    if predict:
        now = datetime.now()
        hour, day, month, year = now.hour, now.day, now.month, now.year
        clicked = 0
        ad_encoded = 1

        X = np.array([[daily_time, age, area_income, daily_internet, male,
                       clicked, ad_encoded, city_encoded, country_encoded,
                       hour, day, month, year]])

        prediction = model.predict(X)[0]

        if prediction == 1:
            st.markdown("""
            <style>
            .pulse {
                animation: pulse 1s ease-in-out 1;
            }
            @keyframes pulse {
                0% { transform: scale(1); }
                50% { transform: scale(1.3); }
                100% { transform: scale(1); }
            }
            </style>
            <h1 class="pulse" style='text-align:center; color:#19afb9; font-size:100px;'>✔️</h1>
            """, unsafe_allow_html=True)
            st.success("The user is likely to CLICK the ad.")

        else:
            st.markdown("""
            <style>
            .shake {
                animation: shake 0.3s ease-in-out 10;
            }
            @keyframes shake {
                0% { transform: translateX(0); }
                25% { transform: translateX(-10px); }
                50% { transform: translateX(10px); }
                75% { transform: translateX(-10px); }
                100% { transform: translateX(0); }
            }
            </style>
            <h1 class="shake" style='text-align:center; color:#dc3545; font-size:100px;'>❌</h1>
            """, unsafe_allow_html=True)
            st.error("The user is NOT likely to click the ad.")

# ---------------------- DASHBOARD TAB ----------------------
with tabs[1]:

    st.markdown("<h2>Analytics Dashboard</h2>", unsafe_allow_html=True)

    st.markdown("""
    <style>
    [data-testid="stMetricValue"] {
        color: #043c54 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #043c54 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Avg Age", round(df["Age"].mean(), 1))
    col2.metric("Avg Income", round(df["Area Income"].mean(), 1))
    col3.metric("Avg Time on Site", round(df["Daily Time Spent on Site"].mean(), 1))
    col4.metric("Avg Internet Use", round(df["Daily Internet Usage"].mean(), 1))
    col5.metric("Click Rate", f"{round(df['Clicked on Ad'].mean()*100, 1)}%")

    def transparent(fig):
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color="#043c54"
        )
        return fig

    st.plotly_chart(transparent(px.histogram(df, x="Age", title="Age Distribution", color_discrete_sequence=["#043c54"])))
    st.plotly_chart(transparent(px.histogram(df, x="Area Income", title="Income Distribution", color_discrete_sequence=["#043c54"])))
    st.plotly_chart(transparent(px.histogram(df, x="Daily Time Spent on Site", title="Time Spent on Site Distribution", color_discrete_sequence=["#043c54"])))
    st.plotly_chart(transparent(px.pie(df, names="Male", title="Gender Distribution (0=Female, 1=Male)", color_discrete_sequence=["#48A6A7", "#043c54"])))
    st.plotly_chart(transparent(px.pie(df, names="Clicked on Ad", title="Ad Click Distribution", color_discrete_sequence=["#48A6A7", "#043c54"])))
    st.plotly_chart(transparent(px.bar(df["Country"].value_counts().head(10), title="Top 10 Countries", color_discrete_sequence=["#043c54"])))
    st.plotly_chart(transparent(px.bar(df["City"].value_counts().head(10), title="Top 10 Cities", color_discrete_sequence=["#043c54"])))

# ---------------------- FOOTER ----------------------
st.markdown(
    """
    <hr style='margin-top:50px; border: 1px solid #043c54;'>

    <p style='text-align:center; font-size:22px; font-weight:bold; color:#043c54;'>
        مع منصتنا… كل نقرة تتحول إلى فرصة تسويقية أقرب للنجاح.
    </p>

    <p style='text-align:center; font-size:16px; font-weight:bold; color:#043c54;'>
        © 2026 AD CLICK PREDICTION — جميع الحقوق محفوظة.
    </p>
    """,
    unsafe_allow_html=True

)
