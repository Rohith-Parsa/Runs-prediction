import streamlit as st
import joblib
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

df = pd.read_csv("playerstat.csv")

# Streamlit config
st.set_page_config(layout="wide")
st.sidebar.title("🧭 Navigation")
option = st.sidebar.radio("Go to", ["🏠 Home", "📁 Dataset", "📊 Visualizations","🧠 Predictor"])

# Title
st.markdown("<h1 style='text-align: center;'>Batsman runs prediction</h1>", unsafe_allow_html=True)
if option == "🏠 Home":
    st.title("🏏 Batsman Runs Prediction")

    st.markdown("""
    <style>
    .big-font {
        font-size:22px !important;
    }
    .small-font {
        font-size:16px !important;
        color: #6c757d;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<p class='big-font'>🔍 What this app does:</p>", unsafe_allow_html=True)

    st.markdown("""
    - 📊 **Displays dataset** used for training the Runs Predictor model.
    - 📈 **Visualizes batting trends** such as Strike Rate, Runs vs Balls Faced, etc.
    - 🧠 **Predicts total career runs** using:
        - 🤖 Random Forest
        - 📉 Linear Regression

    ---

    👉 Use the **sidebar** to navigate through:
    - `📁 Dataset`
    - `📊 Visualizations`
    - `🧠 Predictor`

    ---
    
    👨‍💻 <span class='small-font'>Developed by: <b>P. Rohith</b>, <b>MD. Sandhani</b>, <b>SK. Jahiruddin</b></span>
    """, unsafe_allow_html=True)



# Dataset
elif option == "📁 Dataset":
    st.subheader("📁 Training Dataset")
    st.dataframe(df)
# Visualizations
elif option == "📊 Visualizations":
    st.subheader("📊 Visualizations")
    df['Ave'] = pd.to_numeric(df['Ave'], errors='coerce')
    df['BF'] = pd.to_numeric(df['BF'], errors='coerce')
    df['Runs'] = pd.to_numeric(df['Runs'], errors='coerce')
    df['SR'] = pd.to_numeric(df['SR'], errors='coerce')

    # Drop rows where any of the important columns are missing
    df = df.dropna(subset=['Ave', 'BF', 'Runs', 'SR'])

    # Now plot
    fig = px.scatter(
    df,
    x='BF',
    y='Runs',
    color='SR',
    size='Ave',
    size_max=40,
    hover_data=['100', '50', '0'],
    title='🏏 Balls Faced vs Runs (Color by Strike Rate, Size by Average)')
    st.plotly_chart(fig, use_container_width=True)
    fig = px.histogram(df, x='Ave', nbins=50, marginal='rug',title=' Distribution of Batting Averages')
    st.plotly_chart(fig, use_container_width=True)
    fig = px.box(df, x='100', y='Ave',title=' Batting Average vs Number of Centuries')
    st.plotly_chart(fig, use_container_width=True)
    fig = px.scatter(df, x='Inns', y='Runs', title='Total Runs with Innings Played')
    st.plotly_chart(fig, use_container_width=True)

# Predictor
elif option == "🧠 Predictor":
    st.subheader("🏏 Predict Total Career Runs")

    # Load models and scaler
    rf = joblib.load("runs_model.pkl")  # Random Forest model
    best_lr = joblib.load("linear_model.pkl")  # Linear Regression model
    scaler = joblib.load("runs_scaler.pkl")  # StandardScaler

    st.markdown("### 📥 Enter the Batsman's Career Stats:")

    col1, col2 = st.columns(2)
    with col1:
        innings = st.number_input("Innings Played", min_value=0, value=10)
        notouts = st.number_input("Not Outs", min_value=0, value=2)
        bf = st.number_input("Balls Faced (BF)", min_value=0, value=500)
    with col2:
        centuries = st.number_input("No. of 100s", min_value=0, value=1)
        half_centuries = st.number_input("No. of 50s", min_value=0, value=3)

    if st.button("🚀 Predict Runs"):
        try:
            # Prepare input
            input_data = np.array([[innings, notouts, bf, centuries, half_centuries]])
            input_scaled = scaler.transform(input_data)

            # Predict with both models
            pred_rf = best_rf.predict(input_scaled)[0]
            pred_lr = lr.predict(input_scaled)[0]

            # Display Results
            st.success(f"🌲 Random Forest Prediction: **{int(round(pred_rf))} runs**")
            st.info(f"📉 Linear Regression Prediction: **{int(round(pred_lr))} runs**")

        except Exception as e:
            st.error(f"⚠️ Prediction failed. Error: {e}")
