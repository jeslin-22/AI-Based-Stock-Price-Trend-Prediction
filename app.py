import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from data_preprocessing import (
    download_stock_data,
    create_features
)


# ----------------------------------
# Page configuration
# ----------------------------------

st.set_page_config(
    page_title="AI Stock Trend Prediction",
    page_icon="📈",
    layout="wide"
)


# ----------------------------------
# Title
# ----------------------------------

st.title("📈 AI-Based Stock Price Trend Prediction")

st.write(
    "This application uses Machine Learning to predict "
    "whether a stock price may move UP or DOWN."
)


# ----------------------------------
# Load model
# ----------------------------------

try:
    saved_model = joblib.load("model.pkl")

    model = saved_model["model"]
    features = saved_model["features"]

except FileNotFoundError:

    st.error(
        "model.pkl not found. Please run train_model.py first."
    )

    st.stop()


# ----------------------------------
# Sidebar
# ----------------------------------

st.sidebar.header("Stock Selection")

ticker = st.sidebar.text_input(
    "Enter Stock Symbol",
    "AAPL"
).upper()

period = st.sidebar.selectbox(
    "Historical Data",
    ["1y", "2y", "5y", "10y"]
)


# ----------------------------------
# Load stock data
# ----------------------------------

if st.sidebar.button("Predict Trend"):

    try:

        with st.spinner("Downloading stock data..."):

            data = download_stock_data(
                ticker,
                period
            )

            df = create_features(data)


        # ----------------------------------
        # Latest data
        # ----------------------------------

        latest = df.iloc[-1]

        latest_features = df[features].iloc[-1:]

        prediction = model.predict(
            latest_features
        )[0]

        probabilities = model.predict_proba(
            latest_features
        )[0]


        # ----------------------------------
        # Prediction
        # ----------------------------------

        st.subheader(
            f"Prediction for {ticker}"
        )

        if prediction == 1:

            st.success(
                "📈 Predicted Trend: UP"
            )

        else:

            st.error(
                "📉 Predicted Trend: DOWN"
            )


        # ----------------------------------
        # Probability
        # ----------------------------------

        up_probability = probabilities[1] * 100
        down_probability = probabilities[0] * 100


        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Current Price",
                f"${latest['Close']:.2f}"
            )


        with col2:

            st.metric(
                "UP Probability",
                f"{up_probability:.2f}%"
            )


        with col3:

            st.metric(
                "DOWN Probability",
                f"{down_probability:.2f}%"
            )


        # ----------------------------------
        # Stock price chart
        # ----------------------------------

        st.subheader(
            "Historical Stock Price"
        )

        fig, ax = plt.subplots(
            figsize=(12, 5)
        )

        ax.plot(
            df.index,
            df["Close"]
        )

        ax.set_xlabel("Date")

        ax.set_ylabel("Closing Price")

        ax.set_title(
            f"{ticker} Stock Price"
        )

        ax.grid(True)

        st.pyplot(fig)


        # ----------------------------------
        # Moving Average chart
        # ----------------------------------

        st.subheader(
            "Moving Average Analysis"
        )

        fig2, ax2 = plt.subplots(
            figsize=(12, 5)
        )

        ax2.plot(
            df.index,
            df["Close"],
            label="Close Price"
        )

        ax2.plot(
            df.index,
            df["MA_5"],
            label="5-Day MA"
        )

        ax2.plot(
            df.index,
            df["MA_20"],
            label="20-Day MA"
        )

        ax2.set_xlabel("Date")

        ax2.set_ylabel("Price")

        ax2.set_title(
            f"{ticker} Moving Average"
        )

        ax2.legend()

        ax2.grid(True)

        st.pyplot(fig2)


        # ----------------------------------
        # Technical indicators
        # ----------------------------------

        st.subheader(
            "Technical Indicators"
        )

        indicator_data = pd.DataFrame({

            "Indicator": [
                "RSI",
                "5-Day Moving Average",
                "20-Day Moving Average",
                "Daily Return",
                "Volatility"
            ],

            "Value": [
                f"{latest['RSI']:.2f}",
                f"{latest['MA_5']:.2f}",
                f"{latest['MA_20']:.2f}",
                f"{latest['Daily_Return'] * 100:.2f}%",
                f"{latest['Volatility']:.4f}"
            ]

        })

        st.table(indicator_data)


        # ----------------------------------
        # Recent data
        # ----------------------------------

        st.subheader(
            "Recent Stock Data"
        )

        st.dataframe(
            df[
                [
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "Volume"
                ]
            ].tail(10)
        )


    except Exception as e:

        st.error(
            f"Error: {e}"
        )


# ----------------------------------
# Footer
# ----------------------------------

st.markdown("---")

st.caption(
    "AI-Based Stock Price Trend Prediction | "
    "Machine Learning Project"
)