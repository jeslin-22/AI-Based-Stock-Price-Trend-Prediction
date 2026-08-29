# AI-Based Stock Price Trend Prediction

An AI-based machine learning application that predicts whether a stock price may move **UP or DOWN** using historical stock market data and technical indicators.

The project uses a **Random Forest Classifier** and provides an interactive **Streamlit dashboard** for stock analysis and trend prediction.

##  Project Overview

Stock market prices are influenced by many factors and can be difficult to predict.

This project uses historical stock market data and technical indicators to predict the next-day stock price trend.

The system classifies the expected trend into two categories:

- 📈 **UP**
- 📉 **DOWN**

The application also provides an interactive dashboard for viewing stock prices, moving averages, technical indicators, prediction probabilities, and recent stock data.

##  Objectives

The main objectives of this project are:

1. Collect historical stock market data.
2. Preprocess the collected data.
3. Create useful technical indicators.
4. Train a machine learning model.
5. Predict the next-day stock price trend.
6. Display UP and DOWN prediction probabilities.
7. Visualize historical stock prices.
8. Provide an interactive web dashboard using Streamlit.


##  Features

- 📈 Predicts stock price trend as **UP or DOWN**
- 💰 Displays the current stock price
- 📊 Shows UP and DOWN prediction probabilities
- 📉 Displays historical stock price charts
- 📈 Provides 5-Day and 20-Day Moving Average analysis
- 📌 Displays RSI, Daily Return, and Volatility
- 📋 Shows recent stock market data
- 🔎 Allows users to enter different stock symbols
- 📅 Allows users to select historical data periods
- 🖥️ Provides an interactive Streamlit dashboard


##  Machine Learning Approach

This project uses a **Random Forest Classifier** to predict the next-day stock price trend.

### Input Features

The model uses the following features:

- Open Price
- High Price
- Low Price
- Closing Price
- Trading Volume
- Daily Return
- 5-Day Moving Average
- 20-Day Moving Average
- Volatility
- RSI (Relative Strength Index)

### Target Variable

The target is based on the next day's closing price:

- `1` → Price goes **UP**
- `0` → Price goes **DOWN**

The historical dataset is divided using a time-based split:

- **80%** → Training data
- **20%** → Testing data

This preserves the chronological order of the stock market data.


##  Project Workflow
```text

The project follows these steps:

Historical Stock Data
        ↓
Data Collection
        ↓
Data Preprocessing
        ↓
Feature Engineering
        ↓
Technical Indicators
        ↓
Random Forest Classifier
        ↓
Trend Prediction
        ↓
Streamlit Dashboard





## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming Language |
| Pandas | Data Processing |
| NumPy | Numerical Computation |
| Scikit-learn | Machine Learning |
| Random Forest | Classification Model |
| yfinance | Stock Market Data |
| Matplotlib | Data Visualization |
| Streamlit | Web Application |
| Joblib | Model Saving and Loading |
| Git | Version Control |
| GitHub | Project Repository |



##  Project Structure
```text
AI-Based-Stock-Price-Trend-Prediction/
│
├── screenshots/
│   ├── prediction.png
│   ├── moving-average.png
│   └── technical-indicators.png
│
├── .gitignore
├── app.py
├── data_preprocessing.py
├── model.pkl
├── requirements.txt
├── train_model.py
└── README.md



## How the Project Works

1. Enter Stock Symbol – The user enters a stock symbol.
2. Collect Data – Historical stock data is downloaded using "yfinance".
3. Preprocess Data – The data is cleaned and required features are created.
4. Feature Extraction – Important stock features are selected for prediction.
5. Machine Learning Model – The trained model analyzes the historical data.
6. Trend Prediction – The system predicts whether the stock trend is Up or Down.
7. Display Results – The prediction and stock charts are displayed in the Streamlit application.


##  Installation

### 1. Clone the repository

```bash
git clone
 https://github.com/jeslin-22/AI-Based-Stock-Price-Trend-Prediction.git


 ### 2. Open the project folder

```bash
cd AI-Based-Stock-Price-Trend-Prediction

### 3. Create a virtual environment

python -m venv venv

### 4. Activate the virtual environment

For Windows:

venv\Scripts\activate

### 5. Install the required libraries

pip install -r requirements.txt


### 6. Run the application

streamlit run app.py

### 7. Deactivate the virtual environment

After completing the project, you can deactivate the virtual environment using:

deactivate


## Limitations

- The prediction depends on the quality and availability of historical stock data.
- Stock prices can be affected by unexpected news and market events.
- The system cannot guarantee accurate future stock prices.
- The model predicts the trend, not the exact future stock price.
- Market conditions can change quickly, which may affect prediction accuracy.

##Future Scope

- Improve prediction accuracy using advanced Machine Learning and Deep Learning models.
- Add real-time stock market data.
- Include news and social media sentiment analysis.
- Predict multiple stocks simultaneously.
- Add more interactive charts and dashboards.
- Deploy the application on a cloud platform for public access.


## Conclusion

The AI-Based Stock Price Trend Prediction System uses historical stock market data and Machine Learning to predict the future stock trend as Up or Down. The project provides a simple and user-friendly Streamlit interface for entering stock symbols, viewing historical data, and getting predictions. It demonstrates how AI and Machine Learning can be applied to analyze stock market trends and support better decision-making.


## References

- Python Documentation
- Pandas Documentation
- NumPy Documentation
- Matplotlib Documentation
- Scikit-learn Documentation
- Streamlit Documentation
- yfinance Documentation
- Yahoo Finance – Historical Stock Market Data
