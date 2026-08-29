import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from data_preprocessing import (
    download_stock_data,
    create_features
)

TICKER = "AAPL"

print("Downloading stock data...")

data = download_stock_data(TICKER, period="5y")

print("Data downloaded successfully!")
df = create_features(data)

print("\nFeatures created:")
print(df.tail())
features = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Daily_Return",
    "MA_5",
    "MA_20",
    "Volatility",
    "RSI"
]

X = df[features]
y = df["Target"]

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


print("\nTraining data:", len(X_train))
print("Testing data:", len(X_test))

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    class_weight="balanced"
)

print("\nTraining AI model...")

model.fit(X_train, y_train)

print("Training completed!")
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:")
print(f"{accuracy * 100:.2f}%")


print("\nClassification Report:")
print(classification_report(y_test, predictions))
joblib.dump(
    {
        "model": model,
        "features": features
    },
    "model.pkl"
)

print("\nModel saved as model.pkl")