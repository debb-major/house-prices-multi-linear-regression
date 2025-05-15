import pandas as pd
import joblib

# 1. Load the saved model
model = joblib.load("model.pkl")

# 2. Load and preprocess the dataset (just for feature structure)
df = pd.read_csv("../data/housing_data.csv")
df_encoded = pd.get_dummies(df, drop_first=True)

# 3. Drop target column to get only features
X = df_encoded.drop('price', axis=1)

# 4. Pick a sample row to predict (e.g., the first one)
sample = X.iloc[[0]]  # note: double brackets [[0]] to keep it as a DataFrame

# 5. Predict using the loaded model
prediction = model.predict(sample)

print("Prediction for first house:", prediction[0])
