import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 1. Load data
df = pd.read_csv("../data/housing_data.csv")

# 2. Encode categorical variables (same as your notebook)
df_encoded = pd.get_dummies(df, drop_first=True)

# 3. Define features and target
X = df_encoded.drop('price', axis=1)
y = df_encoded['price']

# 4. Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Make a prediction
y_pred = model.predict(X_test)

# 7. Print results
print("Sample prediction:", y_pred[0])
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

# 8. Save the trained model
import joblib
joblib.dump(model, "model.pkl")
print("Model saved to model.pkl")
