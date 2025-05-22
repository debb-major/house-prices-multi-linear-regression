# import modules
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# 1. Load data
df = pd.read_csv("data/housing_data.csv")

# 2. Encode categorical variables
df_encoded = pd.get_dummies(df, drop_first=True)

# 3. Define features and target
X = df_encoded.drop('price', axis=1)
y = df_encoded['price']

# 4. Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Make predictions
y_pred = model.predict(X_test)

# 7. Print and log results
sample_prediction = y_pred[0]
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Sample prediction:", sample_prediction)
print("Mean Squared Error:", mse)
print(f"Root Mean Squared Error: {rmse}")
print(f"R² Score: {r2}")

# 8. Save the model
joblib.dump(model, "model.pkl")
print("Model saved to model.pkl")

# 9. Log the output to a timestamped file
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"test_results_{timestamp}.txt"

with open(log_filename, 'w') as file:
    file.write(f"Sample prediction: {sample_prediction}\n")
    file.write(f"Mean Squared Error: {mse}\n")
    file.write("Model saved to model.pkl\n")

print("Log file created:", log_filename)
