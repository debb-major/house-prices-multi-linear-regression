# 🏠 Housing Price Prediction - Multiple Linear Regression

This project is a beginner-friendly walkthrough of a multiple linear regression model to predict house prices based on factors like area, number of bedrooms, bathrooms, and stories.

I had so much fun doing this project! It helped me deepen my understanding of how multiple variables interact in real-world data. 🎯

It builds on my first ML project and was a fun next step in understanding how multiple inputs influence a single outcome.

## 🚀 What This Project Covers
- 📊 Data loading and visualization
- 📈 Multiple linear regression model using scikit-learn
- 📉 Model evaluation using MAE and R² Score
- 🔮 Custom predictions using new input values

## 📌 What This Project Teaches
- How to prepare a dataset for machine learning
- How to apply Multiple Linear Regression using scikit-learn
- How to evaluate model performance
- How to interpret model coefficients
- How to visualize actual vs predicted results


## 🧠 Technologies Used
- Python (via Jupyter Notebook, Anaconda)
- pandas, numpy
- scikit-learn
- seaborn & matplotlib for visuals


## 📁 Project Structure

```
house-prices-multi-linear-regression/
├── data/
│   └── housing_data.csv
├── notebook/
│   └── house_price_multilinear_regression.ipynb
│   └── test_predict_model.ipynb # command to test model for prediction
├── test_model.py
├── predict_with_model.py
├── model.pkl
├── requirements.txt
└── README.md
```

## 🧠 Project Intuition
### The trick behind creating your model is knowing the flow.
This is the flow to creating your logic:

- Import Libraries: 
   All tools must be loaded before use – pandas for data, NumPy for numbers, Matplotlib/Seaborn for plots, and scikit-learn for modeling.

- Load the Dataset: 
   Read your CSV file into a pandas DataFrame to work with it in table format.

- Explore & Understand the Data: 
   Use ```.head()```, ```.info()``` and ```.isnull()``` to get a feel for what you're working with.

- Preprocess the Data: 
   Handle missing values (if any)

- Encode categorical features using ```pd.get_dummies()```: 
   Separate features (X) and target (y)

- Split the Data: 
   Use ```train_test_split()``` to divide the data into a training set and a test set.

- Train the Model: 
   Create a ```LinearRegression()``` model and call ```.fit()``` on the training data.

- Make Predictions: 
   Use ```.predict()``` to estimate housing prices on the test set.

- Evaluate Performance: 
   Print metrics like: ```Mean Squared Error (MSE)```, ```R-squared Score (R²)

- Interpret Coefficients: 
   Print out how each feature impacts the predicted price.

- Visualize Results:
   A scatter plot shows how close predicted prices are to actual ones.


## 📦 How to Run
1. Clone the repo:
   ```bash
   git clone https://github.com/debb-major/house-prices-multi-linear-regression.git
   
2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. Open notebook and run:

   ```bash
   jupyter notebook notebook/house_price_multilinear_regression.ipynb

4. (Optional) Test the model from the command line:

   ```bash
   python test_model.py

5. (Optional) Predict with the saved model:

   ```bash
   python predict_with_model.py

## 📈 Sample Output
#### Mean Squared Error: 1.75e+12
#### R-squared Score: 0.65
A decent start — the model predicts 65% of the price variability.

## 📊 Visualization
A scatter plot comparing actual house prices vs predicted ones.

![Graph 1](https://github.com/user-attachments/assets/ed9d3954-84fb-414c-8526-2956466be8b3)



## 🧪 Model Testing & Reusability

This project includes scripts to validate and reuse the model outside the notebook:

- ✅ `test_model.py`: Trains and evaluates the model programmatically
- ✅ `predict_with_model.py`: Loads the saved model (`model.pkl`) and makes a prediction without retraining
- ✅ `model.pkl`: A serialized version of the trained model using `joblib`

This makes the project easier to reuse, test, and extend toward deployment.


## 💬 Final Thoughts
This was an exciting step forward from simple linear regression to working with multiple inputs. It's fascinating how a model can understand patterns from real-world factors like area and furnishings!

I got my dataset from here - https://github.com/aman9801/multiple-linear-regression-on-housing-dataset. Thank you, Aman9801.

Thanks for checking out the project! 🌟

Made with 💙 by Debb-major

## 📌 Note
This project is part of my learning journey into AI/ML.

More beginner-friendly projects coming soon!
---



