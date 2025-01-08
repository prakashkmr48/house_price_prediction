# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()

# Convert the dataset into a DataFrame
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['Price'] = housing.target

# Explore the dataset
print(df.head())

# Visualize the data
sns.pairplot(df, diag_kind='kde')
plt.show()

# Split data into features (X) and target (y)
X = df.drop('Price', axis=1)
y = df['Price']

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Display metrics
st.subheader("Model Evaluation")
st.write(f"Mean Squared Error: {mse}")
st.write(f"Root Mean Squared Error: {rmse}")
st.write(f"R² Score: {r2}")

# Visualize the actual vs predicted prices
st.subheader("Actual vs Predicted Prices")
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
st.pyplot(plt)
