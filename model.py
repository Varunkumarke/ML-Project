import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# Load data
data = pd.read_csv(r"C:\Users\USER\Downloads\houseprice (1).csv")
print(data.head())  # Checking the first few rows

# Splitting into features and target
X = data.iloc[:, :-1]  # Features
y = data.iloc[:, -1]   # Target

# Handling missing values (if any)
imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate model performance
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")

# Save the model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save the scaler too (for future use)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
