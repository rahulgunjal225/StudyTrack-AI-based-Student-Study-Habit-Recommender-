
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
import matplotlib.pyplot as plt


# 1) Load dataset

df = pd.read_excel("modeldata.xlsx")
df.columns = df.columns.str.strip()

print("===== RAW DATA =====")
print(df)


# 2) Outlier removal - Z-Score

z = np.abs(stats.zscore(df[['StudyHours', 'WorksHours', 'playHours', 'SleepHours', 'Marks']]))
df = df[(z < 3).all(axis=1)]

print("\n===== DATA AFTER REMOVING OUTLIERS =====")
print(df)


# 3) Feature Engineering

df["TotalEffort"] = df["StudyHours"] + df["WorksHours"] + df["playHours"]
df["Balance"] = df["SleepHours"] - df["playHours"]

print("\n===== DATA WITH NEW FEATURES =====")
print(df)


# 4) Select features and target

X = df[['StudyHours', 'WorksHours', 'playHours', 'SleepHours', 'TotalEffort', 'Balance']]
y = df['Marks']

# Scaling data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# 5) Train-test split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)


# 6) Train Linear Regression model

model = LinearRegression()
model.fit(X_train, y_train)


# 7) Predictions

pred = model.predict(X_test)


# 8) Model performance metrics

mse = mean_squared_error(y_test, pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, pred)

print("\n===== MODEL PERFORMANCE (Clean Format) =====")
print(f"MSE      : {mse:.3f}")
print(f"RMSE     : {rmse:.3f}")
print(f"R² Score : {r2:.3f}")


# 9) Predict marks for a new student

new_student = pd.DataFrame({
    'StudyHours': [3],
    'WorksHours': [2],
    'playHours': [1],
    'SleepHours': [7]
})

# Add engineered features
new_student["TotalEffort"] = new_student["StudyHours"] + new_student["WorksHours"] + new_student["playHours"]
new_student["Balance"] = new_student["SleepHours"] - new_student["playHours"]

new_scaled = scaler.transform(new_student)
predicted_score = model.predict(new_scaled)[0]

print("\nPredicted Score for New Student:", predicted_score)


# 10) Plot Actual vs Predicted

plt.figure(figsize=(7,5))
plt.scatter(y_test, pred, color="blue")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')

plt.title("Actual vs Predicted Marks (Regression Line)")
plt.xlabel("Actual Marks")
plt.ylabel("Predicted Marks")
plt.grid(True)
plt.show()
