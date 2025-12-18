import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#   LOAD CSV FILE

df = pd.read_csv("student_Data.csv")

print("===== RAW DATA =====")
print(df)

# ===========================
#   FIX COLUMN NAME SPACES
# ===========================
df.columns = df.columns.str.strip()      # remove extra spaces
df.columns = df.columns.str.replace(" ", "", regex=False)  # optional: remove internal spaces

print("\n===== CLEANED COLUMN NAMES =====")
print(df.columns.tolist())


# ===========================
#   CHECK MISSING VALUES
# ===========================
print("\n===== Missing Values =====")
print(df.isnull().sum())


# Fill numeric NaN with mean
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
for col in numeric_cols:
    df[col].fillna(df[col].mean(), inplace=True)

# Fill categorical NaN with mode
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)


# ===========================
#   NUMPY OPERATIONS
# ===========================
print("\n===== NUMPY OPERATIONS =====")
marks = df["Marks"].to_numpy()
study_hours = df["StudyHoursPerWeek"].to_numpy()

print("Mean Marks:", np.mean(marks))
print("Max Marks :", np.max(marks))
print("Min Marks :", np.min(marks))

print("Mean Study Hours:", np.mean(study_hours))


# ===========================
#   PANDAS OPERATIONS
# ===========================
print("\n===== PANDAS OPERATIONS =====")
print(df.describe())

print("\nNames and Marks Only:")
print(df[["Name", "Marks"]])


# ===========================
#   SEABORN VISUALIZATIONS
# ===========================
sns.set(style="whitegrid")

# Scatter Plot
plt.figure(figsize=(7,5))
sns.scatterplot(x=df["StudyHoursPerWeek"], y=df["Marks"], hue=df["Gender"])
plt.title("Study Hours vs Marks")
plt.xlabel("Study Hours Per Week")
plt.ylabel("Marks")
plt.show()

# Attendance Histogram
plt.figure(figsize=(7,5))
sns.histplot(df["AttendanceRate"], kde=True)
plt.title("Attendance Rate Distribution")
plt.show()


# ===========================
#   MATPLOTLIB BAR CHART
# ===========================
plt.figure(figsize=(8,5))
plt.bar(df["Name"], df["Marks"])
plt.xticks(rotation=45)
plt.title("Marks Comparison")
plt.xlabel("Students")
plt.ylabel("Marks")
plt.show()


print("\nALL OPERATIONS COMPLETED SUCCESSFULLY!")
