# 🎓 StudyTrack – AI Based Student Study Habit Recommender

StudyTrack is an AI-powered Streamlit dashboard that predicts student academic performance based on study habits and lifestyle factors.  
It uses Machine Learning models to provide **performance prediction, visual analytics, clustering, and personalized recommendations**.

---

## 🚀 Features

- 🔐 User Authentication (Login & Sign Up)
- 🧠 AI Model Training using Student Data
- 📊 Interactive Data Visualizations
- 🎯 Individual Student Performance Prediction
- 📂 Bulk Student Prediction & Recommendations
- 📥 Downloadable Prediction Results (CSV)

---

## 🛠️ Technologies Used

- **Python**
- **Streamlit** – Web dashboard
- **Pandas & NumPy** – Data processing
- **Scikit-learn**
  - RandomForestRegressor
  - K-Means Clustering
  - StandardScaler
- **Plotly** – Interactive charts
- **MySQL** – Database
- **bcrypt** – Secure password hashing

---

## 📂 Project Structure

├── app.py / dashboard.py # Main Streamlit application
├── auth.py # Authentication logic
├── db.py # Database connection
├── studytrack_model.pkl # Saved ML model (auto-generated)
├── requirements.txt # Dependencies
└── README.md # Project documentation

yaml
Copy code

---

## ⚙️ How to Run the Project (Local Setup)

### 1️⃣ Install Required Libraries
```bash
pip install -r requirements.txt
2️⃣ Setup MySQL Database
Create a database named:

sql
Copy code
CREATE DATABASE studytrack;
Create required tables (users table for authentication).

3️⃣ Configure Database Connection
Update database credentials in db.py:

python
Copy code
password="YOUR_DB_PASSWORD"
4️⃣ Run the Application
bash
Copy code
streamlit run app.py
The dashboard will open in your browser.

📊 Dataset Requirements
The dataset should include columns such as:

StudyHours

SleepHours

Attendance

Marks (Target column)

Optional columns:

Name

StudentID

Gender

Supported formats:

CSV

Excel (.xlsx)

📈 Machine Learning Details
Prediction Model: Random Forest Regressor

Evaluation Metric: R² Score

Clustering: K-Means (3 clusters)

High Performer

Average Performer

Low Performer

🔐 Security Note
For security reasons:

Database passwords are not included in the public repository

Users must configure database credentials locally

🎯 Use Cases
Student performance analysis

Academic mentoring & counseling

Educational data analytics

Internship 

👨‍💻 Author
Rahul Gunjal
Aspiring Software Developer
AI & Data Analytics Enthusiast 🚀 Run the project using Streamlit
