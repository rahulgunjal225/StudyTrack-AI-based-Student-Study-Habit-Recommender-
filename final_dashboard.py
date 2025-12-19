import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import sqlite3
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# PAGE CONFIG

st.set_page_config(page_title="StudyTrack AI based Student Study Habit Recommender", layout="wide")

st.title("🎓 StudyTrack AI based Student Study Habit Recommender")
st.caption("AI-Driven Analysis • Clustering • Predictive Insights")


# CUSTOM CSS FOR BOLD SIDEBAR

st.markdown("""
<style>
    .sidebar .sidebar-content .stRadio > label {
        font-weight: 900 !important;
        font-size: 1.2rem !important;
        color: #1f77b4 !important;
    }
    .sidebar-header {
        font-weight: 900 !important;
        font-size: 1.5rem !important;
        color: #1f77b4 !important;
    }
</style>
""", unsafe_allow_html=True)


# SIDEBAR - BOLD NAVIGATION ✅

st.sidebar.markdown("### 📂 **Dashboard**")

uploaded_file = st.sidebar.file_uploader(
    "Upload Dataset (CSV / Excel / JSON / SQLite)",
    type=["csv", "xlsx", "json", "db", "sqlite"]
)

#  BOLD NAVIGATION BAR EXACTLY AS REQUESTED
page = st.sidebar.radio(
    "Navigate",
    ["🏠 **Overview**", "📄 **Dataset**", "📊 **Visual Analysis**", "👥 **Clustering**", "🎯 **Prediction**"]
)


# LOAD DATA

if uploaded_file is None:
    st.info("👈 Please upload a dataset to continue")
    st.stop()

fname = uploaded_file.name.lower()

if fname.endswith(".csv"):
    data = pd.read_csv(uploaded_file)
elif fname.endswith(".xlsx"):
    data = pd.read_excel(uploaded_file)
elif fname.endswith(".json"):
    data = pd.read_json(uploaded_file)
elif fname.endswith(".db") or fname.endswith(".sqlite"):
    with open("temp.db", "wb") as f:
        f.write(uploaded_file.getbuffer())
    conn = sqlite3.connect("temp.db")
    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)
    table = st.sidebar.selectbox("Select Table", tables["name"])
    data = pd.read_sql(f"SELECT * FROM {table}", conn)
    conn.close()
else:
    st.error("Unsupported file format")
    st.stop()


# COLUMN STANDARDIZATION

data.columns = data.columns.str.strip().str.lower().str.replace(" ", "")

rename_map = {
    "studyhours": "StudyHours",
    "workhours": "WorksHours",
    "workshours": "WorksHours",
    "playhours": "playHours",
    "sleephours": "SleepHours",
    "marks": "Marks",
    "finalgrade": "Marks"
}
data.rename(columns=rename_map, inplace=True)

required_cols = ["StudyHours", "WorksHours", "playHours", "SleepHours", "Marks"]
missing = [c for c in required_cols if c not in data.columns]

if missing:
    st.error(f"❌ Missing required columns: {missing}")
    st.write("Available columns:", list(data.columns))
    st.stop()


# DERIVED COLUMN : STUDY LEVEL

def study_level(hours):
    if hours <= 2:
        return "Low"
    elif hours <= 5:
        return "Medium"
    else:
        return "High"

data["StudyLevel"] = data["StudyHours"].apply(study_level)


# OVERVIEW PAGE

if page == "🏠 **Overview**":
    st.markdown("""
    <style>
    .hero-box {
        background: linear-gradient(120deg, #e3f2fd, #fffde7);
        padding: 40px;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 30px;
    }
    .feature-card {
        background: white;
        padding: 25px;
        border-radius: 14px;
        box-shadow: 0 6px 15px rgba(0,0,0,0.1);
        text-align: center;
        transition: transform 0.3s ease;
    }
    .feature-card:hover {
        transform: translateY(-5px);
    }
    </style>
    <div class="hero-box">
        <h1>🚀 AI Based Student Study Habit Recommender</h1>
        <h4 style="color:#555;">
            Empowering Students through Predictive Analytics
        </h4>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="feature-card">
            <h3>🎯 Precision</h3>
            <p>Predicts student marks using Machine Learning models.</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="feature-card">
            <h3>🧠 Intelligence</h3>
            <p>Analyzes study habits, sleep, play and work patterns.</p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="feature-card">
            <h3>📈 Growth</h3>
            <p>Provides insights to improve academic performance.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Students", len(data))
    m2.metric("Average Marks", round(data["Marks"].mean(), 2))
    m3.metric("Highest Marks", int(data["Marks"].max()))
    m4.metric("High Study Level", len(data[data["StudyLevel"]=="High"]))

    st.markdown("---")
    
    st.subheader("🚀 Dashboard Modules")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown("### 📊 Visual Analysis"); st.caption("Charts & Insights")
    with c2: st.markdown("### 👥 Student Clustering"); st.caption("K-Means Algorithm")
    with c3: st.markdown("### 🎯 Marks Prediction"); st.caption("Linear Regression")
    with c4: st.markdown("### ⬇️ Download Report"); st.caption("CSV Export")


# DATASET PAGE

elif page == "📄 **Dataset**":
    st.header("📄 Dataset Preview")
    st.dataframe(data, use_container_width=True)
    with st.expander("📈 Statistics"):
        st.write(data.describe())


# VISUAL ANALYSIS PAGE

elif page == "📊 **Visual Analysis**":
    st.header("📊 Visual Analysis")

    st.subheader("📌 Study Hours vs Marks (Colored by Study Level)")
    fig1 = px.scatter(
        data, x="StudyHours", y="Marks", color="StudyLevel", size="Marks",
        hover_data=["Marks", "StudyLevel"],
        title="Study Hours vs Marks (Colored by Study Level)",
        color_discrete_map={"Low": "#d62728", "Medium": "#1f77b4", "High": "#2ca02c"}
    )
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("📊 Average Marks by Study Hours")
    avg_df = data.groupby("StudyHours")["Marks"].mean().reset_index()
    fig2 = px.bar(avg_df, x="StudyHours", y="Marks", title="Average Marks by Study Hours", color="Marks")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("🔥 Correlation Heatmap between Variables")
    st.caption("Closer to +1 or -1 indicates strong relationship")
    fig_corr, ax_corr = plt.subplots(figsize=(6,5))
    sns.heatmap(data[required_cols].corr(), annot=True, cmap="coolwarm", linewidths=0.5, fmt=".2f", ax=ax_corr)
    st.pyplot(fig_corr)

    st.markdown("""
    **What it shows:**
    - **Red** → strong positive relation
    - **Blue** → negative relation  
    - **Near 0** → weak/no relation
    """)

    st.subheader("💡 Key Insights Summary")
    avg_study = round(data["StudyHours"].mean(), 2)
    avg_marks = round(data["Marks"].mean(), 2)
    top_student_marks = int(data["Marks"].max())
    high_study_count = len(data[data["StudyLevel"] == "High"])
    
    corr_matrix = data[required_cols].corr()
    study_marks_corr = corr_matrix.loc["StudyHours", "Marks"]
    sleep_marks_corr = corr_matrix.loc["SleepHours", "Marks"]
    
    st.markdown(f"""
    - 📘 **Average Study Hours:** {avg_study} hrs/day  
    - 🧮 **Average Marks:** {avg_marks}  
    - 🏆 **Top Score Achieved:** {top_student_marks}  
    - 📈 **High Study Level Students:** {high_study_count}  

    **Observations:**
    - ✅ Strong positive correlation between Study Hours and Marks ({study_marks_corr:.2f}).
    - ⚠️ Sleep Hours show {abs(sleep_marks_corr):.2f} impact on Marks.
    - ❌ Play Hours and Work Hours have weak or negative influence on Marks.
    """)


# CLUSTERING PAGE

elif page == "👥 **Clustering**":
    st.header("👥 Student Clustering (K-Means)")

    scaler = StandardScaler()
    scaled = scaler.fit_transform(data[required_cols])

    kmeans = KMeans(n_clusters=3, random_state=42)
    data["Cluster"] = kmeans.fit_predict(scaled)

    data["Performance"] = data["Cluster"].map({
        0: "Low Performer",
        1: "Average Performer",
        2: "High Performer"
    })

    fig, ax = plt.subplots()
    sns.scatterplot(data=data, x="StudyHours", y="Marks", hue="Performance", palette="Set2", ax=ax)
    st.pyplot(fig)

    st.dataframe(data[["StudyHours", "Marks", "Cluster", "Performance", "StudyLevel"]], use_container_width=True)


# PREDICTION PAGE

elif page == "🎯 **Prediction**":
    st.header("🎯 Marks Prediction (Linear Regression)")

    c1, c2 = st.columns(2)
    with c1:
        study = st.slider("Study Hours", 0, 10, 4)
        work = st.slider("Work Hours", 0, 6, 2)
    with c2:
        play = st.slider("Play Hours", 0, 6, 1)
        sleep = st.slider("Sleep Hours", 4, 10, 7)

    X = data[["StudyHours", "WorksHours", "playHours", "SleepHours"]]
    y = data["Marks"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(X_scaled, y)

    data["Predicted_Marks"] = model.predict(X_scaled)
    r2 = r2_score(y, data["Predicted_Marks"])

    st.success(f"✅ Model trained | R² Score: {round(r2*100,2)}%")

    pred = model.predict(scaler.transform([[study, work, play, sleep]]))[0]
    st.success(f"📌 Predicted Marks: {pred:.2f}")

    if pred >= 80:
        st.info("Excellent performance expected.")
    elif pred >= 60:
        st.warning("Average performance. Needs improvement.")
    else:
        st.error("Low performance. Focus required.")

    fig, ax = plt.subplots()
    ax.scatter(y, data["Predicted_Marks"])
    ax.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
    ax.set_xlabel("Actual Marks")
    ax.set_ylabel("Predicted Marks")
    st.pyplot(fig)

    st.markdown("---")
    st.header("⬇️ Download Full Analysis Report")

    report = data.copy()
    report["R2_Score"] = round(r2*100, 2)

    csv = report.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📥 Download Student Analytics Report",
        csv,
        "Student_Analytics_Report.csv",
        "text/csv"
    )


# FOOTER

st.markdown("---")
st.caption("AI-Based Student Study Habit Analysis and Recommendation System")
