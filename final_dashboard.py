import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle
import os
from pathlib import Path

np.random.seed(42)

st.set_page_config(page_title="Study Track AI Based Student Study Habit Recommender", layout="wide")

from auth import create_user, login_user, reset_password

# LOGIN SESSION
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "forgot" not in st.session_state:
    st.session_state.forgot = False

# RESPONSIVE LOGIN PAGE S
if not st.session_state.logged_in:
    
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* STREAMLIT RESET */
.main { padding: 0.5rem !important; }
.css-1d391kg { padding-top: 1rem !important; }

/* SMALL PREMIUM PURPLE CONTAINER */
.auth-container {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 2.5rem 1rem;
    background: radial-gradient(
        circle at center,
        rgba(139,92,246,0.22),
        transparent 70%
    );
    max-width: none;
    margin: 0;
}

.login-box {
    position: relative;
    background: white !important;
    border-radius: 22px !important;
    padding: 2rem 2rem 2.2rem !important;
    max-width: 420px;
    box-shadow: 0 18px 40px rgba(0,0,0,0.12);
}

/* TOP GRADIENT LINE */
.login-box::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    border-radius: 22px 22px 0 0;
    background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
}

/* LOGO */
.logo-title { text-align: center; margin-bottom: 1.3rem; }
.logo {
    font-size: 2.4rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #667eea, #764ba2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.subtitle { color: #6b7280; font-size: 0.95rem; }

/* FORM TITLE */
.form-title {
    text-align: center;
    font-size: 1.25rem;
    font-weight: 700;
    margin-bottom: 1.2rem;
    color: #1f2937;
}

/* INPUTS */
.stTextInput input {
    border-radius: 14px !important;
    height: 52px !important;
    background: #f9fafb !important;
    border: 2px solid #e5e7eb !important;
}
.stTextInput label {
    font-size: 1.3rem !important;
    font-weight: 800 !important;
    color: #111827 !important;
}

.stTextInput input:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 3px rgba(102,126,234,0.15) !important;
}

/* LOGIN / SIGNUP BUTTON LOOK */
.stButton > button {
    height: 52px !important;
    border-radius: 14px !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    box-shadow: 0 8px 20px rgba(0,0,0,0.15) !important;
}

/* LOGIN / SIGNUP TAB DEFAULT (WHITE) */
#login_tab button,
#signup_tab button {
    background: #ffffff !important;
    color: #111827 !important;
    border: 2px solid #e5e7eb !important;
}

/* ACTIVE TAB */
body:has(#login_tab:focus) #login_tab button,
body:has(#login_tab:active) #login_tab button,
body:has(#signup_tab:focus) #signup_tab button,
body:has(#signup_tab:active) #signup_tab button {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important;
}

/* PRIMARY BUTTON – ENTER DASHBOARD */
button[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important;
    border-radius: 14px !important;
    height: 52px !important;
    font-weight: 700 !important;
    position: relative;
    transition: 0.3s ease;
}

button[data-testid="baseButton-primary"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 30px rgba(102,126,234,0.5);
}

button[data-testid="baseButton-primary"]::after {
    content: "→";
    position: absolute;
    right: 18px;
    opacity: 0;
    transition: 0.3s ease;
}

button[data-testid="baseButton-primary"]:hover::after {
    opacity: 1;
    right: 14px;
}

/* MOBILE */
@media (max-width: 600px) {
    .auth-container { margin: 1.5rem; padding: 2rem 1rem; }
    .logo { font-size: 2.1rem !important; }
}
.stTextInput { margin-bottom: 0.7rem !important; }
.form-title { margin-bottom: 0.8rem !important; }
</style>
""", unsafe_allow_html=True)

    # MAIN LOGIN LAYOUT
    st.markdown("""
    <div class="auth-container">
        <div class="login-box">
            <div class="logo-title">
                <h1 class="logo">🎓 StudyTrack</h1>
                <p class="subtitle">AI-Powered Student Analytics</p>
            </div>
    """, unsafe_allow_html=True)

    # LOGIN / SIGN UP BUTTONS
    st.markdown('<div style="display:flex; gap:12px; margin-bottom:20px;">', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        if st.button("🔐 Login", use_container_width=True, key="login_tab"):
            st.session_state.auth_mode = "login"

    with c2:
        if st.button("📝 Sign Up", use_container_width=True, key="signup_tab"):
            st.session_state.auth_mode = "signup"
    st.markdown("</div>", unsafe_allow_html=True)

    # DEFAULT MODE
    if "auth_mode" not in st.session_state:
        st.session_state.auth_mode = "login"

    if st.session_state.auth_mode == "login":
        st.markdown('<div class="form-title">Welcome Back</div>', unsafe_allow_html=True)
        
        st.markdown('<div style="font-size:1.35rem;font-weight:800;color:#111827;margin-bottom:6px;">👤 Username</div>', unsafe_allow_html=True)
        username = st.text_input("", key="login_username", placeholder="Enter username")
        
        st.markdown('<div style="font-size:1.35rem;font-weight:800;color:#111827;margin-bottom:6px;">🔒 Password</div>', unsafe_allow_html=True)
        password = st.text_input("", type="password", key="login_password", placeholder="Enter password")
        
        col1, col2 = st.columns([3,1])
        with col1:
            if st.button("🚀 **Enter Dashboard**", key="login_btn", type="primary"):
                if username and password:
                    user = login_user(username, password)
                    if user:
                        st.session_state.logged_in = True
                        st.session_state.user_id = user["id"]
                        st.success("✅ Dashboard loading...")
                        st.rerun()
                    else:
                        st.error("❌ Invalid credentials!")
                else:
                    st.warning("⚠️ Fill all fields")

    elif st.session_state.auth_mode == "signup":
        st.markdown('<div class="form-title">Create Account</div>', unsafe_allow_html=True)
        
        st.markdown('<div style="font-size:1.35rem;font-weight:800;color:#111827;margin-bottom:6px;">👤 Username</div>', unsafe_allow_html=True)
        username = st.text_input("", key="signup_username", placeholder="Enter username")
        
        st.markdown('<div style="font-size:1.35rem;font-weight:800;color:#111827;margin-bottom:6px;">📧 Email</div>', unsafe_allow_html=True)
        email = st.text_input("", key="signup_email", placeholder="Enter email")
        
        st.markdown('<div style="font-size:1.35rem;font-weight:800;color:#111827;margin-bottom:6px;">🔒 Password</div>', unsafe_allow_html=True)
        password = st.text_input("", type="password", key="signup_password", placeholder="Enter password")
        
        if st.button("✅ **Create Account**", key="signup_btn"):
            if all([username, email, password]) and len(password) >= 6:
                create_user(username, email, password)
                st.success("🎉 Account created! Login now.")
            else:
                st.error("⚠️ Complete all fields (6+ char password)")

    st.markdown("</div></div>", unsafe_allow_html=True)
    st.stop()

# SIDEBAR STYLE 
st.markdown("""
<style>
/* SIDEBAR BACKGROUND */
[data-testid="stSidebar"] {
    background-color: #f8fafc;
    padding-top: 20px;
}

/* SIDEBAR TITLE */
.sidebar-title {
    font-size: 1.4rem;
    font-weight: 900;
    color: #1e3a8a;
    margin-bottom: 12px;
}

/* NAV ITEMS (RADIO LABELS) */
[data-testid="stSidebar"] label {
    font-size: 1.15rem !important;
    font-weight: 800 !important;
    color: #111827 !important;
    padding: 8px 6px !important;
}

/* SELECTED ITEM */
[data-testid="stSidebar"] input:checked + div {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white !important;
    border-radius: 10px;
    padding: 6px;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown(
    "<div class='sidebar-title'>📌 Dashboard</div>",
    unsafe_allow_html=True
)

page = st.sidebar.radio(
    "",
    [
        "🏠 Home",
        "🧠 Model Training",
        "📊 Data Visuals",
        "🎯 Student Recommendation"
    ]
)

st.sidebar.markdown("---")

if st.sidebar.button("🚪 Logout"):
    st.session_state.logged_in = False
    st.session_state.pop("user_id", None)
    st.rerun()

# HOME PAGE
if page == "🏠 Home":
    page_bg = """
    <style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #E3F2FD 10%, #FFF8E1 90%);
        color: #0d47a1;
    }
    [data-testid="stSidebar"] { background: #f5f5f5; }
    h1, h2, h3 { color: #0d47a1 !important; }
    </style>
    """
    st.markdown(page_bg, unsafe_allow_html=True)

    st.markdown("""
    <style>
    .hero-container {
        background: linear-gradient(90deg, #e3f2fd, #fffde7);
        border-radius: 18px;
        padding: 55px 40px;
        margin: 25px 0 35px 0;
        text-align: center;
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 900;
        color: #1f2937;
        margin-bottom: 10px;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #4b5563;
        font-weight: 500;
    }
    </style>

    <div class="hero-container">
        <div class="hero-title">🚀 AI Based Student Study Habit Recommender</div>
        <div class="hero-subtitle">
            Empowering Students through Predictive Analytics
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=120)
        st.subheader("🎯 Smart Performance Prediction")
        st.write("Advanced Machine Learning models analyze study habits to accurately predict academic performance and learning outcomes.")


    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/4140/4140037.png", width=120)
        st.subheader("🧠 Intelligent Behavior Analysis")
        st.write("AI identifies hidden patterns from study hours, sleep cycles, attendance, and lifestyle habits to understand student behavior deeply.")


    with col3:
        st.image("https://cdn-icons-png.flaticon.com/512/4140/4140048.png", width=120)
        st.subheader("📈 Personalized Growth Strategy")
        st.write("Students receive data-driven recommendations tailored to their strengths and weaknesses for continuous academic improvement.")

    st.divider()
    st.info("**Tip:** Go to 'Model Training' to upload your batch data and train the model.")

# 📂 MODEL TRAINING PAGE
elif page == "🧠 Model Training":
    st.header("📂 Train the AI Model")

    model_path = "studytrack_model.pkl"
    
    file_type = st.radio("Select Data Source:", ["CSV", "Excel"])
    uploaded_file = None

    if file_type == "CSV":
        uploaded_file = st.file_uploader("Upload Student Performance CSV", type=["csv"])
    elif file_type == "Excel":
        uploaded_file = st.file_uploader("Upload Student Performance Excel", type=["xlsx"])

    if uploaded_file is None and "model" not in st.session_state and os.path.exists(model_path):
        with open(model_path, "rb") as f:
            st.session_state["model"] = pickle.load(f)
        st.info("📂 Loaded saved model!")

    if uploaded_file is not None:
        if file_type == "CSV":
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)

        data.columns = data.columns.str.strip()
        data = data.loc[:, ~data.columns.str.contains("^Unnamed")]

        drop_cols = ["StudentID",  "Gender"]
        data = data.drop(columns=[c for c in drop_cols if c in data.columns])

        if "Online Classes Taken" in data.columns:
            data["Online Classes Taken"] = data["Online Classes Taken"].map(
                {"TRUE": 1, "FALSE": 0, True: 1, False: 0}
            )

        data = data.fillna(0)

        st.success("✅ Data Loaded Successfully!")
        st.dataframe(data.head(10))

        data_numeric = data.copy()
        
        st.success(f"✅ Data Cleaned! Using {len(data_numeric.columns)} columns, {len(data_numeric)} rows")
        st.dataframe(data_numeric.head())

        if len(data_numeric) < 5:
            st.error("❌ Need at least 5 rows of data!")
            st.stop()

        if "Marks" not in data_numeric.columns:
            st.error("❌ 'Marks' column missing in dataset!")
            st.stop()

        target_col = "Marks"
        
        X = data_numeric.drop(columns=[target_col, "Name"], errors="ignore")
        y = data_numeric[target_col]

        q1 = y.quantile(0.10)
        q3 = y.quantile(0.90)
        mask = (y >= q1) & (y <= q3)
        X = X[mask]
        y = y[mask]

        st.info(f"📊 Features: {list(X.columns)}")
        st.info(f"🎯 Target (Marks): {target_col}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(
                n_estimators=500,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features="sqrt",
                random_state=42
            ))
        ])

        pipeline.fit(X_train, y_train)

        importances = pipeline.named_steps["model"].feature_importances_
        imp_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)

        important_features = imp_df[imp_df["Importance"] > 0.05]["Feature"].tolist()
        
        if len(important_features) < 2:
            st.warning("⚠️ Too few strong features, using all features.")
            X_selected = X.copy()
        else:
            X_selected = X[important_features]

        st.subheader("🔍 Feature Importance (Model Explanation)")
        st.dataframe(imp_df.head(10), use_container_width=True)
        
        fig_imp = px.bar(imp_df.head(8), x="Importance", y="Feature", 
                        orientation='h', title="Top 8 Most Important Features",
                        color="Importance", color_continuous_scale="Viridis")
        st.plotly_chart(fig_imp, use_container_width=True)

        st.info("🔄 Retraining model with top features only...")
        X_train_sel, X_test_sel, y_train_sel, y_test_sel = train_test_split(
            X_selected, y, test_size=0.2, random_state=42 
        )
        pipeline.fit(X_train_sel, y_train_sel)

        y_pred = pipeline.predict(X_test_sel)
        score = r2_score(y_test_sel, y_pred)

        st.success("🎯 Test Split Performance")
        st.metric("R² Score", f"{score:.3f}")

        result_df = pd.DataFrame({
            "Actual Marks": y_test_sel.values,
            "Predicted Marks": np.round(y_pred, 2)
        }).reset_index(drop=True)

        st.write("### 📊 Actual vs Predicted Marks")
        st.dataframe(result_df)
        st.line_chart(result_df, use_container_width=True)

        st.session_state["model"] = pipeline
        st.session_state["data"] = X_selected.copy()
        st.session_state["original_data"] = data_numeric 
        st.session_state["features"] = list(X_selected.columns)
        st.session_state["target"] = target_col

        if score > 0.6:  
            with open(model_path, "wb") as f:
                pickle.dump(pipeline, f)
            st.success("💾 Model saved for future use!")
        else:
            st.info(f"📈 Good model for small dataset! (R²: {score:.3f})")

# 📊 DATA VIsuals PAGE
elif page == "📊 Data Visuals":
    st.header("📊 Data Visuals Dashboard")

    if "data" not in st.session_state:
        st.warning("⚠️ Please train the model first in the 'Model Training' section.")
        st.stop()

    target_col = st.session_state.get("target")
    df = st.session_state["original_data"].copy()
    study_col = "StudyHours" if "StudyHours" in df.columns else None

    st.success("✅ Data Loaded Successfully for Visuals!")
    st.subheader("👀 Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("🔥 Correlation Heatmap between Variables")
    num_df = df.select_dtypes(include=['int64', 'float64'])
    if not num_df.empty and target_col in num_df.columns:
        corr = num_df.corr()
        fig_heatmap = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                               title="Correlation Heatmap (Closer to 1 or -1 means strong relation)")
        st.plotly_chart(fig_heatmap, use_container_width=True)

    st.divider()
    st.subheader("🎯 Study Hours vs Marks")
    if study_col and target_col in df.columns:
        fig = px.scatter(
            df,
            x="StudyHours",
            y="Marks",
            title="Study Hours vs Marks"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    
   
    # 🔴 STEP 1: MARKS DISTRIBUTION
    st.divider()
    st.subheader("📈 Marks Distribution (Overall Performance Spread)")
    fig_dist = px.histogram(
        df,
        x="Marks",
        nbins=10,
        title="Distribution of Student Marks",
        color_discrete_sequence=["#636EFA"]
    )
    fig_dist.update_layout(
        xaxis_title="Marks",
        yaxis_title="Number of Students",
        bargap=0.15
    )

    st.plotly_chart(fig_dist, use_container_width=True)

    # 🔴 STEP 2: STUDENT CLUSTERING
    st.divider()
    st.subheader("🧩 Student Performance Clustering (K-Means)")
    # Using only important columns
    if "StudyHours" in df.columns:
        cluster_df = df[["StudyHours", "Marks"]].copy()

        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        cluster_df["Cluster"] = kmeans.fit_predict(cluster_df)

        fig_cluster = px.scatter(
            cluster_df,
            x="StudyHours",
            y="Marks",
            color=cluster_df["Cluster"].astype(str),
            title="Student Segmentation based on Study Hours & Marks",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_cluster.update_layout(
            xaxis_title="Study Hours",
            yaxis_title="Marks"
        )
        st.plotly_chart(fig_cluster, use_container_width=True)
    st.divider()
    st.subheader("💡 Key Insights Summary")

    avg_study = df["StudyHours"].mean() if "StudyHours" in df.columns else 0
    avg_marks = df["Marks"].mean()

    avg_sleep = df["SleepHours"].mean() if "SleepHours" in df.columns else None
    avg_attendance = df["Attendance"].mean() if "Attendance" in df.columns else None

    if "Name" in df.columns:
        top_row = df.loc[df["Marks"].idxmax()]
        top_student = top_row["Name"]
    else:
        top_student = f"Student #{df['Marks'].idxmax() + 1}"

    top_marks = df["Marks"].max()

    st.markdown(f"""
    - 🧠 **Average Study Hours:** {avg_study:.2f} hrs/day  
    - 🎯 **Average Marks:** {avg_marks:.2f}%  
    - 👑 **Top Performer:** {top_student} ({top_marks}%)  
    """)

    if avg_sleep is not None:
        st.markdown(f"- 😴 **Average Sleep Hours:** {avg_sleep:.2f} hrs/day")
    if avg_attendance is not None:
        st.markdown(f"- 🏫 **Average Attendance:** {avg_attendance:.1f}%")
# 🎯 STUDENT RECOMMENDATION PAGE
elif page == "🎯 Student Recommendation":
    st.header("🎯 Student Recommendation System")

    if "model" not in st.session_state:
        st.warning("⚠️ Please train the model first.")
        st.stop()

    model = st.session_state["model"]
    features = [f for f in st.session_state["features"] if f != st.session_state["target"] and "id" not in f.lower()]

    st.subheader("🧍 Individual Student Prediction")
    student_name = st.text_input("👤 Student Name")

    col1, col2 = st.columns(2)
    with col1: sleep = st.number_input("😴 Sleep Hours", 0.0, 12.0, 6.0)
    with col2: exercise = st.number_input("🏃 Exercise Hours", 0.0, 5.0, 1.0)

    col3, col4 = st.columns(2)
    with col3: study = st.number_input("📘 Study Hours", 0.0, 12.0, 4.0)
    with col4: social = st.number_input("📱 Social Media Hours", 0.0, 6.0, 2.0)

    play = st.number_input("🎮 Play Hours", 0.0, 6.0, 1.0)

    input_data = {}
    for f in features:
        if "study" in f.lower():
            input_data[f] = study
        elif "sleep" in f.lower():
            input_data[f] = sleep
        elif "play" in f.lower():
            input_data[f] = play
        elif "social" in f.lower(): 
            input_data[f] = social
        elif "attendance" in f.lower():
            input_data[f] = 80
        else:
            input_data[f] = 0

    if st.button("🎯 Predict Individual Marks", use_container_width=True):
        input_df = pd.DataFrame([input_data])
        input_df = input_df[features] 
        predicted = model.predict(input_df)[0]
        predicted = min(max(predicted, 0), 100)
        trees = model.named_steps["model"].estimators_
        if len(trees) > 0:
            all_tree_preds = np.stack([tree.predict(input_df) for tree in trees])
            confidence = np.std(all_tree_preds)
        else:
            confidence = 0.0


        st.metric("📊 Predicted Score", f"{predicted:.1f}/100")
        st.metric("📉 Prediction Confidence", f"±{confidence:.2f}")

        if predicted >= 85:
            st.success("🏆 Excellent performance expected!")
        elif predicted >= 70:
            st.info("👍 Good performance. Minor improvements needed.")
        else:
            st.warning("📘 Needs improvement. Increase focus & study time.")

    st.divider()
    st.subheader("📂 Bulk Upload – Predict & Cluster Multiple Students")
    bulk_file = st.file_uploader("Upload CSV / Excel", type=["csv", "xlsx"])

    if bulk_file:
        if bulk_file.name.endswith(".csv"):
            bulk_df = pd.read_csv(bulk_file)
        else:
            bulk_df = pd.read_excel(bulk_file)
        
        bulk_df.columns = bulk_df.columns.str.strip()
        bulk_df = bulk_df.loc[:, ~bulk_df.columns.str.contains("^Unnamed")]
        
        if "Online Classes Taken" in bulk_df.columns:
            bulk_df["Online Classes Taken"] = bulk_df["Online Classes Taken"].map(
                {"TRUE": 1, "FALSE": 0, True: 1, False: 0}
            )
        
        st.dataframe(bulk_df.head())
        meta_cols = [c for c in ["StudentID", "Name", "Student"] if c in bulk_df.columns]
        bulk_df = bulk_df.dropna(how="all")
        meta_df = bulk_df[meta_cols].copy() if meta_cols else pd.DataFrame()    
        for col in features:
            if col not in bulk_df.columns:
                bulk_df[col] = 0
        
        ml_df = bulk_df[features].fillna(0)
        st.success(f"✅ Validated {len(ml_df)} students")
        preds = model.predict(ml_df)
        
        result_df = pd.concat([meta_df.reset_index(drop=True), ml_df.reset_index(drop=True)], axis=1)
        result_df["Predicted_Marks"] = np.round(preds, 2)
        cluster_features = result_df[["Predicted_Marks"]].copy()
        if "StudyHours" in result_df.columns:
            cluster_features["StudyHours"] = result_df["StudyHours"]
        if "SleepHours" in result_df.columns:
            cluster_features["SleepHours"] = result_df["SleepHours"]
        
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        result_df["PerformanceCluster"] = kmeans.fit_predict(cluster_features)
        def get_performance_level(mark):
            if mark >= 85: return "High Performer"
            elif mark >= 70: return "Average Performer"
            else: return "Low Performer"
        result_df["PerformanceLevel"] = result_df["Predicted_Marks"].apply(get_performance_level)
        def get_recommendation(mark):
            if mark >= 85: return " Excellent! Maintain consistency & advanced revision."
            elif mark >= 70: return " Good! Increase focus & reduce distractions."
            elif mark >= 50: return " Average. Improve study hours & sleep cycle."
            else: return " Critical. Immediate mentoring required."
        result_df["Recommendation"] = result_df["Predicted_Marks"].apply(get_recommendation)
        st.success("✅ Bulk Prediction Completed")
        st.dataframe(result_df)
        st.subheader("📈 Student Performance Visualization")
        chart_df = result_df.copy()
    
        if "Student" not in chart_df.columns and "Name" not in chart_df.columns and "StudentID" not in chart_df.columns:
            chart_df["Student"] = ["Student " + str(i+1) for i in range(len(chart_df))]
        student_col = next((c for c in ["Student", "Name", "StudentID"] if c in chart_df.columns), chart_df.index.name or "Student")
        fig = px.scatter(
            chart_df,
            x=student_col,
            y="Predicted_Marks",
            size="Predicted_Marks",
            color="PerformanceLevel",
            hover_data={"Predicted_Marks": True, "PerformanceCluster": True, "Recommendation": True},
            title="Student-wise Predicted Marks & Performance Analysis",
            size_max=35,
            color_discrete_map={"Low Performer": "#EF5350", "Average Performer": "#FFA726", "High Performer": "#66BB6A"}
        )
        fig.update_traces(marker=dict(opacity=0.85, line=dict(width=1, color="black")))
        fig.update_layout(xaxis_title="Students", yaxis_title="Predicted Marks", yaxis_range=[0, 100], template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("📊 Performance Cluster Distribution")
        cluster_fig = px.bar(
            chart_df,
            x="PerformanceLevel",
            color="PerformanceLevel",
            title="Student Distribution by Performance Level",
            text_auto=True,
            color_discrete_map={"Low Performer": "#EF5350", "Average Performer": "#FFA726", "High Performer": "#66BB6A"}
        )
        cluster_fig.update_layout(
            xaxis={'categoryorder':'array', 'categoryarray':['Low Performer','Average Performer','High Performer']}
        )
        st.plotly_chart(cluster_fig, use_container_width=True)
        preferred_order = (
            meta_cols +
            features +
            ["Predicted_Marks", "PerformanceLevel", "Recommendation"]
        )
        preferred_order = [c for c in preferred_order if c in result_df.columns]
        result_df = result_df[preferred_order]
        st.download_button("📥 Download Results", result_df.to_csv(index=False).encode("utf-8"), "bulk_predictions.csv", "text/csv")

# 📜 FOOTER
st.markdown("---")
st.markdown("<center>© StudyTrack AI based Student Study Habit Recommender</center>", unsafe_allow_html=True)
