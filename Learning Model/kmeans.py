
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt


# 1) Load Excel File

df = pd.read_excel("modeldata.xlsx")
df.columns = df.columns.str.strip()

print("\n===== RAW DATA =====")
print(df)

# 2) Outlier Removal (Z-score Method)

print("\nRemoving Outliers using Z-score...")

z = np.abs(stats.zscore(df[['StudyHours','WorksHours','playHours','SleepHours','Marks']]))
df = df[(z < 3).all(axis=1)]

print("\n===== DATA AFTER REMOVING OUTLIERS =====")
print(df)


# 3) Feature Selection

X = df[['StudyHours', 'WorksHours', 'playHours', 'SleepHours', 'Marks']]


# 4) Scaling the Data

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# 5) Elbow Method to Determine Best K

sse = []
K_range = range(1, 10)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    sse.append(km.inertia_)

plt.figure(figsize=(8,6))
plt.plot(K_range, sse, marker='o', linewidth=2)

plt.title("Optimal Number of Clusters (Elbow Method)", fontsize=16, fontweight='bold')
plt.xlabel("Number of Clusters (K)", fontsize=12)
plt.ylabel("SSE (Sum of Squared Errors)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)

plt.show()

print("\nElbow Method SSE Values:", sse)


# 6) Train Final K-Means Model (K = 3)

kmeans = KMeans(n_clusters=3, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_scaled)

print("\n===== CLUSTERED DATA =====")
print(df)


# 7) Silhouette Score (Cluster Quality)

sil_score = silhouette_score(X_scaled, df['Cluster'])
print("\nSilhouette Score:", sil_score)


# 8) Cluster Centroids (Original Scale)

centroids = scaler.inverse_transform(kmeans.cluster_centers_)
centroid_df = pd.DataFrame(centroids, columns=X.columns)

print("\n===== CLUSTER CENTROIDS (Original Scale) =====")
print(centroid_df)


# 9) Automatic Recommendation System

cluster_avg_marks = df.groupby("Cluster")['Marks'].mean()
print("\nCluster-wise Average Marks:")
print(cluster_avg_marks)

def recommend(cluster):
    if cluster_avg_marks[cluster] < 65:
        return " Needs More Effort"
    elif cluster_avg_marks[cluster] < 80:
        return " Average Student"
    else:
        return " Excellent Student"

df['Recommendation'] = df['Cluster'].apply(recommend)

print("\n===== FINAL DATA WITH RECOMMENDATION =====")
print(df)


# 10) Save Final Output

df.to_excel("KMeans_Advanced_Output.xlsx", index=False)
print("\nFile Saved: KMeans_Advanced_Output.xlsx")


# 11) Seaborn Visualization

plt.figure(figsize=(8,6))
sns.scatterplot(
    data=df,
    x="StudyHours",
    y="Marks",
    hue="Cluster",
    palette="viridis",
    s=150
)
plt.title("K-Means Clustering (Study Hours vs Marks)", fontsize=15, fontweight='bold')
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()
