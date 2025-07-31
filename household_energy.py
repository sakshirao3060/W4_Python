import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load dataset
df = pd.read_csv("household_power_consumption.csv", sep=';', na_values='?', low_memory=False)
print("Original Data Shape:", df.shape)

# Step 2: Drop rows with missing values
df_clean = df.dropna()
print("After Dropping NA:", df_clean.shape)

# Step 3: Drop 'Date' and 'Time' columns
df_features = df_clean.drop(columns=['Date', 'Time'])

# Step 4: Convert to float
df_features = df_features.astype(float).sample(n=10000, random_state=42)
# Step 5: Scale the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_features)

# Step 6: PCA for 2D visualization
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

# Step 7: KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(pca_data)

# Step 8: Evaluate using silhouette score
score = silhouette_score(pca_data, labels)
print(f"Silhouette Score: {score:.2f}")

# Step 9: Visualize clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=pca_data[:, 0], y=pca_data[:, 1], hue=labels, palette='Set2', s=60)
plt.title("Household Energy Usage Clusters")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title='Cluster')
plt.tight_layout()
plt.show()