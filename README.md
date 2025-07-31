# 🔌 Household Energy Usage Clustering

This project applies **unsupervised machine learning** to segment households based on their energy consumption patterns using **PCA** and **K-Means clustering**.

---

## 📊 Objective

To group households into meaningful clusters based on their power meter readings, helping identify patterns like high, moderate, or low energy usage.

---

## 📁 Dataset

- **Name**: Individual household electric power consumption
- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption)
- **Records**: 2,075,259 rows
- **Features**: 9 columns including:
  - Global active power
  - Global reactive power
  - Voltage
  - Sub metering 1, 2, 3, etc.

---

## 🧪 Technologies Used

- Python 3
- Pandas
- NumPy
- Scikit-learn (`StandardScaler`, `PCA`, `KMeans`, `silhouette_score`)
- Seaborn & Matplotlib for visualization

---

## 🛠️ How It Works

1. Load the dataset using semicolon-separated values (`sep=';'`) and treat `'?'` as missing.
2. Drop rows with missing values.
3. Remove non-numeric columns (`Date`, `Time`).
4. Normalize the numeric features using `StandardScaler`.
5. Reduce dimensionality with `PCA` to 2 components.
6. Apply `KMeans` clustering with 3 clusters.
7. Evaluate clustering using **Silhouette Score**.
8. Visualize the clusters in 2D PCA space.

---

## ▶️ How to Run

1. Clone or download this repository.
2. Make sure to unzip the dataset `household_power_consumption.csv` is in the same folder.
3. Install dependencies (if needed):

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
