import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.title("🌍 Country Development Cluster Analyzer")

# Load dataset
df = pd.read_csv("clustering_data.csv")

country_col = "Country"

st.subheader("Dataset Preview")
st.write(df.head())

# -----------------------------
# LOAD SAVED MODELS
# -----------------------------
scaler = joblib.load("clustering_scaler.pkl")
pca = joblib.load("clustering_pca.pkl")
kmeans = joblib.load("clustering_kmeans.pkl")

# -----------------------------
# USE ALL NUMERIC FEATURES
# -----------------------------
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
features = numeric_cols

X = df[features]

# -----------------------------
# TRANSFORM DATA
# -----------------------------
X_scaled = scaler.transform(X)
X_pca = pca.transform(X_scaled)

df['Cluster'] = kmeans.predict(X_pca)

# -----------------------------
# CLUSTER LABELING
# -----------------------------
cluster_means = df.groupby('Cluster')[features[0]].mean()
sorted_clusters = cluster_means.sort_values().index.tolist()

cluster_labels = {
    sorted_clusters[0]: "Underdeveloped",
    sorted_clusters[1]: "Developing",
    sorted_clusters[2]: "Developed"
}

df['Development'] = df['Cluster'].map(cluster_labels)

# -----------------------------
# INPUT MODE
# -----------------------------
mode = st.radio(
    "Choose Input Type",
    ["Select Existing Country", "Enter Custom Values"]
)

# -----------------------------
# EXISTING COUNTRY
# -----------------------------
if mode == "Select Existing Country":

    selected_country = st.selectbox(
        "Choose a country",
        df[country_col].unique()
    )

    country_data = df[df[country_col] == selected_country].iloc[0]
    input_data = country_data[features]

    st.write("### Selected Country Data")
    st.write(input_data)

# -----------------------------
# CUSTOM INPUT
# -----------------------------
else:
    st.subheader("Enter Custom Country Data")
    custom_country_name = st.text_input(
        "Country Name",
        value="My Country"
    )

    user_input = {}

    for col in features:
        user_input[col] = st.number_input(
            f"{col}",
            value=float(df[col].mean())
        )

    input_data = pd.Series(user_input)

    st.write("### Your Input Data")
    st.write(input_data)
    # -----------------------------
# COMMON NAME VARIABLE
# -----------------------------
if mode == "Select Existing Country":
    name = selected_country
else:
    name = custom_country_name

# -----------------------------
# PREDICTION
# -----------------------------
input_df = pd.DataFrame([input_data])
input_df = input_df[features]  # ensure correct order

scaled = scaler.transform(input_df)
pca_data = pca.transform(scaled)
cluster = kmeans.predict(pca_data)[0]

dev_label = cluster_labels[cluster]

st.success(f"🌍 Predicted Development: **{dev_label}** (Cluster {cluster})")

# -----------------------------
# VISUALIZATION (PCA)
# -----------------------------
plt.figure(figsize=(8,6))

for c in range(3):
    subset = X_pca[df['Cluster'] == c]
    plt.scatter(
        subset[:, 0],
        subset[:, 1],
        label=cluster_labels[c]
    )

# Plot selected/custom point
point = pca.transform(scaler.transform(input_df))

plt.scatter(
    point[:, 0],
    point[:, 1],
    color='black',
    marker='X',
    s=200
)
# ✅ Add country name label
plt.text(
    point[0, 0],
    point[0, 1],
    name,
    fontsize=10,
    ha='right'
)


plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("Country Clusters (PCA View)")
plt.legend()
plt.grid(True)

st.pyplot(plt)