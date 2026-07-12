import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

st.set_page_config(page_title="Product Demand Segments", layout="wide")

st.title("📦 Product Demand Segments")

df = pd.read_csv("train.csv")

df["Order Date"] = pd.to_datetime(
    df["Order Date"],
    dayfirst=True,
    errors="coerce"
)

df["Year"] = df["Order Date"].dt.year
df["Month"] = df["Order Date"].dt.to_period("M")

monthly = (
    df.groupby(["Sub-Category", "Month"])["Sales"]
    .sum()
    .reset_index()
)

monthly["Year"] = monthly["Month"].dt.year

features = []

for subcat in monthly["Sub-Category"].unique():

    temp = monthly[
        monthly["Sub-Category"] == subcat
    ]

    total_sales = temp["Sales"].sum()

    yearly = temp.groupby("Year")["Sales"].sum()

    if len(yearly) > 1:
        growth = yearly.pct_change().mean()
    else:
        growth = 0

    volatility = temp["Sales"].std()

    avg_order = df[
        df["Sub-Category"] == subcat
    ]["Sales"].mean()

    features.append([
        subcat,
        total_sales,
        growth,
        volatility,
        avg_order
    ])

cluster_df = pd.DataFrame(
    features,
    columns=[
        "Sub-Category",
        "Total Sales",
        "Growth",
        "Volatility",
        "Average Order Value"
    ]
)

cluster_df.fillna(0, inplace=True)

X = cluster_df.drop(
    "Sub-Category",
    axis=1
)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

cluster_df["Cluster"] = kmeans.fit_predict(X_scaled)

labels = {
    0:"High Volume, Stable Demand",
    1:"Growing Demand",
    2:"Low Volume, High Volatility",
    3:"Declining Demand"
}

cluster_df["Demand Group"] = cluster_df["Cluster"].map(labels)

pca = PCA(n_components=2)

components = pca.fit_transform(X_scaled)

cluster_df["PCA1"] = components[:,0]
cluster_df["PCA2"] = components[:,1]

fig, ax = plt.subplots(figsize=(9,6))

scatter = ax.scatter(
    cluster_df["PCA1"],
    cluster_df["PCA2"],
    c=cluster_df["Cluster"],
    s=120
)

for i in range(len(cluster_df)):
    ax.text(
        cluster_df["PCA1"][i],
        cluster_df["PCA2"][i],
        cluster_df["Sub-Category"][i],
        fontsize=8
    )

ax.set_title("Product Demand Segments")

st.pyplot(fig)

st.subheader("Demand Cluster Table")

st.dataframe(
    cluster_df[
        [
            "Sub-Category",
            "Demand Group"
        ]
    ],
    use_container_width=True
)

st.subheader("Recommended Stocking Strategy")

strategy = pd.DataFrame({
    "Demand Group":[
        "High Volume, Stable Demand",
        "Growing Demand",
        "Low Volume, High Volatility",
        "Declining Demand"
    ],
    "Recommendation":[
        "Maintain high inventory levels.",
        "Increase inventory gradually.",
        "Keep limited stock and replenish based on demand.",
        "Reduce inventory and avoid overstocking."
    ]
})

st.table(strategy)