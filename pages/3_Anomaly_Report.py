import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="Anomaly Report", layout="wide")

st.title("🚨 Sales Anomaly Report")

df = pd.read_csv("train.csv")

df["Order Date"] = pd.to_datetime(
    df["Order Date"],
    dayfirst=True,
    errors="coerce"
)

df = df.dropna(subset=["Order Date"])

weekly_sales = (
    df.groupby(pd.Grouper(key="Order Date", freq="W"))["Sales"]
    .sum()
    .reset_index()
)

iso = IsolationForest(
    contamination=0.05,
    random_state=42
)

weekly_sales["Anomaly"] = iso.fit_predict(
    weekly_sales[["Sales"]]
)

anomalies = weekly_sales[
    weekly_sales["Anomaly"] == -1
]

fig, ax = plt.subplots(figsize=(12,5))

ax.plot(
    weekly_sales["Order Date"],
    weekly_sales["Sales"],
    label="Weekly Sales"
)

ax.scatter(
    anomalies["Order Date"],
    anomalies["Sales"],
    color="red",
    label="Anomalies",
    s=50
)

ax.set_title("Isolation Forest - Weekly Sales Anomalies")
ax.set_xlabel("Date")
ax.set_ylabel("Sales")
ax.legend()

st.pyplot(fig)

st.subheader("Detected Anomalies")

st.dataframe(
    anomalies[["Order Date","Sales"]],
    use_container_width=True
)

st.metric(
    "Total Anomalies",
    len(anomalies)
)