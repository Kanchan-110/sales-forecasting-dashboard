import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Sales Overview", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")

    df["Order Date"] = pd.to_datetime(
        df["Order Date"],
        dayfirst=True,
        errors="coerce"
    )

    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.to_period("M").astype(str)

    return df

df = load_data()

st.title("📈 Sales Overview Dashboard")

yearly_sales = (
    df.groupby("Year")["Sales"]
    .sum()
    .reset_index()
)

fig1 = px.bar(
    yearly_sales,
    x="Year",
    y="Sales",
    title="Total Sales by Year",
    text_auto=".2s"
)

st.plotly_chart(fig1, use_container_width=True)

monthly_sales = (
    df.groupby("Month")["Sales"]
    .sum()
    .reset_index()
)

fig2 = px.line(
    monthly_sales,
    x="Month",
    y="Sales",
    markers=True,
    title="Monthly Sales Trend"
)

st.plotly_chart(fig2, use_container_width=True)

st.subheader("Sales by Region and Category")

region = st.multiselect(
    "Select Region",
    sorted(df["Region"].unique()),
    default=sorted(df["Region"].unique())
)

category = st.multiselect(
    "Select Category",
    sorted(df["Category"].unique()),
    default=sorted(df["Category"].unique())
)

filtered = df[
    (df["Region"].isin(region)) &
    (df["Category"].isin(category))
]

summary = (
    filtered.groupby(["Region", "Category"])["Sales"]
    .sum()
    .reset_index()
)

fig3 = px.bar(
    summary,
    x="Region",
    y="Sales",
    color="Category",
    barmode="group",
    title="Sales by Region and Category"
)

st.plotly_chart(fig3, use_container_width=True)