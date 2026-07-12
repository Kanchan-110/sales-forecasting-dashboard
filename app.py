import streamlit as st

st.set_page_config(
    page_title="Sales Forecast Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Sales Forecasting Dashboard")

st.markdown("""
Welcome to the **Sales Forecasting Dashboard**.

Use the **sidebar** to navigate through the dashboard pages.

### Pages
- 📈 Sales Overview
- 🔮 Forecast Explorer
- 🚨 Anomaly Report
- 📦 Product Demand Segments
""")