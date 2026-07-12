import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Forecast Explorer", layout="wide")

st.title("🔮 Forecast Explorer")

df = pd.read_csv("train.csv")

df["Order Date"] = pd.to_datetime(
    df["Order Date"],
    dayfirst=True,
    errors="coerce"
)

forecast_type = st.selectbox(
    "Forecast By",
    ["Category", "Region"]
)

if forecast_type == "Category":
    selected = st.selectbox(
        "Select Category",
        sorted(df["Category"].unique())
    )
    segment = df[df["Category"] == selected]

else:
    selected = st.selectbox(
        "Select Region",
        sorted(df["Region"].unique())
    )
    segment = df[df["Region"] == selected]

months = st.slider(
    "Forecast Horizon (Months)",
    1,
    3,
    3
)

monthly = (
    segment
    .groupby(pd.Grouper(key="Order Date", freq="MS"))["Sales"]
    .sum()
    .reset_index()
)

monthly.columns = ["Date", "Sales"]

monthly["Lag_1"] = monthly["Sales"].shift(1)
monthly["Lag_2"] = monthly["Sales"].shift(2)
monthly["Lag_3"] = monthly["Sales"].shift(3)

monthly["Rolling_Mean_3"] = (
    monthly["Sales"]
    .rolling(3)
    .mean()
    .shift(1)
)

monthly["Month"] = monthly["Date"].dt.month
monthly["Quarter"] = monthly["Date"].dt.quarter
monthly["Season"] = ((monthly["Month"] % 12 + 3) // 3)

monthly = monthly.dropna()

X = monthly[
    [
        "Lag_1",
        "Lag_2",
        "Lag_3",
        "Rolling_Mean_3",
        "Month",
        "Quarter",
        "Season"
    ]
]

y = monthly["Sales"]

X_train = X[:-3]
X_test = X[-3:]

y_train = y[:-3]
y_test = y[-3:]

model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

test_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, test_pred)
rmse = np.sqrt(mean_squared_error(y_test, test_pred))
predictions = []

last = monthly.iloc[-1].copy()

for i in range(months):

    X_future = pd.DataFrame({
        "Lag_1":[last["Lag_1"]],
        "Lag_2":[last["Lag_2"]],
        "Lag_3":[last["Lag_3"]],
        "Rolling_Mean_3":[last["Rolling_Mean_3"]],
        "Month":[last["Month"]],
        "Quarter":[last["Quarter"]],
        "Season":[last["Season"]]
    })

    pred = model.predict(X_future)[0]
    predictions.append(pred)

    lag1 = pred
    lag2 = last["Lag_1"]
    lag3 = last["Lag_2"]

    rolling = (lag1 + lag2 + lag3) / 3

    month = (last["Month"] % 12) + 1
    quarter = ((month - 1) // 3) + 1
    season = ((month % 12 + 3) // 3)

    last["Lag_1"] = lag1
    last["Lag_2"] = lag2
    last["Lag_3"] = lag3
    last["Rolling_Mean_3"] = rolling
    last["Month"] = month
    last["Quarter"] = quarter
    last["Season"] = season

future_dates = pd.date_range(
    monthly["Date"].max() + pd.offsets.MonthBegin(1),
    periods=months,
    freq="MS"
)

forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Forecast Sales": np.round(predictions, 2)
})

st.subheader("Forecast")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=monthly["Date"],
        y=monthly["Sales"],
        mode="lines+markers",
        name="Actual Sales"
    )
)

fig.add_trace(
    go.Scatter(
        x=forecast_df["Date"],
        y=forecast_df["Forecast Sales"],
        mode="lines+markers",
        name="Forecast"
    )
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Forecast Table")
st.dataframe(forecast_df, use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    st.metric("MAE", f"{mae:.2f}")

with col2:
    st.metric("RMSE", f"{rmse:.2f}")

    from sklearn.metrics import mean_absolute_percentage_error

mape = mean_absolute_percentage_error(y_test, test_pred)

st.metric("MAPE", f"{mape:.2%}")