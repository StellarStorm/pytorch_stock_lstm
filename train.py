import pandas as pd
import streamlit as st
import torch

from dataset import StockDataset
from model import LSTMClassifier
from stock_data import fetch_stock_data

st.title("Stock Price Direction and Percentage Change Prediction")

ticker = st.text_input("Enter stock ticker symbol")
print("Ticker:", ticker)

if ticker:
    df = fetch_stock_data(ticker)
    st.write("Historical Data", df.tail())

    model = LSTMClassifier(input_size=5)
    model.load_state_dict(torch.load("model.pth"))
    model.eval()

    last_30_days = df.tail(30)
    ds = StockDataset(last_30_days)
    if len(ds) == 0:
        st.error("Not enough data for prediction.")
        st.stop()
    x, _, _ = ds[len(ds) - 1]
    x = x.unsqueeze(0)

    out_dir, out_pct = model(x)
    probs = torch.softmax(out_dir, dim=2).squeeze(0).tolist()

    forecast_pct = out_pct.squeeze(0).tolist()
    forecast_days = [f"Day +{i+1}" for i in range(5)]
    forecast_df = pd.DataFrame(
        {
            "Day": forecast_days,
            "Predicted % Change": [f"{p*100:.2f}%" for p in forecast_pct],
        }
    )

    st.markdown("**Forecasted Percentage Change for Next 5 Days:**")
    st.dataframe(forecast_df)

    probs_df = pd.DataFrame(probs, columns=["Down %", "Up %"])
    probs_df["Day"] = [f"Day +{i+1}" for i in range(5)]
    probs_df["Down %"] = probs_df["Down %"].apply(lambda p: f"{p*100:.2f}%")
    probs_df["Up %"] = probs_df["Up %"].apply(lambda p: f"{p*100:.2f}%")
    probs_df = probs_df[["Day", "Down %", "Up %"]]
    st.markdown("**Prediction Confidence (Softmax Probabilities):**")
    st.dataframe(probs_df)
