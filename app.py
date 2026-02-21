import os
import subprocess

import pandas as pd
import streamlit as st
import torch

from dataset import StockDataset
from model import LSTMClassifier
from stock_data import fetch_stock_data

st.title("📈 LSTM Stock Price Movement Predictor")
ticker = st.text_input("Enter Stock Ticker", value="AAPL")
window_size = 30  # synced across training and prediction
st.text(f"Window Size (Days): {window_size}")

if st.button("Train Model"):
    with st.spinner("Training..."):
        subprocess.run(["python", "train.py"])
        st.success("Model retrained.")

model_path = "model.pth"
if os.path.exists(model_path):
    mod_time = os.path.getmtime(model_path)
    st.caption(
        f"📦 Last trained model saved: {pd.to_datetime(mod_time, unit='s')}"
    )

if st.button("Predict Future Movement"):
    df = fetch_stock_data(ticker)
    st.text(f"Total rows after cleaning: {len(df)}")
    if len(df) <= window_size:
        st.warning(f"Not enough raw data (need > window size={window_size}).")
    else:
        current_price = float(df['Close'].iloc[-1])
        previous_close = (
            df['Close'].iloc[-2].item() if len(df) > 1 else current_price
        )
        price_change = current_price - previous_close
        percent_change = (
            (price_change / previous_close) * 100 if previous_close != 0 else 0
        )
        st.markdown("**Current Price and change from yesterday close**")
        st.metric(
            label="",  # leave label empty since we're using markdown above
            value=f"${current_price:.2f}",
            delta=f"{price_change:+.2f} ({percent_change:+.2f}%)",
        )
        dataset = StockDataset(df, window=window_size)
        st.text(f"Dataset length (#samples): {len(dataset)}")
        if len(dataset) == 0:
            st.warning("Zero samples in dataset after windowing.")
        else:
            x, _, _ = dataset[len(dataset) - 1]
            st.text(f"Final sequence shape (no batch yet): {x.shape}")
            if x.numel() == 0 or x.shape[0] == 0:
                st.error("Empty sequence detected; cannot run model.")
            else:
                x = x.unsqueeze(0)
                st.text(f"Input shape including batch: {x.shape}")
                model = LSTMClassifier(input_size=5)
                model.load_state_dict(torch.load("model.pth"))
                model.eval()
                try:
                    with torch.no_grad():
                        out_dir, out_pct = model(x)
                        probs = (
                            torch.softmax(out_dir, dim=2).squeeze(0).tolist()
                        )
                        # Derive direction directly from percent change
                        pct_preds = out_pct.squeeze(0).tolist()
                        preds = [1 if c > 0 else 0 for c in pct_preds]
                except RuntimeError as e:
                    st.error(f"Runtime error in model inference: {e}")
                    raise
                expected_prices = [current_price * (1 + c) for c in pct_preds]
                forecast_df = pd.DataFrame(
                    {
                        "Day": [f"Day +{i+1}" for i in range(5)],
                        "Prediction": [
                            "📈 Up" if p == 1 else "📉 Down" for p in preds
                        ],
                        "Expected Change %": [
                            f"{c*100:.2f}%" for c in pct_preds
                        ],
                        "Expected Price": [
                            f"${p:.2f}" for p in expected_prices
                        ],
                    }
                )
                st.dataframe(forecast_df)
                # Add mini line chart below the prediction table
                chart_df = forecast_df.copy()
                chart_df["Expected Change %"] = [
                    float(val.strip('%'))
                    for val in chart_df["Expected Change %"]
                ]
                chart_df.set_index("Day", inplace=True)
                st.line_chart(chart_df["Expected Change %"], height=200)
                forecast_df["Ticker"] = ticker
                forecast_df["Date"] = pd.Timestamp.now().strftime("%Y-%m-%d")
                forecast_df.to_csv(
                    "prediction_log.csv",
                    mode='a',
                    header=not pd.io.common.file_exists("prediction_log.csv"),
                    index=False,
                )
                probs_df = pd.DataFrame(probs, columns=["Down %", "Up %"])
                probs_df["Day"] = [f"Day +{i+1}" for i in range(5)]
                probs_df["Down %"] = probs_df["Down %"].apply(
                    lambda p: f"{p*100:.2f}%"
                )
                probs_df["Up %"] = probs_df["Up %"].apply(
                    lambda p: f"{p*100:.2f}%"
                )
                probs_df = probs_df[["Day", "Down %", "Up %"]]
                st.markdown(
                    "**Prediction Confidence (Softmax Probabilities):**"
                )
                st.dataframe(probs_df)
