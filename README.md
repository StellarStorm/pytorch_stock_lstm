# 🧠 PyTorch Stock Price Forecasting App

This project uses **PyTorch** and **LSTM networks** to predict stock price direction and percentage change over the next 5 days. It features a real-time **Streamlit UI** for live prediction and interactive visualization.

## 📦 Features

- Predicts both **Up/Down direction** and **expected % change**
- 5-day multi-output forecast using an LSTM model
- Softmax-based confidence visualization
- Fully interactive UI via Streamlit
- Real-time data pulled using `yfinance`

## 🧰 Tech Stack

- PyTorch
- Streamlit
- yFinance
- Pandas

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/viplavfauzdar/pytorch_stock_lstm.git
cd pytorch_stock_lstm
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app

```bash
streamlit run train.py
```

## 📈 How It Works

- Takes the last 30 days of OHLCV data
- Feeds it into an `nn.LSTM`
- Outputs:
  - `5x2` direction logits (Up/Down) → softmax → confidence
  - `5` float % changes → displayed as predictions

## 🗂 Files

- `model.py`: LSTM model definition
- `train.py`: Streamlit UI & inference app
- `stock_data.py`: Data fetching using yFinance
- `dataset.py`: Windowed PyTorch dataset

## 🧠 Author

Built by [Viplav Fauzdar](https://viplavfauzdar.com)

## 📄 License

MIT
