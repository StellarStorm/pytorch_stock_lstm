import torch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from dataset import StockDataset
from model import LSTMClassifier
from stock_data import fetch_stock_data


def evaluate(model, data_loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            outputs = model(x_batch)
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(y_batch.numpy())
            y_pred.extend(preds.numpy())
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(report)


if __name__ == "__main__":
    df = fetch_stock_data("AAPL")
    _, test_df = train_test_split(df, test_size=0.2, shuffle=False)
    test_ds = StockDataset(test_df)
    test_dl = DataLoader(test_ds, batch_size=32)

    model = LSTMClassifier(input_size=5)
    model.load_state_dict(torch.load("model.pth"))
    evaluate(model, test_dl)
