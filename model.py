import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True
        )
        self.fc_dir = nn.Linear(hidden_size, 5 * 2)  # 5 days × 2-class
        self.fc_pct = nn.Linear(hidden_size, 5)  # 5 days percent change

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        x = hn[-1]
        direction_logits = self.fc_dir(x).view(
            -1, 5, 2
        )  # [batch, 5 days, 2 classes]
        pct_change = self.fc_pct(x)  # [batch, 5 days]
        return direction_logits, pct_change
