import torch
from torch.utils.data import Dataset


class StockDataset(Dataset):
    def __init__(self, df, window=10):
        self.window = window
        self.features = df.drop(columns='Target').values
        self.labels = df['Target'].values

    def __len__(self):
        return max(0, len(self.features) - self.window - 5)

    def __getitem__(self, idx):
        if idx < 0 or idx + self.window + 5 > len(self.features):
            raise IndexError(f"Index {idx} out of bounds")
        x = self.features[idx : idx + self.window]
        direction_labels = self.labels[
            idx + self.window : idx + self.window + 5
        ]
        pct_changes = self.features[
            idx + self.window : idx + self.window + 5, 3
        ]  # use Close for % change reference
        y_pct = (
            pct_changes - self.features[idx + self.window - 1, 3]
        ) / self.features[idx + self.window - 1, 3]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(direction_labels, dtype=torch.long),
            torch.tensor(y_pct, dtype=torch.float32),
        )
