import os

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from dataset import StockDataset
from model import LSTMClassifier, accel
from stock_data import fetch_stock_data


def train_model(
    ticker, epochs=50, batch_size=32, learning_rate=0.001, hidden_size=64
):
    """
    Train the LSTM stock prediction model.

    Args:
        ticker: Stock ticker symbol
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        hidden_size: Hidden size of LSTM
    """
    # Fetch data
    print(f'Fetching data for {ticker}...')
    df = fetch_stock_data(ticker)
    print(f'Downloaded {len(df)} days of data')

    # Split data into train and validation sets (80-20 split)
    train_df, val_df = train_test_split(df, test_size=0.2, shuffle=False)

    # Create datasets
    train_dataset = StockDataset(train_df)
    val_dataset = StockDataset(val_df)

    print(f'Training samples: {len(train_dataset)}')
    print(f'Validation samples: {len(val_dataset)}')

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    device = accel()
    print(f'Using device: {device}')

    model = LSTMClassifier(input_size=5, hidden_size=hidden_size, num_layers=1)
    model.to(device)

    # Loss functions
    criterion_direction = nn.CrossEntropyLoss()
    criterion_pct = nn.MSELoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    print('\nStarting training...')
    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_dir_loss = 0.0
        train_pct_loss = 0.0

        for batch_idx, (x, y_dir, y_pct) in enumerate(train_loader):
            x = x.to(device)
            y_dir = y_dir.to(device)
            y_pct = y_pct.to(device)

            # Forward pass
            dir_logits, pct_pred = model(x)

            # Reshape for loss calculation
            # dir_logits: [batch, 5, 2] -> [batch*5, 2]
            # y_dir: [batch, 5] -> [batch*5]
            batch_size_curr = x.size(0)
            dir_logits_flat = dir_logits.view(batch_size_curr * 5, 2)
            y_dir_flat = y_dir.view(batch_size_curr * 5)

            # Calculate losses
            loss_dir = criterion_direction(dir_logits_flat, y_dir_flat)
            loss_pct = criterion_pct(pct_pred, y_pct)
            loss = loss_dir + loss_pct

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_dir_loss += loss_dir.item()
            train_pct_loss += loss_pct.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_train_dir_loss = train_dir_loss / len(train_loader)
        avg_train_pct_loss = train_pct_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_dir_loss = 0.0
        val_pct_loss = 0.0

        with torch.no_grad():
            for x, y_dir, y_pct in val_loader:
                x = x.to(device)
                y_dir = y_dir.to(device)
                y_pct = y_pct.to(device)

                # Forward pass
                dir_logits, pct_pred = model(x)

                # Reshape for loss calculation
                batch_size_curr = x.size(0)
                dir_logits_flat = dir_logits.view(batch_size_curr * 5, 2)
                y_dir_flat = y_dir.view(batch_size_curr * 5)

                # Calculate losses
                loss_dir = criterion_direction(dir_logits_flat, y_dir_flat)
                loss_pct = criterion_pct(pct_pred, y_pct)
                loss = loss_dir + loss_pct

                val_loss += loss.item()
                val_dir_loss += loss_dir.item()
                val_pct_loss += loss_pct.item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_dir_loss = val_dir_loss / len(val_loader)
        avg_val_pct_loss = val_pct_loss / len(val_loader)

        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f'Epoch [{epoch + 1}/{epochs}]')
            print(
                f'  Train - Total: {avg_train_loss:.4f} | Direction: {avg_train_dir_loss:.4f} | Pct: {avg_train_pct_loss:.4f}'
            )
            print(
                f'  Val   - Total: {avg_val_loss:.4f} | Direction: {avg_val_dir_loss:.4f} | Pct: {avg_val_pct_loss:.4f}'
            )

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                model.state_dict(),
                os.path.join('models', f'{ticker}_model.pth'),
            )
            print(
                f'  ✓ Model saved (best validation loss: {best_val_loss:.4f})'
            )

    print('\nTraining complete!')
    print(f'Best validation loss: {best_val_loss:.4f}')
    return model
