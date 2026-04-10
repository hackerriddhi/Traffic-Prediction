import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# =========================
# CREATE SEQUENCES
# =========================
def create_sequences(data, seq_len=5):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len][0])  # traffic
    return np.array(X), np.array(y)


# =========================
# LSTM MODEL
# =========================
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


# =========================
# TRAIN FUNCTION
# =========================
def train_lstm(df, seq_len=5):

    feature_cols = [
        "traffic", "Hour", "DayOfWeek",
        "Is_Weekend", "Is_Peak_Hour",
        "lag_1", "rolling_mean_3",
        "hour_sin", "hour_cos",
        "dow_sin", "dow_cos",
        "weekend_peak"
    ]

    data = df[feature_cols].values

    # SCALE DATA
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    X, y = create_sequences(data_scaled, seq_len)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    model = LSTMModel(input_size=X.shape[2])

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # TRAINING
    for epoch in range(100):
        output = model(X)
        loss = criterion(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    return model, scaler, data_scaled[-seq_len:]