
import math
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class TimeSformer(nn.Module):
    """
    Time Series Transformer for direct multi-horizon prediction.
    Encodes a sequence of past features (lookback window) and predicts
    future log returns for specific horizons (e.g., 7, 30, 60, 90, 180, 365 days).
    """
    def __init__(self, num_features, d_model=64, nhead=4, num_layers=2, 
                 dim_feedforward=256, dropout=0.1, output_dim=1):
        super(TimeSformer, self).__init__()
        
        self.d_model = d_model
        
        # Input embedding: maps feature vector to d_model
        self.input_linear = nn.Linear(num_features, d_model)
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Output head
        # We can either predict one horizon (output_dim=1) or multiple (output_dim=N)
        self.output_linear = nn.Linear(d_model, output_dim)
        
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.input_linear.weight.data.uniform_(-initrange, initrange)
        self.output_linear.bias.data.zero_()
        self.output_linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        # src shape: [batch_size, seq_len, num_features]
        
        # 1. Embed features
        src = self.input_linear(src) * math.sqrt(self.d_model) # [batch, seq, d_model]
        
        # 2. Add Positional Encoding
        src = self.pos_encoder(src)
        
        # 3. Transformer Encode
        output = self.transformer_encoder(src) # [batch, seq, d_model]
        
        # 4. Global Average Pooling or Take Last Token
        # Here we take the last token as it summarizes the sequence up to t
        output = output[:, -1, :] # [batch, d_model]
        
        # 5. Predict
        prediction = self.output_linear(output) # [batch, output_dim]
        return prediction

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0) # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class CryptoSequenceDataset(Dataset):
    """
    Creates sliding windows for Time Series training.
    """
    def __init__(self, X, y, seq_len=60):
        """
        X: numpy array of features [num_samples, num_features]
        y: numpy array of targets [num_samples, ] or [num_samples, num_targets]
        seq_len: lookback window size
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        # Input: idx to idx+seq_len
        # Target: target at idx+seq_len (we predict for T based on T-seq_len...T-1)
        # Actually, in standard TS forecasting:
        # We use data from [i : i+seq_len] to predict target at [i+seq_len] (or future horizon)
        
        # If y is aligned with X (i.e., y[t] is the target for X[t]), 
        # then we want to use X[i : i+seq_len] to predict y[i+seq_len] ??
        # 
        # Wait, let's clarify alignment.
        # X[t] usually contains features known at time t.
        # Target y[t] is usually the return from t to t+h.
        # So at time t, we know X[t-seq_len+1 ... t]. We want to predict y[t].
        
        # Implementation:
        # X_window = X[idx : idx+seq_len]
        # y_target = y[idx + seq_len - 1] 
        # (Assuming y is already aligned such that y[t] is target for X[t])
        
        return self.X[idx : idx + self.seq_len], self.y[idx + self.seq_len - 1]
