import torch
import torch.nn as nn
from dotenv import load_dotenv
import os

load_dotenv()

class PromoterCNN(nn.Module):
    def __init__(self, seqLength):
        super(PromoterCNN, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=4, out_channels=256, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(256)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * (seqLength // 4), 128)  # Adjusted for sequence length
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 1)  # Binary classification

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change shape to (batch, channels, seq_length)
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return torch.sigmoid(self.fc2(x))

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCELoss()

    def forward(self, y_pred, y_true):
        bce_loss = self.bce(y_pred, y_true)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        loss = self.alpha * (1 - p_t) ** self.gamma * bce_loss
        return loss.mean()