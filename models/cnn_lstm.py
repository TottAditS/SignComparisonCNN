import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


# =========================
# CNN ENCODER (UPGRADE)
# =========================
class CNN_Encoder(nn.Module):
    def __init__(self, feature_dim=512):
        super().__init__()

        base_model = resnet18(weights=ResNet18_Weights.DEFAULT)
        modules = list(base_model.children())[:-1]
        self.cnn = nn.Sequential(*modules)

        self.fc = nn.Sequential(
            nn.Linear(512, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# =========================
# ATTENTION (UPGRADE)
# =========================
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, lstm_out):
        # (B, T, H)
        weights = torch.softmax(self.attn(lstm_out), dim=1)
        context = (lstm_out * weights).sum(dim=1)
        return context


# =========================
# MAIN MODEL (UPGRADE)
# =========================
class CNN_LSTM(nn.Module):
    def __init__(self, num_classes, hidden_dim=256):  # ⬅️ naik dari 128
        super().__init__()

        self.encoder = CNN_Encoder()

        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.5  # ⬅️ lebih kuat
        )

        self.attention = Attention(hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape

        x = x.view(B * T, C, H, W)

        features = self.encoder(x)  # (B*T, 512)

        features = features.view(B, T, -1)

        lstm_out, _ = self.lstm(features)

        context = self.attention(lstm_out)

        out = self.classifier(context)

        return out