import torch
import torch.nn as nn
import torchvision.models as models


class MobileNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        base_model = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.DEFAULT
        )

        self.features = base_model.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.output_dim = 1280

    def forward(self, x):
        B, T, C, H, W = x.shape

        x = x.view(B * T, C, H, W)

        x = self.features(x)
        x = self.pool(x)

        x = x.view(B, T, -1)  # (B, T, 1280)

        return x


class TransformerHead(nn.Module):
    def __init__(self, input_dim, num_heads=4, num_layers=2):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

    def forward(self, x):
        return self.transformer(x)


class MobileNetTransformer(nn.Module):
    def __init__(self, num_classes, seq_len=20):
        super().__init__()

        self.encoder = MobileNetEncoder()

        self.pos_embedding = nn.Parameter(
            torch.randn(1, seq_len, self.encoder.output_dim)
        )

        self.transformer = TransformerHead(
            input_dim=self.encoder.output_dim
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.encoder.output_dim),
            nn.Linear(self.encoder.output_dim, num_classes)
        )

    def forward(self, x):
        # (B, T, C, H, W)
        x = self.encoder(x)

        # positional encoding
        x = x + self.pos_embedding[:, :x.size(1), :]

        # transformer
        x = self.transformer(x)

        # pooling (lebih stabil)
        x = x.mean(dim=1)

        x = self.classifier(x)

        return x