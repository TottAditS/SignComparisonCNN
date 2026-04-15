import torch
import torch.nn as nn
import torchvision.models as models


class MobileNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        base_model = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.DEFAULT
        )

<<<<<<< HEAD
        # ===== PATCH FIRST CONV (3 → 6 channel) =====
        first_conv = base_model.features[0][0]

        new_conv = nn.Conv2d(
            in_channels=6,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=False
        )

        with torch.no_grad():
            new_conv.weight[:, :3] = first_conv.weight
            new_conv.weight[:, 3:] = first_conv.weight

        base_model.features[0][0] = new_conv
        # ============================================

=======
>>>>>>> parent of b2a8462 (Mobile Net Fine Tune Checkpoint)
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
            dropout=0.3,
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

<<<<<<< HEAD
        #  PROJECTION (WAJIB)
        self.proj = nn.Linear(1280, 512)

        #  posisi embedding sesuai dim baru
        self.pos_embedding = nn.Parameter(
            torch.randn(1, seq_len, 512) * 0.02
=======
        self.pos_embedding = nn.Parameter(
            torch.randn(1, seq_len, self.encoder.output_dim)
>>>>>>> parent of b2a8462 (Mobile Net Fine Tune Checkpoint)
        )

        self.transformer = TransformerHead(
            input_dim=self.encoder.output_dim
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.encoder.output_dim),
            nn.Linear(self.encoder.output_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # (B, T, C, H, W)
        x = self.encoder(x)

<<<<<<< HEAD
        #  projection dulu
        x = self.proj(x)

=======
>>>>>>> parent of b2a8462 (Mobile Net Fine Tune Checkpoint)
        # positional encoding
        x = x + self.pos_embedding[:, :x.size(1), :]

        # transformer
        x = self.transformer(x)

<<<<<<< HEAD
        # pooling
        x = x[:, -1, :]
=======
        # pooling (lebih stabil)
        x = x.mean(dim=1)
>>>>>>> parent of b2a8462 (Mobile Net Fine Tune Checkpoint)

        x = self.classifier(x)

        return x