import torch
import torch.nn as nn
import torchvision.models as models


class MobileNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        base_model = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.DEFAULT
        )

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

        self.features = base_model.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.output_dim = 1280

    def forward(self, x):
        B, T, C, H, W = x.shape

        x = x.view(B * T, C, H, W)

        x = self.features(x)
        x = self.pool(x)

        x = x.view(B, T, -1)

        return x

    def set_trainable_blocks(self, num_unfrozen_blocks=None):
        """
        Freeze/unfreeze self.features by top-level block. Mirrors
        CNN_Encoder.set_trainable_blocks() in models/cnn_lstm.py so both
        encoders can be fine-tuned to a matched depth for a fair
        CNN_LSTM-vs-MobileNetTransformer comparison (the original code only
        ever unfroze MobileNetV2's last 4 blocks via
        `model.encoder.features[-4:]`, see train/train_mobile_net.ipynb,
        while CNN_LSTM was always fully unfrozen).

        num_unfrozen_blocks:
            None -> unfreeze everything (full fine-tune).
            0    -> freeze the entire backbone (pure feature extractor).
            N>0  -> unfreeze only the last N of self.features' 19 top-level
                    blocks (the original hardcoded behaviour was N=4).

        Returns the total number of top-level blocks (for reference/logging).
        """
        blocks = list(self.features.children())
        total = len(blocks)

        if num_unfrozen_blocks is None or num_unfrozen_blocks >= total:
            for p in self.features.parameters():
                p.requires_grad = True
            return total

        if num_unfrozen_blocks < 0:
            raise ValueError("num_unfrozen_blocks must be >= 0 or None")

        for p in self.features.parameters():
            p.requires_grad = False

        if num_unfrozen_blocks > 0:
            for block in blocks[-num_unfrozen_blocks:]:
                for p in block.parameters():
                    p.requires_grad = True

        return total

    @property
    def num_blocks(self):
        return len(list(self.features.children()))


class AttentionPool(nn.Module):
    """
    Bahdanau-style additive attention pooling over the time dimension.

    Mirrors the `Attention` class in models/cnn_lstm.py exactly (same
    Linear -> Tanh -> Linear -> softmax -> weighted sum). It is added here as
    an alternative to MobileNetTransformer's original last-token pooling
    (`x[:, -1, :]`), which discards everything the Transformer encoder computed
    for earlier frames. Using the same pooling mechanism in both architectures
    also removes one more confound when comparing CNN_LSTM and
    MobileNetTransformer head-to-head (see pooling="attention" below).
    """

    def __init__(self, hidden_dim):
        super().__init__()

        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # x: (B, T, H)
        weights = torch.softmax(self.attn(x), dim=1)
        context = (x * weights).sum(dim=1)
        return context


class TransformerHead(nn.Module):
    def __init__(self, input_dim, num_heads=8, num_layers=4):
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
    """
    Parameters
    ----------
    num_classes : int
    seq_len : int
        Sequence length used to size the learned positional embedding.
    num_layers : int
        Number of Transformer encoder layers. NOTE: earlier versions of this
        model hardcoded num_layers=4 regardless of what config.json said
        (some saved run configs claim "transformer_layers": 2 even though the
        checkpoint was actually trained with 4 layers -- this was a stale/
        misleading metadata bug, not a real ablation). This is now a real
        constructor argument so future config.json files are trustworthy.
    pooling : {"last", "mean", "attention"}
        How to collapse the Transformer's (B, T, H) output into (B, H) before
        classification.
          - "last": original behaviour, x[:, -1, :]. Only uses the final
            timestep's representation, so it works more like a decoder
            readout and throws away the context the encoder built for every
            other frame.
          - "mean": average over all T timesteps. Cheap, parameter-free,
            uses the full sequence.
          - "attention": Bahdanau-style attention pooling (AttentionPool
            above), identical in form to the pooling already used by
            CNN_LSTM (models/cnn_lstm.py). This is the fairest choice when
            comparing the two architectures, since both then aggregate their
            per-frame temporal representations the same way.
    """

    def __init__(self, num_classes, seq_len=20, num_layers=4, pooling="last"):
        super().__init__()

        if pooling not in ("last", "mean", "attention"):
            raise ValueError(f"pooling must be 'last', 'mean', or 'attention', got {pooling!r}")

        self.pooling = pooling
        self.encoder = MobileNetEncoder()

        #  PROJECTION
        self.proj = nn.Linear(1280, 512)

        #  posisi embedding
        self.pos_embedding = nn.Parameter(
            torch.randn(1, seq_len, 512) * 0.02
        )

        self.transformer = TransformerHead(
            input_dim=512,
            num_heads=8,
            num_layers=num_layers
        )

        if self.pooling == "attention":
            self.attention_pool = AttentionPool(hidden_dim=512)

        self.classifier = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # (B, T, C, H, W)
        x = self.encoder(x)

        #  projection
        x = self.proj(x)

        # positional encoding
        x = x + self.pos_embedding[:, :x.size(1), :]

        # transformer
        x = self.transformer(x)

        # pooling
        if self.pooling == "last":
            x = x[:, -1, :]
        elif self.pooling == "mean":
            x = x.mean(dim=1)
        else:  # "attention"
            x = self.attention_pool(x)

        x = self.classifier(x)

        return x