import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

# CNN ENCODER

class CNN_Encoder(nn.Module):
    def __init__(self, feature_dim=512):
        super().__init__()

        base_model = resnet18(weights=ResNet18_Weights.DEFAULT)

        first_conv = base_model.conv1

        # conv  (6 channel)
        new_conv = nn.Conv2d(
            in_channels=6,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=False
        )

        # copy weight RGB → 6 channel
        with torch.no_grad():
            new_conv.weight[:, :3] = first_conv.weight
            new_conv.weight[:, 3:] = first_conv.weight

        # replace conv1
        base_model.conv1 = new_conv

        #
        self.cnn = nn.Sequential(*list(base_model.children())[:-1])

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

    def set_trainable_blocks(self, num_unfrozen_blocks=None):
        """
        Freeze/unfreeze the ResNet18 backbone (self.cnn) by top-level block,
        for controlled fine-tuning-depth ablations comparable to
        MobileNetEncoder.set_trainable_blocks() in models/mobile_net.py.

        The original codebase fully unfroze ResNet18 for CNN_LSTM but only
        unfroze MobileNetV2's last 4 blocks for MobileNetTransformer
        (see train/train_cnn.ipynb vs train/train_mobile_net.ipynb) -- an
        unmatched comparison. This method lets both encoders be fine-tuned to
        the same controllable depth.

        num_unfrozen_blocks:
            None -> unfreeze everything (full fine-tune; this codebase's
                    original CNN_LSTM behaviour).
            0    -> freeze the entire backbone (pure feature extractor).
            N>0  -> unfreeze only the last N top-level children of self.cnn
                    (children order: conv1, bn1, relu, maxpool, layer1,
                    layer2, layer3, layer4, avgpool -- 9 total), keeping
                    earlier ones frozen. Mirrors
                    model.encoder.features[-N:] on the MobileNet side.

        self.fc (the projection head) is always left trainable, since it has
        no pretrained weights of its own.

        Returns the total number of top-level blocks (for reference/logging).
        """
        blocks = list(self.cnn.children())
        total = len(blocks)

        if num_unfrozen_blocks is None or num_unfrozen_blocks >= total:
            for p in self.cnn.parameters():
                p.requires_grad = True
            return total

        if num_unfrozen_blocks < 0:
            raise ValueError("num_unfrozen_blocks must be >= 0 or None")

        for p in self.cnn.parameters():
            p.requires_grad = False

        if num_unfrozen_blocks > 0:
            for block in blocks[-num_unfrozen_blocks:]:
                for p in block.parameters():
                    p.requires_grad = True

        return total

    @property
    def num_blocks(self):
        return len(list(self.cnn.children()))

# ATTENTION

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

# MAIN MODEL

class CNN_LSTM(nn.Module):
    def __init__(self, num_classes, hidden_dim=256):
        super().__init__()

        self.encoder = CNN_Encoder()

        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.5
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