import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import torchvision.transforms as transforms


class VideoDataset(Dataset):
    def __init__(self, root_dir, seq_len=20, transform=None):
        self.root_dir = root_dir
        self.seq_len = seq_len
        self.transform = transform

        self.samples = []
        self.labels = []
        self.label_map = {}

        self._load_dataset()

    def _load_dataset(self):
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"Path not found: {self.root_dir}")

        label_names = sorted([
            d for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d))
        ])

        self.label_map = {label: idx for idx, label in enumerate(label_names)}

        for label in label_names:
            label_path = os.path.join(self.root_dir, label)

            for vid in os.listdir(label_path):
                vid_path = os.path.join(label_path, vid)

                if not os.path.isdir(vid_path):
                    continue

                # pastikan ada frame di dalamnya
                frames = [f for f in os.listdir(vid_path) if f.endswith(".jpg")]
                if len(frames) == 0:
                    continue

                self.samples.append(vid_path)
                self.labels.append(self.label_map[label])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vid_path = self.samples[idx]
        label = self.labels[idx]

        frame_list = sorted([
            f for f in os.listdir(vid_path)
            if f.endswith(".jpg")
        ])

        # =========================
        # FRAME SAMPLING (FIXED)
        # =========================
        if len(frame_list) >= self.seq_len:
            indices = torch.linspace(0, len(frame_list) - 1, self.seq_len).long()
            frames = [frame_list[i] for i in indices]
        else:
            frames = frame_list

        imgs = []

        # =========================
        # SEQUENCE CONSISTENT AUGMENT
        # =========================
        seed = torch.randint(0, 10000, (1,)).item()

        for frame in frames:
            img_path = os.path.join(vid_path, frame)

            try:
                img = Image.open(img_path).convert("RGB")
            except:
                continue

            if self.transform:
                torch.manual_seed(seed)
                img = self.transform(img)

            imgs.append(img)

        # =========================
        # PADDING
        # =========================
        while len(imgs) < self.seq_len:
            imgs.append(imgs[-1])

        imgs = torch.stack(imgs)  # (T, C, H, W)

        return imgs, label


# =========================
# TRANSFORMS
# =========================
def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),

        transforms.RandomAffine(
            degrees=10,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05)
        ),

        transforms.ColorJitter(brightness=0.2, contrast=0.2),

        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_transform, val_transform


# =========================
# DATALOADER + IMBALANCE HANDLING
# =========================
def get_dataloader(
    root_dir,
    batch_size=8,
    transform=None,
    use_weighted_sampler=False
):
    dataset = VideoDataset(root_dir, transform=transform)

    if use_weighted_sampler:
        labels = dataset.labels
        class_counts = np.bincount(labels)

        print("Class counts:", class_counts)

        # inverse frequency
        class_weights = 1.0 / class_counts

        # weight per sample
        sample_weights = [class_weights[label] for label in labels]

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=4,
            pin_memory=True
        )

    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

    return loader, dataset


# =========================
# CLASS WEIGHT (FOR LOSS)
# =========================
def get_class_weights(dataset, device):
    labels = dataset.labels
    class_counts = np.bincount(labels)

    weights = 1.0 / class_counts
    weights = weights / weights.sum() * len(weights)

    weights = torch.tensor(weights, dtype=torch.float).to(device)

    print("Class weights:", weights)

    return weights