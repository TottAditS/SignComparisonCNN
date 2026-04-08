import os
import torch
from torch.utils.data import Dataset, DataLoader
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

        label_names = sorted(os.listdir(self.root_dir))
        self.label_map = {label: idx for idx, label in enumerate(label_names)}

        for label in label_names:
            label_path = os.path.join(self.root_dir, label)

            for vid in os.listdir(label_path):
                vid_path = os.path.join(label_path, vid)

                if os.path.isdir(vid_path):
                    self.samples.append(vid_path)
                    self.labels.append(self.label_map[label])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vid_path = self.samples[idx]
        label = self.labels[idx]

        frames = sorted(os.listdir(vid_path))[:self.seq_len]

        imgs = []
        for frame in frames:
            img_path = os.path.join(vid_path, frame)
            img = Image.open(img_path).convert("RGB")

            if self.transform:
                img = self.transform(img)

            imgs.append(img)

        # padding kalau frame kurang
        while len(imgs) < self.seq_len:
            imgs.append(imgs[-1])

        imgs = torch.stack(imgs)  # (seq_len, C, H, W)

        return imgs, label


def get_dataloader(root_dir, batch_size=8, shuffle=True):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = VideoDataset(root_dir, transform=transform)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)