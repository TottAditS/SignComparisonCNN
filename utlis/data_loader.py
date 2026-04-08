import torch
from torch.utils.data import Dataset
import os
import cv2
import numpy as np

class VideoDataset(Dataset):
    def __init__(self, root_dir, label_map, target_frames=20):
        self.samples = []
        self.label_map = label_map
        self.target_frames = target_frames

        for label in os.listdir(root_dir):
            label_path = os.path.join(root_dir, label)

            if not os.path.isdir(label_path):
                continue

            for video in os.listdir(label_path):
                video_path = os.path.join(label_path, video)
                self.samples.append((video_path, label_map[label]))

    def __len__(self):
        return len(self.samples)

    def uniform_sample(self, frames):
        if len(frames) == 0:
            return [np.zeros((224,224,3), dtype=np.uint8)] * self.target_frames

        if len(frames) >= self.target_frames:
            idxs = np.linspace(0, len(frames) - 1, self.target_frames).astype(int)
            return [frames[i] for i in idxs]

        result = frames.copy()
        while len(result) < self.target_frames:
            result.append(result[-1])

        return result

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]

        frames = []
        images = sorted(os.listdir(video_path))

        for img in images:
            img_path = os.path.join(video_path, img)
            frame = cv2.imread(img_path)

            if frame is None:
                continue

            frame = frame / 255.0
            frames.append(frame)

        frames = self.uniform_sample(frames)

        frames = np.array(frames)
        frames = torch.from_numpy(frames).permute(0,3,1,2).float()

        return frames, label