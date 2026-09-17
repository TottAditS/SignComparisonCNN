import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import torchvision.transforms as transforms

# ImageNet normalization stats used by get_transforms() below. Kept as module
# constants so VideoDataset can invert Normalize() when motion_mode=
# "optical_flow" needs true pixel intensities (see _to_gray_uint8).
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class VideoDataset(Dataset):
    def __init__(self, root_dir, seq_len=20, transform=None, motion_mode="diff"):
        """
        motion_mode : {"diff", "optical_flow"}
            How the extra 3 channels (concatenated after the 3 RGB channels,
            giving the 6-channel input both models expect) are computed:
              - "diff" (default, original behaviour): per-pixel difference
                between the current and previous NORMALIZED RGB frame. Cheap,
                but computed post-Normalize, so it represents a difference of
                normalized pixel values rather than a true motion signal.
              - "optical_flow": dense Farneback optical flow (OpenCV) between
                the current and previous frame, computed on true pixel
                intensities (Normalize is inverted first -- see
                _to_gray_uint8), encoded as 3 channels: horizontal flow,
                vertical flow, and magnitude. This is Novelty #4 for the
                paper: it tests whether a classic, still-cheap motion
                representation improves on the naive frame-difference channel
                without requiring a learned optical-flow network (e.g. RAFT).
        """
        if motion_mode not in ("diff", "optical_flow"):
            raise ValueError(f"motion_mode must be 'diff' or 'optical_flow', got {motion_mode!r}")

        self.root_dir = root_dir
        self.seq_len = seq_len
        self.transform = transform
        self.motion_mode = motion_mode

        self.samples = []
        self.labels = []
        self.label_map = {}
        # Cache of sorted frame filenames per sample, populated once here in
        # _load_dataset() instead of being re-listed from disk on every single
        # __getitem__() call (which used to run os.listdir(vid_path) again for
        # every sample, every epoch, every worker -- a pure I/O bottleneck with
        # no effect on training results, since the directory contents never
        # change between epochs). This is a speed-only fix: __getitem__ below
        # now reuses self.frame_lists[idx] instead of re-scanning the disk.
        self.frame_lists = []

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

                # test apakah ada frame (dan cache hasilnya, dipakai lagi di __getitem__)
                frames = sorted(f for f in os.listdir(vid_path) if f.endswith(".jpg"))
                if len(frames) == 0:
                    continue

                self.samples.append(vid_path)
                self.labels.append(self.label_map[label])
                self.frame_lists.append(frames)

    def __len__(self):
        return len(self.samples)

    # Maximum expected pixel displacement (in pixels) between consecutive
    # frames, used to rescale optical flow into roughly [-1, 1] (and flow
    # magnitude into roughly [0, 1]) -- a comparable dynamic range to the
    # ImageNet-normalized RGB channels, rather than raw unbounded pixel
    # displacements. 20px is a reasonable cap for hand/arm motion between
    # frames sampled from a ~20-frame-per-clip BISINDO video; tune if needed.
    FLOW_MAX_DISPLACEMENT = 20.0

    def _to_gray_uint8(self, img_normalized):
        """
        Converts a (3, H, W) tensor produced by the Normalize(IMAGENET_MEAN,
        IMAGENET_STD) step in get_transforms() back into an OpenCV-compatible
        uint8 grayscale numpy array, for optical flow computation on true
        pixel intensities rather than normalized values.

        ASSUMPTION: this dataset is only ever used with transforms built by
        get_transforms() in this module, whose pipelines both end with
        Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD). If a custom
        transform without that final Normalize step is passed in, this
        inversion will be wrong; motion_mode="diff" has no such requirement.
        """
        img_unnorm = img_normalized * IMAGENET_STD + IMAGENET_MEAN
        img_unnorm = img_unnorm.clamp(0.0, 1.0)
        img_np = (img_unnorm.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)  # (H, W, 3) RGB
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        return gray

    def _compute_optical_flow_channels(self, prev_gray, curr_gray):
        """
        Dense Farneback optical flow between two consecutive grayscale
        frames, encoded as 3 channels: normalized horizontal flow, normalized
        vertical flow, and normalized magnitude. Using classic Farneback
        (built into OpenCV, already a project dependency) keeps this ablation
        free of any additional model dependency (e.g. a learned flow network
        such as RAFT), matching the project's existing "cheap, no extra
        training" motion-cue philosophy while replacing the naive frame
        difference with an actual motion estimate.
        """
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )  # (H, W, 2) float32, in pixels

        flow_x = np.clip(flow[..., 0] / self.FLOW_MAX_DISPLACEMENT, -1.0, 1.0)
        flow_y = np.clip(flow[..., 1] / self.FLOW_MAX_DISPLACEMENT, -1.0, 1.0)
        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2) / self.FLOW_MAX_DISPLACEMENT
        magnitude = np.clip(magnitude, 0.0, 1.0)

        motion = np.stack([flow_x, flow_y, magnitude], axis=0).astype(np.float32)  # (3, H, W)
        return torch.from_numpy(motion)

    def __getitem__(self, idx):
        vid_path = self.samples[idx]
        label = self.labels[idx]

        # Reuses the sorted frame list cached in _load_dataset() instead of
        # re-running os.listdir(vid_path) on every access (speed-only change,
        # identical result since folder contents don't change across epochs).
        frame_list = self.frame_lists[idx]

        # ===== FRAME SAMPLING =====
        if len(frame_list) >= self.seq_len:
            indices = torch.linspace(0, len(frame_list) - 1, self.seq_len).long()
            frames = [frame_list[i] for i in indices]
        else:
            frames = frame_list

        imgs = []
        prev_img = None    # previous NORMALIZED RGB tensor, used when motion_mode == "diff"
        prev_gray = None   # previous uint8 grayscale frame, used when motion_mode == "optical_flow"

        # seed augment konsisten per sequence
        seed = torch.randint(0, 10000, (1,)).item()

        for frame in frames:
            img_path = os.path.join(vid_path, frame)

            try:
                img = Image.open(img_path).convert("RGB")
            except:
                continue

            if self.transform:
                torch.manual_seed(seed)
                img = self.transform(img)  # (3, H, W)

            # ===== MOTION CHANNELS (diff or optical flow) =====
            if self.motion_mode == "diff":
                if prev_img is None:
                    motion = torch.zeros_like(img)
                else:
                    motion = img - prev_img
                prev_img = img
            else:  # "optical_flow"
                curr_gray = self._to_gray_uint8(img)
                if prev_gray is None:
                    motion = torch.zeros(3, img.shape[1], img.shape[2])
                else:
                    motion = self._compute_optical_flow_channels(prev_gray, curr_gray)
                prev_gray = curr_gray

            # 6 channel
            img_6ch = torch.cat([img, motion], dim=0)  # (6, H, W)

            imgs.append(img_6ch)

        # ===== HANDLE FRAME KOSONG =====
        if len(imgs) == 0:
            dummy = torch.zeros(6, 224, 224)
            imgs = [dummy for _ in range(self.seq_len)]

        # ===== PADDING =====
        while len(imgs) < self.seq_len:
            imgs.append(imgs[-1])

        imgs = torch.stack(imgs)  # (T, 6, H, W)

        return imgs, label



# Augmentasi Pre Process Dataset

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



# DATALOADER + IMBALANCE HANDLING

def get_dataloader(
    root_dir,
    batch_size=8,
    transform=None,
    use_weighted_sampler=False,
    seq_len=20,
    motion_mode="diff",
    num_workers=4
):
    dataset = VideoDataset(root_dir, seq_len=seq_len, transform=transform, motion_mode=motion_mode)

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
            num_workers=num_workers,
            pin_memory=True,
            # Speed-only: keeps worker processes alive across epochs instead
            # of respawning them every epoch (respawn overhead is significant
            # once num_workers > 0). No effect on training results.
            persistent_workers=(num_workers > 0),
            # Speed-only: lets each worker stay a couple batches ahead of the
            # GPU instead of the default 2, smoothing out I/O spikes. No
            # effect on training results.
            prefetch_factor=4 if num_workers > 0 else None,
        )

    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
            prefetch_factor=4 if num_workers > 0 else None,
        )

    return loader, dataset

# CLASS WEIGHT (FOR LOSS)

def get_class_weights(dataset, device):
    labels = dataset.labels
    class_counts = np.bincount(labels)

    weights = 1.0 / class_counts
    weights = weights / weights.sum() * len(weights)

    weights = torch.tensor(weights, dtype=torch.float).to(device)

    print("Class weights:", weights)

    return weights