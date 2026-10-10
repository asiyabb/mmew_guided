import os
import glob
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import GroupShuffleSplit
from torchvision import transforms
from config.config import Config

class MMEWSequenceDataset(Dataset):
    def __init__(self, mode="macro", transform=None):
        self.mode = mode
        self.samples = []
        self.subjects = []  # Track subject IDs for subject-grouped splits
        self.transform = transform or self.get_val_transforms()

        if mode == "macro":
            self._load_macro_samples()
        elif mode == "micro":
            self._load_micro_samples()

    @staticmethod
    def get_train_transforms():
        """Spatial Data Augmentations for Training."""
        return transforms.Compose([
            transforms.Resize(Config.IMAGE_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.2))
        ])

    @staticmethod
    def get_val_transforms():
        """Standard Evaluation Transforms."""
        return transforms.Compose([
            transforms.Resize(Config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _load_macro_samples(self):
        for subject in sorted(os.listdir(Config.MACRO_DIR)):
            sub_path = os.path.join(Config.MACRO_DIR, subject)
            if not os.path.isdir(sub_path):
                continue
            for emotion in Config.EMOTIONS:
                emo_path = os.path.join(sub_path, emotion)
                if os.path.isdir(emo_path):
                    frames = sorted(glob.glob(os.path.join(emo_path, "*.jpg")))
                    if len(frames) > 0:
                        self.samples.append((frames, Config.LABEL_MAP[emotion]))
                        self.subjects.append(subject)

    def _load_micro_samples(self):
        df = pd.read_excel(Config.EXCEL_ANNOTATIONS) if os.path.exists(Config.EXCEL_ANNOTATIONS) else None
        for emotion in Config.EMOTIONS:
            emo_path = os.path.join(Config.MICRO_DIR, emotion)
            if not os.path.isdir(emo_path):
                continue
            for seq_folder in sorted(os.listdir(emo_path)):
                seq_path = os.path.join(emo_path, seq_folder)
                if os.path.isdir(seq_path):
                    frames = sorted(glob.glob(os.path.join(seq_path, "*.jpg")))
                    if len(frames) > 0:
                        subject = seq_folder.split("_")[0]
                        self.samples.append((frames, Config.LABEL_MAP[emotion]))
                        self.subjects.append(subject)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths, label = self.samples[idx]
        indices = torch.linspace(0, len(frame_paths) - 1, Config.SEQ_LEN).long()
        sampled_paths = [frame_paths[i] for i in indices]

        tensors = []
        for path in sampled_paths:
            img = Image.open(path).convert("RGB")
            tensors.append(self.transform(img))

        video_tensor = torch.stack(tensors)  # Shape: [SEQ_LEN, C, H, W]
        return video_tensor, torch.tensor(label, dtype=torch.long)


class AugmentedSubset(Dataset):
    """Applies dataset transformations per subset."""
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        frame_paths, label = self.subset.dataset.samples[self.subset.indices[idx]]
        indices = torch.linspace(0, len(frame_paths) - 1, Config.SEQ_LEN).long()
        sampled_paths = [frame_paths[i] for i in indices]

        tensors = []
        for path in sampled_paths:
            img = Image.open(path).convert("RGB")
            tensors.append(self.transform(img))

        video_tensor = torch.stack(tensors)
        return video_tensor, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.subset)


def get_three_way_subject_splits(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Splits subjects strictly into Train, Validation, and Test sets with ZERO data leakage.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Ratios must sum to 1.0"
    
    # Split 1: Separate Test set from Train+Val
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    train_val_idx, test_idx = next(gss_test.split(dataset.samples, groups=dataset.subjects))
    
    train_val_subjects = [dataset.subjects[i] for i in train_val_idx]
    train_val_samples = [dataset.samples[i] for i in train_val_idx]
    
    # Split 2: Separate Train and Validation
    relative_val_ratio = val_ratio / (train_ratio + val_ratio)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=relative_val_ratio, random_state=seed)
    rel_train_idx, rel_val_idx = next(gss_val.split(train_val_samples, groups=train_val_subjects))
    
    train_idx = [train_val_idx[i] for i in rel_train_idx]
    val_idx = [train_val_idx[i] for i in rel_val_idx]

    raw_train = Subset(dataset, train_idx)
    raw_val = Subset(dataset, val_idx)
    raw_test = Subset(dataset, test_idx)

    train_dataset = AugmentedSubset(raw_train, transform=MMEWSequenceDataset.get_train_transforms())
    val_dataset = AugmentedSubset(raw_val, transform=MMEWSequenceDataset.get_val_transforms())
    test_dataset = AugmentedSubset(raw_test, transform=MMEWSequenceDataset.get_val_transforms())

    return train_dataset, val_dataset, test_dataset