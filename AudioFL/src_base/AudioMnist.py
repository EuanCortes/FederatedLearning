import torch
from torchvision.datasets import VisionDataset
import os
import h5py
import numpy as np
from typing import Tuple, Optional, Callable


class AudioMNIST(VisionDataset):
    """AudioMNIST Dataset using preprocessed HDF5 files

    Args:
        root: Root directory containing preprocessed_data folder
        model_type: Either 'AlexNet' or 'AudioNet'
        task: Either 'digit' or 'gender'
        split: Cross-validation split number (0-4 for digit, 0-3 for gender)
        mode: 'train', 'validate', or 'test'
        transform: A function/transform for the data
        target_transform: A function/transform for the labels
    """

    def __init__(
        self,
        root: str,
        model_type: str = "AudioNet",
        task: str = "digit",
        split: int = 0,
        mode: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        super().__init__(root, transform=transform, target_transform=target_transform)

        assert model_type in [
            "AlexNet",
            "AudioNet",
        ], "model_type must be 'AlexNet' or 'AudioNet'"
        assert task in ["digit", "gender"], "task must be 'digit' or 'gender'"
        assert mode in [
            "train",
            "validate",
            "test",
        ], "mode must be 'train', 'validate', or 'test'"

        self.model_type = model_type
        self.task = task
        self.split = split
        self.mode = mode

        # Load file paths from split txt file
        self.file_paths = self._load_split_files()

    def _load_split_files(self):
        """Load file paths from the split txt files created by preprocess.py"""
        split_file = os.path.join(
            self.root, f"{self.model_type}_{self.task}_{self.split}_{self.mode}.txt"
        )

        if not os.path.exists(split_file):
            raise FileNotFoundError(
                f"Split file not found: {split_file}\n"
                f"Make sure you've run the preprocess file"
            )

        with open(split_file, "r") as f:
            paths = [os.path.normpath(line.strip()) for line in f.readlines()]

        return paths

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            index: Index
        Returns:
            tuple: (data, label) where label is [digit, gender]
        """
        filepath = self.file_paths[index]

        with h5py.File(filepath, "r") as f:
            data = np.array(f["data"])
            label = np.array(f["label"])

        # FIX: Squeeze to remove extra dimensions
        data = data.squeeze()  # [1,1,1,8000] -> [8000]

        # Convert to torch tensors
        data = torch.from_numpy(data).float()
        label = torch.from_numpy(label).long()

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return data, label

    def __len__(self) -> int:
        return len(self.file_paths)
