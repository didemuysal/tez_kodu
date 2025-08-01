import os
import h5py
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Tuple
from torchvision import transforms

class BrainTumourDataset(Dataset): 
    def __init__(self, data_folder: str, filenames: List[str], labels: List[int], is_train: bool = True): 
        self.data_folder = data_folder 
        self.filenames = filenames 
        self.labels = labels 
        self.is_train = is_train 
        
        if self.is_train:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomAffine(degrees=7, translate=(0.05, 0.05), scale=(0.9, 1.1), shear=20),
                transforms.ColorJitter(brightness=0.05, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self) -> int: 
        return len(self.filenames)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]: 
        filepath = os.path.join(self.data_folder, self.filenames[index])

        with h5py.File(filepath, "r") as f:
            image = f["cjdata"]["image"][()]
        image = image.astype(np.float32)
        image /= image.max()
        image_tensor = self.transform(image)
        label = torch.tensor(self.labels[index] - 1, dtype=torch.long)
        return image_tensor, label
