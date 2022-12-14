import os
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from typing import Tuple, List
from PIL import Image

def tensor_transforms() -> transforms.Compose:
    custom_transform = [
        transforms.Resize((299, 299)),
        transforms.ToTensor()
    ]
    return transforms.Compose(custom_transform)

class PA100KDataset(Dataset):
    def __init__(self, 
                dataset_path: str,
                partition_path: str,
                split: str = 'train',
                partition_idx: int = 0,
                transform = None, **kwargs) -> None:
        super().__init__()
        if os.path.exists(dataset_path):
            self.dataset = pickle.load(open(dataset_path, "rb"))
        else:
            print(f"{dataset_path} doesn't exist!")
            raise ValueError

        if os.path.exists(partition_path):
            self.partition = pickle.load(open(partition_path, "rb"))
        else:
            print(f"{partition_path} doesn't exist!")
            raise ValueError
        
        if  split not in self.partition:
            print(f"Split {split} does not exist in dataset!")
            raise ValueError
        if partition_idx > len(self.partition[split]) - 1:
            print(f"Partition idx: {partition_idx} is out of range in partition!")
            raise ValueError
    
        self.transform = transform
        self.root = self.dataset['root']
        self.att_name = [self.dataset['att_name'][i] for i in self.dataset['selected_attribute']]
        self.images = []
        self.labels = []
        for idx in self.partition[split][partition_idx]:
            self.images.append(self.dataset["image"][idx])
            label_tmp = np.array(self.dataset['att'][idx])[self.dataset['selected_attribute']].tolist()
            self.labels.append(torch.Tensor(label_tmp))

    def __getitem__(self, index) -> Tuple:
        image, target = self.images[index], self.labels[index]
        image_path = os.path.join(self.dataset["root"], image)
        img = Image.open(image_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, target, image

    def __len__(self) -> int:
        return len(self.images)

def dataloader(dataset_path: str, 
            partition_path: str,
            batch_size: int,
            split: str,
            shuffle: bool = True,
            num_workers: int = 2) -> List:
    dataset = PA100KDataset(
        dataset_path=dataset_path,
        partition_path=partition_path,
        split=split,
        transform=tensor_transforms()
    )
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader