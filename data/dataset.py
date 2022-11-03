import os
import torch
from torch.utils.data import Dataset

class PA100KDataset(Dataset):
    def __init__(self, root: str, label, transform = None) -> None:
        super().__init__()