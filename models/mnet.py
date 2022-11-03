import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

from torch import Tensor

class MainNet(nn.Module):
    def __init__(self, num_classes: int = 26, feat_out: bool = False) -> None:
        super().__init__()
        
    def forward(self, x) -> Tensor:
        pass