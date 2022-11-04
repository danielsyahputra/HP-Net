import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

from torch import Tensor

"""
Reference: 

- Inception-V2: https://towardsdatascience.com/a-simple-guide-to-the-versions-of-the-inception-network-7fc52b863202
- https://arxiv.org/pdf/1512.00567.pdf

"""

class MainNet(nn.Module):
    def __init__(self, num_classes: int = 26, feat_out: bool = False) -> None:
        super().__init__()
        
    def forward(self, x) -> Tensor:
        pass

class InceptionA(nn.Module):
    def __init__(self, 
                in_channels, 
                base_1x1_out,
                base_3x3_1_out,
                base_3x3_2_out,
                base_5x5_1_out,
                base_5x5_2_out,
                base_5x5_3_out,
                base_pool_out) -> None:
        
        # Note: You can refer to Inception-V2 architecture from the papers that I mentioned above.

        super().__init__()
        self.branch1x1 = ConvBlock(in_channels, base_1x1_out, kernel_size=1)
        self.branch3x3_1 = ConvBlock(in_channels, base_3x3_1_out, kernel_size=1)
        self.branch3x3_2 = ConvBlock(base_3x3_1_out, base_3x3_2_out, kernel_size=3, padding=1)
        self.branch3x3dbl_1 = ConvBlock(in_channels, base_5x5_1_out, kernel_size=1)
        self.branch3x3dbl_2 = ConvBlock(base_5x5_1_out, base_5x5_2_out, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = ConvBlock(base_5x5_2_out, base_5x5_3_out, kernel_size=3, padding=1)
        self.branch_pool = ConvBlock(in_channels, base_pool_out, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        branch_pool = F.avg_pool2d(x, kernel_size=1, stride=1)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, **kwargs),
            nn.BatchNorm2d(out_channels, eps=0.001)
        )
    
    def forward(self, x):
        x = self.block(x)
        return F.relu(x, inplace=True)

class InceptionBlock1(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x):
        pass