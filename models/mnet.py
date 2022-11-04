import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

"""
Reference: 

- Inception-V2: https://towardsdatascience.com/a-simple-guide-to-the-versions-of-the-inception-network-7fc52b863202
- https://arxiv.org/pdf/1512.00567.pdf

"""

class MainNet(nn.Module):
    def __init__(self, num_classes: int = 26, feat_out: bool = False) -> None:
        super().__init__()
        self.conv2d_7x7 = ConvBlock(in_channels=3, out_channels=32, kernel_size=7, stride=2, padding=3)
        self.conv2d_1x1 = ConvBlock(in_channels=32, out_channels=32, kernel_size=1)
        self.conv2d_3x3 = ConvBlock(in_channels=32, out_channels=96, kernel_size=3, padding=1)

        self.inception1 = InceptionBlock1()
        self.inception2 = InceptionBlock2()
        self.inception3 = InceptionBlock3()

        self.fc = nn.Linear(in_features=512, out_features=num_classes)
        self.feat_out = feat_out

    def forward(self, x) -> Tensor:
        x = self.conv2d_7x7(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.conv2d_1x1(x)
        x = self.conv2d_3x3(x)
        input_inception = F.max_pool2d(x, kernel_size=3, stride=2)
        inception1_output = self.inception1(input_inception)
        inception2_output = self.inception2(inception1_output)
        inception3_output = self.inception3(inception2_output)

        fc_input = F.avg_pool2d(inception3_output, kernel_size=9, stride=1)
        fc_input = F.dropout(x, training=self.training)
        fc_input = fc_input.view(fc_input.size(0), -1)
        output = self.fc(fc_input)
        if self.feat_out:
            return input_inception, inception1_output, inception2_output, inception3_output
        return output

class InceptionA(nn.Module):
    def __init__(self, 
                in_channels, 
                base_1x1_out,
                base_3x3_1_out,
                base_3x3_2_out,
                base_3x3dbl_1_out,
                base_3x3dbl_2_out,
                base_3x3dbl_3_out,
                base_pool_out) -> None:
        
        # Note: You can refer to Inception-V2 architecture from the papers that I mentioned above.

        super().__init__()
        self.branch1x1 = ConvBlock(in_channels, base_1x1_out, kernel_size=1)
        self.branch3x3_1 = ConvBlock(in_channels, base_3x3_1_out, kernel_size=1)
        self.branch3x3_2 = ConvBlock(base_3x3_1_out, base_3x3_2_out, kernel_size=3, padding=1)
        self.branch3x3dbl_1 = ConvBlock(in_channels, base_3x3dbl_1_out, kernel_size=1)
        self.branch3x3dbl_2 = ConvBlock(base_3x3dbl_1_out, base_3x3dbl_2_out, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = ConvBlock(base_3x3dbl_2_out, base_3x3dbl_3_out, kernel_size=3, padding=1)
        self.branch_pool = ConvBlock(in_channels, base_pool_out, kernel_size=1)

    def forward(self, x) -> Tensor:
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

class InceptionB(nn.Module):
    def __init__(self,
                in_channels,
                base_1x1_1_out,
                base_3x3_1_out,
                base_3x3dbl_1_out,
                base_3x3dbl_2_out,
                base_3x3dbl_3_out) -> None:
        super().__init__()
        self.branch3x3_1 = ConvBlock(in_channels, base_1x1_1_out, kernel_size=1)
        self.branch3x3_2 = ConvBlock(base_1x1_1_out, base_3x3_1_out, kernel_size=3, stride=2, padding=1)

        self.branch3x3dbl_1 = ConvBlock(in_channels, base_3x3dbl_1_out, kernel_size=1)
        self.branch3x3dbl_2 = ConvBlock(base_3x3dbl_1_out, base_3x3dbl_2_out, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = ConvBlock(base_3x3dbl_2_out, base_3x3dbl_3_out, kernel_size=3, stride=2, padding=1)

    def forward(self, x) -> Tensor:
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs,dim=1)

class InceptionBlock1(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.module1 = InceptionA(in_channels=96, base_1x1_out=32, base_3x3_1_out=32, base_3x3_2_out=32,
                                base_3x3dbl_1_out=32, base_3x3dbl_2_out=48, base_3x3dbl_3_out=48, base_pool_out=16)
        self.module2 = InceptionB(in_channels=128, base_1x1_1_out=64, base_3x3_1_out=80,
                                base_3x3dbl_1_out=32, base_3x3dbl_2_out=48, base_3x3dbl_3_out=48)

    def forward(self, x) -> Tensor:
        x = self.module1(x)
        x = self.module2(x)
        return x

class InceptionBlock2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.module1 = InceptionA(in_channels=256, base_1x1_out=112, base_3x3_1_out=32, base_3x3_2_out=48,
                                base_3x3dbl_1_out=48, base_3x3dbl_2_out=64, base_3x3dbl_3_out=64, base_pool_out=64)
        self.module2 = InceptionB(in_channels=288, base_1x1_1_out=64, base_3x3_1_out=86,
                                base_3x3dbl_1_out=96, base_3x3dbl_2_out=128, base_3x3dbl_3_out=128)

    def forward(self, x) -> Tensor:
        x = self.module1(x)
        x = self.module2(x)
        return x

class InceptionBlock3(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.module1 = InceptionA(in_channels=502, base_1x1_out=176, base_3x3_1_out=96, base_3x3_2_out=160,
                                base_3x3dbl_1_out=80, base_3x3dbl_2_out=112, base_3x3dbl_3_out=112, base_pool_out=64)
        self.module2 = InceptionA(in_channels=512, base_1x1_out=176, base_3x3_1_out=96, base_3x3_2_out=160,
                                base_3x3dbl_1_out=96, base_3x3dbl_2_out=112, base_3x3dbl_3_out=112, base_pool_out=64)

    def forward(self, x) -> Tensor:
        x = self.module1(x)
        x = self.module2(x)
        return x

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