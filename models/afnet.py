from mnet import *

class AFNet(nn.Module):
    def __init__(self, num_classes=26, att_out=False, feat_out=False, af_name="AF2") -> None:
        super().__init__()
        with torch.inference_mode():
            self.main_net = MainNet(feat_out=True)
        self.att_out = att_out
        self.feat_out = feat_out
        self.att_channel = 8

        self.af_index = int(af_name[2])
        if self.af_index == 1:
            self.att = ConvBlock(256, self.att_channel, kernel_size=1)
        elif self.af_index == 2:
            self.att = ConvBlock(502, self.att_channel, kernel_size=1)
        elif self.af_index == 3:
            self.att = ConvBlock(512, self.att_channel, kernel_size=1)
        else:
            raise ValueError

        self.att_branch1 = nn.Sequential(
            InceptionBlock1(),
            InceptionBlock2(),
            InceptionBlock3()
        )
        self.att_branch2 = nn.Sequential(
            InceptionBlock2(),
            InceptionBlock3()
        )
        self.att_branch3 = InceptionBlock3()
        self.fc = nn.Linear(512 * 3 * self.att_channel, num_classes)

    def forward(self, x):
        feature0, feature1, feature2, feature3 = self.main_net(x)
        if self.af_index == 1:
            att = self.att(feature1)
            att1 = F.upsample(att, scale_factor=2)
            att2 = att
            att3 = F.avg_pool2d(att, kernel_size=2, stride=2)
        elif self.af_index == 2:
            att = self.att(feature2)
        elif self.af_index == 3:
            pass
        else:
            raise ValueError
        