import torch
import torch.nn as nn
import torch.nn.functional as F
from models.afnet import *
from models.mnet import *

class HydraPlusNet(nn.Module):
    def __init__(self, num_classes: int = 26, att_out: bool = False) -> None:
        super().__init__()
        self.att_out = att_out
        with torch.no_grad():
            if self.att_out:
                self.main_net = MainNet(feat_out=True)
                self.af1 = AFNet(att_out=True, feat_out=True, af_name="AF1")
                self.af2 = AFNet(att_out=True, feat_out=True, af_name="AF2")
                self.af3 = AFNet(att_out=True, feat_out=True, af_name="AF3")
            else:
                self.main_net = MainNet(feat_out=True)
                self.af1 = AFNet(feat_out=True, af_name="AF1")
                self.af2 = AFNet(feat_out=True, af_name="AF2")
                self.af3 = AFNet(feat_out=True, af_name="AF3")
        self.fc = nn.Linear(512 * 73, num_classes)

    def forward(self, x):
        _, _, _, feature0 = self.main_net(x)
        if self.att_out:
            feature1, att1 = self.af1(x)
            feature2, att2 = self.af2(x)
            feature3, att3 = self.af3(x)
        else:
            feature1 = self.af1(x)
            feature2 = self.af2(x)
            feature3 = self.af3(x)

        ret = torch.cat((feature0, feature1, feature2, feature3), dim=1)
        fc_input = F.avg_pool2d(ret, kernel_size=9, stride=1)
        fc_input = F.dropout(fc_input, training=self.training)
        fc_input = fc_input.view(fc_input.size(0), -1)
        ret = self.fc(fc_input)
        if self.att_out:
            return att1, att2, att3, ret
        return ret