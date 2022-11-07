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
        # For better explnation of input size from each feature, see MainNet implementation.
        feature0, feature1, feature2, feature3 = self.main_net(x)
        if self.af_index == 1:
            att = self.att(feature1)
            att1 = F.upsample(att, scale_factor=2)
            att2 = att
            att3 = F.avg_pool2d(att, kernel_size=2, stride=2)
        elif self.af_index == 2:
            att = self.att(feature2)
            att2 = F.upsample(att, scale_factor=2)
            att1 = F.upsample(att2, scale_factor=2)
            att3 = att
        elif self.af_index == 3:
            att = self.att(feature3)
            att2 = F.upsample(att, scale_factor=2)
            att1 = F.upsample(att2, scale_factor=2)
            att3 = att
        else:
            raise ValueError

        # Attention 1
        att1_width, att1_height = att1.size()[2], att1.size()[3]
        for i in range(self.att_channel):
            temp = att1[:, i].clone()
            temp = temp.view(-1, 1, att1_width, att1_height).expand(-1, 96, att1_width, att1_height)
            att_feature0 = feature0 * temp
            att_feature3 = self.att_branch1(att_feature0)
            if i == 0:
                ret = att_feature3
            else:
                ret = torch.cat((ret, att_feature3), dim=1)
        
        # Attention 2
        att2_width, att2_height = att2.size()[2], att2.size()[3]
        for i in range(self.att_channel):
            temp = att2[:, i].clone()
            temp = temp.view(-1, 1, att2_width, att2_height).expand(-1, 256, att2_width, att2_height)
            att_feature1 = feature1 * temp
            att_feature3 = self.att_branch2(att_feature1)
            ret = torch.cat((ret, att_feature3), dim=1)

        # Attention 3
        att3_width, att3_height = att3.size([2]), att3.size()[3]
        for i in range(self.att_channel):
            temp = att3[:, i].clone()
            temp = temp.view(-1, 1, att3_width, att3_height).expand(-1, 502, att3_width, att3_height)
            att_feature2 = feature2 * temp
            att_feature3 = self.att_branch3(att_feature2)
            ret = torch.cat((ret, att_feature3), dim=1)
        
        gap_output = F.avg_pool2d(ret, kernel_size=9, stride=1)
        fc_input = F.dropout(gap_output, training=self.training)
        fc_input = fc_input.view(fc_input.size(0), -1)
        pred_class = self.fc(fc_input)
        if self.att_out:
            if self.feat_out:
                return ret, att.cpu()
            else:
                return pred_class, att.cpu()
        else:
            if self.feat_out:
                return ret
            else:
                return pred_class

    def load_att_branch_weight(self):
        branch1_incept1 = {"0." + k: v for k, v in self.main_net.inception1.state_dict().items()}
        branch1_incept2 = {"1." + k: v for k, v in self.main_net.inception2.state_dict().items()}
        branch1_incept3 = {"2." + k: v for k, v in self.main_net.inception3.state_dict().items()}
        branch1_incept1.update(branch1_incept2)
        branch1_incept1.update(branch1_incept3)

        self.att_branch1.load_state_dict(branch1_incept1)

        branch2_incept2 = {"0." + k: v for k, v in self.main_net.inception2.state_dict().items()}
        branch2_incept3 = {"1." + k: v for k, v in self.main_net.inception3.state_dict().items()}

        branch2_incept2.update(branch2_incept3)
        self.att_branch2.load_state_dict(branch2_incept2)

        # Incept3
        self.att_branch3.load_state_dict(self.main_net.inception3.state_dict())