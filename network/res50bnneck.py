
import torchvision
import torch.nn as nn

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 1.0)

class Res50BNNeck(nn.Module):

    def __init__(self, part_num):
        super(Res50BNNeck, self).__init__()

        self.part_num = part_num

        resnet = torchvision.models.resnet50(pretrained=True)
        resnet.layer4[0].conv2.stride = (1, 1)
        resnet.layer4[0].downsample[0].stride = (1, 1)
        self.resnet_conv = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.bn = nn.BatchNorm2d(2048)
        self.bn.apply(weights_init_kaiming)

    def forward(self, x):
        features_map = self.resnet_conv(x)
        features = self.gap(features_map)
        local_feature_list = []
        for i in range(self.part_num):
            local_feature_list.append(self.gap(features_map[:, :, i * 4: (i + 1) * 4, :]))
        bn_features = self.bn(features)
        bn_local_feature_list = []
        for i in range(self.part_num):
            bn_local_feature_list.append(self.bn(self.gap(features_map[:, :, i * 4: (i + 1) * 4, :])))

        return features, bn_features, local_feature_list, bn_local_feature_list