import math

import torch
from torchvision.models import ResNet
from torchvision import models
from torch import nn
import torch.nn.functional as F


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16, alpha=0.):
        super(SELayer, self).__init__()
        assert alpha >= 0 and alpha <= 1
        self.alpha = alpha
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * (self.alpha * y + (1 - self.alpha))


class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes * 4, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def se_resnet50(num_classes, path_to_model=None):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBottleneck, [3, 4, 6, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    if path_to_model is not None:
        model.load_state_dict(torch.load(path_to_model))
    return model


class SE_Resnet50(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()

        self.bottleneck_conf = [3, 4, 6, 3]
        self.net = ResNet(SEBottleneck, self.bottleneck_conf, num_classes=1000)
        self.net.avgpool = nn.AdaptiveAvgPool2d(1)
        self.net.load_state_dict(models.resnet50(pretrained=True).state_dict(), strict=False)

        self.fc = nn.Sequential(
            nn.Linear(self.net.fc.in_features + 1, 512),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        self.net.fc = nn.Dropout(0.0)

    def set_SE_alpha(self, alpha):
        assert alpha >= 0 and alpha <= 1
        for idx, block in enumerate([self.net.layer1, self.net.layer2, self.net.layer3, self.net.layer4]):
            for i in range(self.bottleneck_conf[idx]):
                block[i].se.alpha = alpha

    def forward(self, x, O):
        out = self.net(x)
        out = out.view(out.size(0), -1)
        out = torch.cat([out, O], 1)
        return F.softmax(self.fc(out), dim=1)
