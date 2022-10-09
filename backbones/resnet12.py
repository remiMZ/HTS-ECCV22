'''
https://github.com/cyvius96/few-shot-meta-baseline/blob/master/models/resnet12.py
'''
import torch.nn as nn

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, 
                    padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, 
                    bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.relu = nn.LeakyReLU(0.1)
        self.conv1 = conv3x3(in_planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)

        self.downsample = downsample    # residual branch downsample block
        self.stride = stride

        self.maxpool = nn.MaxPool2d(2)

    def forward(self, inputs):
        identity = inputs

        outputs = self.conv1(inputs)
        outputs = self.bn1(outputs)
        outputs = self.relu(outputs)

        outputs = self.conv2(outputs)
        outputs = self.bn2(outputs)
        outputs = self.relu(outputs)

        outputs = self.conv3(outputs)
        outputs = self.bn3(outputs)

        if self.downsample is not None:
            identity = self.downsample(inputs)

        outputs += identity
        outputs = self.relu(outputs)

        outputs = self.maxpool(outputs)

        return outputs

class ResNet(nn.Module):

    def __init__(self, block=BasicBlock, n_channels=[64, 128, 256, 512], zero_init_residual=False):
        super(ResNet, self).__init__()
        self.in_planes = 3

        self.layer1 = self._make_layer(block, n_channels[0])
        self.layer2 = self._make_layer(block, n_channels[1])
        self.layer3 = self._make_layer(block, n_channels[2])
        self.layer4 = self._make_layer(block, n_channels[3])
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self._init_conv()

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)


    def _make_layer(self, block, planes, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_planes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def _init_conv(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs):

        outputs = self.layer1(inputs)
        outputs = self.layer2(outputs)
        outputs = self.layer3(outputs)
        outputs = self.layer4(outputs)

        # outputs = self.avgpool(outputs)

        return outputs

def resnet12(**kwargs):
    return ResNet(BasicBlock, n_channels=[64, 128, 256, 512], **kwargs)

def resnet12_wide(**kwargs):
    return ResNet(BasicBlock, n_channels=[64, 160, 320, 640], **kwargs)

