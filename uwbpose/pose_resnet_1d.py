# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------
# SSH

import torch
import torch.nn as nn
import torch.nn.functional as F

BN_MOMENTUM = 0.1
INPUT_D = 1

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, 7, stride, padding=3)
        self.bn1 = nn.BatchNorm1d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, 7, stride, padding=3)
        self.bn2 = nn.BatchNorm1d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out

class Class_ResNet(nn.Module):    
    def __init__(self, block, layers):
        self.inplanes = 64
        self.deconv_with_bias = False
        print("---------------flatten pose net---------------")
        super(Class_ResNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(self.inplanes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(block, 64, 2,stride=1)  # Bottleneck 3
        #self.do1 = nn.Dropout(0.5)
        self.layer2 = self._make_layer(block, 96, 2, stride=2)
        #self.do2 = nn.Dropout(0.5)
        self.layer3 = self._make_layer(block, 128, 2, stride=2)  # Bottleneck 6
        #self.do3 = nn.Dropout(0.5)
        self.layer4 = self._make_layer(block, 192, 2, stride=2)  # Bottlenect 3
        #self.do4 = nn.Dropout(0.5)
        self.layer5 = self._make_layer(block, 256, 2, stride=2)
        self.linear = nn.Linear(256*7, 256*7)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(256*7, 9)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample=None
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        #x = self.do1(x)
        x = self.layer2(x)
        #x = self.do2(x)
        x = self.layer3(x)
        #x = self.do3(x)
        x = self.layer4(x)
        #x = self.do4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = self.relu(x)
        t = x
        x = self.linear2(x)
        return x,t

class PoseResNet(nn.Module):
    def __init__(self, block, layers):
        self.inplanes = 64
        self.deconv_with_bias = False
        print("---------------flatten pose net---------------")
        super(PoseResNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(9, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(self.inplanes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(block, 64, layers[0],stride=1)      # Bottleneck 3
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)      # Bottleneck 6
        #self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        #self.layer5 = nn.Linear(128*25, 2048)
        self.deconv_layer = self._make_deconv_layer(
            4,  # NUM_DECONV_LAYERS
            [256,256,256,256],  # NUM_DECONV_FILTERS
            [3,4,4,4],  # NUM_DECONV_KERNERLS
        )
        self.final_layer = nn.Conv2d(
            in_channels=256,  # NUM_DECONV_FILTERS[-1]
            out_channels=13,  # NUM_JOINTS,
            kernel_size=1,  # FINAL_CONV_KERNEL
            stride=1,
            padding=0  # if FINAL_CONV_KERNEL = 3 else 1
        )

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 0
            output_padding = 0
        elif deconv_kernel == 5:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):  
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        layers = []
        s = 2
        self.inplanes = 256
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)
            if i==0:
                s=3
            else:
                s=2
            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=s,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x) 
        #x = self.layer4(x)
        x = x.view(x.size(0),256,5,5)
        x = self.deconv_layer(x)
        x = self.final_layer(x)
        
        return x
'''
class PoseResNet(nn.Module):
    def __init__(self, block, layers):
        super(PoseResNet, self).__init__()
        self.a1 = PoseResNet_element(block,layers)
        self.a2 = PoseResNet_element(block,layers)
        self.a3 = PoseResNet_element(block,layers)
        self.a4 = PoseResNet_element(block,layers)
        self.a5 = PoseResNet_element(block,layers)
        self.a6 = PoseResNet_element(block,layers)
        self.a7 = PoseResNet_element(block,layers)
        self.a8 = PoseResNet_element(block,layers)
        self.a9 = PoseResNet_element(block,layers)
        self.a10 = PoseResNet_element(block,layers)
        self.a11 = PoseResNet_element(block,layers)
        self.a12 = PoseResNet_element(block,layers)
        self.a13 = PoseResNet_element(block,layers)
    def forward(self, x):
        x1 = self.a1(x)
        x2 = self.a2(x)
        x3 = self.a3(x)
        x4 = self.a4(x)
        x5 = self.a5(x)
        x6 = self.a6(x)
        x7 = self.a7(x)
        x8 = self.a8(x)
        x9 = self.a9(x)
        x10 = self.a10(x)
        x11 = self.a11(x)
        x12 = self.a12(x)
        x13 = self.a13(x)
        out = torch.cat((x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13),1)
        return out
'''
resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
              }

def get_Classifier(num_layer):
    num_layers = num_layer
    block_class, layers = resnet_spec[num_layers]
    model = Class_ResNet(block_class,layers)
    return model

def get_2d_pose_net(num_layer, input_depth):
    global INPUT_D
    INPUT_D = input_depth
    num_layers = num_layer
    block_class, layers = resnet_spec[num_layers]

    # model = PoseResNet(block_class, layers, cfg, **kwargs)

    model = PoseResNet(block_class, layers)

    # if is_train and cfg.MODEL.INIT_WEIGHTS:
    #    model.init_weights(cfg.MODEL.PRETRAINED)
    # model.init_weights('models/imagenet/resnet50-19c8e357.pth')
    return model
