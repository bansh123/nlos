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

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     bias=False)

class conv_block(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):

        x = self.conv(x)
        return x

class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class U_Net(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, in_ch=1, out_ch=1):
        super(U_Net, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        #self.Conv5 = conv_block(filters[3], filters[4])

        #self.Up5 = up_conv(filters[4], filters[3])
        #self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

       # self.active = torch.nn.Sigmoid()

    def forward(self, x):

        e1 = self.Conv1(x)
        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)
        
        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)
        
        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)
        
        d4 = self.Up4(e4)
        d4 = torch.cat((e3,d4),dim=1)
        
        d4 = self.Up_conv4(d4)
        
        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)
        
        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        
        out = self.Conv(d2)
        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
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

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=2):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=1)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Sequential(
            nn.Linear(512*25, 512*5),
            nn.ReLU(inplace=True),
            nn.Linear(512*5, 2),
        )
        

    def _make_layer(self, block, planes, blocks, stride):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 3)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class Discriminator2(nn.Module):
    """Discriminator model for source domain."""

    def __init__(self, input_dims, hidden_dims, output_dims):
        """Init discriminator."""
        super(Discriminator2, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(2048, 1012),
            nn.ReLU(),
            nn.Linear(1012, 1012),
            nn.ReLU(),
            nn.Linear(1012, 1),
        )

    def forward(self, x):
        """Forward the discriminator."""
        #x = self.conv1(x)
        #x = self.conv2(x)
        out = x.view(x.size(0), -1)
        out = self.layer(out)
        return out

class Class_ResNet(nn.Module):    
    def __init__(self, block, layers):
        self.inplanes = 64
        self.deconv_with_bias = False
        print("---------------flatten pose net---------------")
        super(Class_ResNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.inplanes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(block, 64, layers[0],stride=1)  # Bottleneck 3
        self.layer2 = self._make_layer(block, 128, layers[1],stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # Bottleneck 6
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # Bottlenect 3
        self.linear = nn.Linear(512*25, 512*5)
        self.relu=nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(512*5, 9)
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x) 
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        out = self.linear(x)
        out2 = self.relu(out)
        out2 = self.linear2(out2)
        return out2


class PoseResNet(nn.Module):
    def __init__(self, block, layers):
        self.inplanes = 64
        self.deconv_with_bias = False
        print("---------------flatten pose net---------------")
        super(PoseResNet, self).__init__()
        # ResNet
        #print(INPUT_D, self.inplanes)
        '''
        self.U = U_Net()
        self.D = ResNet(BasicBlock, [2, 2, 2, 2])
        '''
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, self.inplanes, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.inplanes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inplanes, self.inplanes, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.inplanes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0],stride=1)  # Bottleneck 3
        self.layer2 = self._make_layer(block, 128, layers[1],stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # Bottleneck 6
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        #self.layer5 = nn.Linear(3200, 2048)
        # used for deconv layers   num_deconv_layers,  num_deconv_filters, num_deconv_kernels
        self.deconv_layer = self._make_deconv_layer(
            3,  # NUM_DECONV_LAYERS
            [256,256,256],  # NUM_DECONV_FILTERS
            [4,4,4],  # NUM_DECONV_KERNERLS
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
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

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
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)
            '''
            if i==0:
                s=3
            else:
                s=2
            '''
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
        # x = F.interpolate(x, scale_factor=40, mode='bilinear', align_corners=False)
        #print("raw shape: ",x1.shape)
        '''
        x1 = self.U(x1)
        x2 = self.U(x2)
        #print("after U shape: ",x1.shape)
        x3 = torch.cat([x1,x2],dim=1)
        out1 = self.D(x3)
        #print("D out1 shape : ",out1.shape)
        x1 = self.conv1(x1)
        x1 = self.layer1(x1)
        x1 = self.layer2(x1)
        x1 = self.layer3(x1) 
        x1 = self.layer4(x1)
        x1 = self.deconv_layer(x1)
        x1 = self.final_layer(x1)

        x2 = self.conv1(x2)
        x2 = self.layer1(x2)
        x2 = self.layer2(x2)
        x2 = self.layer3(x2) 
        x2 = self.layer4(x2)
        x2 = self.deconv_layer(x2)
        x2 = self.final_layer(x2)
        #print("out2 shape: ",x1.shape)
        return out1,x1,x2
        '''
        #x = self.U(x)        
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x) 
        x = self.layer4(x)
        
        #x = x.view(x.size(0), -1)
        #x = self.layer5(x)
        #x = x.view(x.size(0),2048,1,1)
        
        x = self.deconv_layer(x)
        
        x = self.final_layer(x)
        
        return x

    def init_weights(self, pretrained=''):
        pass

class ADDAEncoder(nn.Module):
    def __init__(self, block, layers):
        self.inplanes = 64
        self.deconv_with_bias = False
        print("---------------flatten pose net---------------")
        super(ADDAEncoder, self).__init__()
        #print(INPUT_D, self.inplanes)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.inplanes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0],stride=1)  # Bottleneck 3
        self.layer2 = self._make_layer(block, 64, layers[1],stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)  # Bottleneck 6
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2)
        self.layer5 = nn.Linear(3200, 2048)
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x) 
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.layer5(x)
        return x

    def init_weights(self, pretrained=''):
        pass

resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}


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

def get_Discriminator():
    '''
    num_layers = num_layer
    block_class, layers = resnet_spec[num_layers]
    model = ResNet(block_class,layers)
    '''
    model = Discriminator2(12800,1600,2)
    return model

def get_encoder(num_layer, input_depth):
    num_layers = num_layer
    block_class, layers = resnet_spec[num_layers]
    model = ADDAEncoder(block_class,layers)
    return model


def get_Classifier(num_layer):
    num_layers = num_layer
    block_class, layers = resnet_spec[num_layers]
    model = Class_ResNet(block_class,layers)
    return model
