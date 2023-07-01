#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by C. L. Wang on 2019/5/11
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_model_parameters(model):
    total_parameters = 0
    for layer in list(model.parameters()):
        layer_parameter = 1
        for l in list(layer.size()):
            layer_parameter *= l
        total_parameters += layer_parameter
    return total_parameters


def _weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        if type(m.bias) is not type(None):
            m.bias.data.zero_()


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        out = F.relu6(x + 3., self.inplace) / 6.
        return out * x


def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SqueezeBlock(nn.Module):
    def __init__(self, exp_size, divide=4):
        super(SqueezeBlock, self).__init__()
        self.dense = nn.Sequential(
            nn.Linear(exp_size, exp_size // divide),
            nn.ReLU(inplace=True),
            nn.Linear(exp_size // divide, exp_size),
            h_sigmoid()
        )

    def forward(self, x):
        batch, channels, height, width = x.size()
        out = F.avg_pool2d(x, kernel_size=[height, width]).view(batch, -1)
        out = self.dense(out)
        out = out.view(batch, channels, 1, 1)
        # out = hard_sigmoid(out)

        return out * x


class MobileBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, nonLinear, SE, exp_size, dropout_rate=1.0):
        super(MobileBlock, self).__init__()
        self.out_channels = out_channels
        self.nonLinear = nonLinear
        self.SE = SE
        self.dropout_rate = dropout_rate
        padding = (kernel_size - 1) // 2
        self.sppf = SPPF(in_channels=in_channels,out_channels=in_channels)
        self.use_connect = (stride == 1 and in_channels == out_channels)  # 残差条件

        if self.nonLinear == "RE":
            activation = nn.ReLU
        else:
            activation = h_swish

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, exp_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(exp_size),
            activation(inplace=True)
        )

        self.depth_conv = nn.Sequential(
            nn.Conv2d(exp_size, exp_size, kernel_size=kernel_size, stride=stride, padding=padding, groups=exp_size),
            nn.BatchNorm2d(exp_size),
        )

        if self.SE:
            self.squeeze_block = SqueezeBlock(exp_size)

        self.point_conv = nn.Sequential(
            nn.Conv2d(exp_size, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            activation(inplace=True)
        )

    def forward(self, x):
        # MobileNetV2
        x = self.sppf(x)
        out = self.conv(x)  # 转换通道 in->exp
        out = self.depth_conv(out)  # 深度卷积 exp->exp

        # Squeeze and Excite
        if self.SE:
            out = self.squeeze_block(out)

        # point-wise conv
        out = self.point_conv(out) # 转换通道 exp->out

        # connection
        if self.use_connect:
            return x + out
        else:
            return out


class MobileNetV3(nn.Module):
    def __init__(self, output_channels=256, multiplier=1.0):
        super(MobileNetV3, self).__init__()
        self.activation_HS = nn.ReLU6(inplace=True)
        self.output_channels = output_channels
        print("output_channels: ", self.output_channels)
        layers = [
                [16, 16, 3, 2, "RE", True, 16],
                [16, 24, 3, 2, "RE", False, 72],
                [24, 24, 3, 1, "RE", False, 88],
                [24, 40, 5, 2, "RE", True, 96],
                [40, 40, 5, 1, "RE", True, 240],
                [40, 40, 5, 1, "RE", True, 240],
                [40, 48, 5, 1, "HS", True, 120],
                [48, 48, 5, 1, "HS", True, 144],
                [48, 96, 5, 2, "HS", True, 288],
                [96, 96, 5, 1, "HS", True, 576],
                [96, 96, 5, 1, "HS", True, 545],
            ]
        init_conv_out = _make_divisible(16 * multiplier)
        self.init_conv = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=init_conv_out, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(init_conv_out),
                h_swish(inplace=True),
            )

        self.block = []
        for in_channels, out_channels, kernel_size, stride, nonlinear, se, exp_size in layers:
                in_channels = _make_divisible(in_channels * multiplier)
                out_channels = _make_divisible(out_channels * multiplier)
                exp_size = _make_divisible(exp_size * multiplier)
                self.block.append(MobileBlock(in_channels, out_channels, kernel_size, stride, nonlinear, se, exp_size))
        self.block = nn.Sequential(*self.block)

        out_conv1_in = _make_divisible(96 * multiplier)
        out_conv1_out = _make_divisible(576 * multiplier)
        self.out_conv1 = nn.Sequential(
                nn.Conv2d(out_conv1_in, out_conv1_out, kernel_size=1, stride=1),
                SqueezeBlock(out_conv1_out),
                nn.BatchNorm2d(out_conv1_out),
                h_swish(inplace=True),
            )
        self.se1 = CBAM(int(16*multiplier))
        self.se2 = CBAM(int(96*multiplier))
        out_conv2_in = _make_divisible(576 * multiplier)
        out_conv2_out = _make_divisible(1280 * multiplier)
        self.out_conv2 = nn.Sequential(
                nn.Conv2d(out_conv2_in, out_conv2_out, kernel_size=1, stride=1),
                h_swish(inplace=True),
                nn.Conv2d(out_conv2_out, self.output_channels, kernel_size=1, stride=1),
            )

        self.apply(_weights_init)

    def forward(self, x):
        # 起始部分
        out = self.init_conv(x)
        out = self.se1(out)
        # 中间部分
        out = self.block(out)
        out = self.se2(out)
        # 最后部分
        out = self.out_conv1(out)

        batch, channels, height, width = out.size()
        out = self.out_conv2(out)
        return out

class CBL(torch.nn.Module):
    def __init__(self, input_channels, output_channels):
        super(CBL,self).__init__()
        padding,stride = adjusted_padding_stride(3)
        self.separate_conv = torch.nn.Conv2d(in_channels=input_channels, out_channels=input_channels,kernel_size=(3,3),padding=padding,stride = stride,groups=input_channels)
        self.bn1 = torch.nn.BatchNorm2d(input_channels)
        self.point_conv = torch.nn.Conv2d(in_channels = input_channels,out_channels=output_channels,kernel_size=(1,1))
        self.bn2 = torch.nn.BatchNorm2d(output_channels)
        self.lrelu = torch.nn.functional.leaky_relu
        
    def forward(self,x):
        y = self.separate_conv(x)
        y = self.bn1(y)
        y = self.point_conv(y)
        y = self.bn2(y)
        y = self.lrelu(y)
        return y

class SPPF(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SPPF,self).__init__()
        self.CBL = CBL(input_channels=in_channels,output_channels=out_channels)
        padding,stride = adjusted_padding_stride(5)
        self.parallel1 = DepthWiseConv(in_channels=in_channels,out_channels=out_channels,kernel_size=7,padding=3,stride = 1)
        self.parallel2 = DepthWiseConv(in_channels=in_channels,out_channels=out_channels,kernel_size=5,padding=2,stride = 1)
        self.parallel3 = DepthWiseConv(in_channels=in_channels,out_channels=out_channels,kernel_size=3,padding=1,stride = 1)
        self.conv = nn.Conv2d(4*in_channels, 4*in_channels, 7, padding=3,groups=in_channels)
        self.sigmoid = nn.Sigmoid()
        self.CBL2 = CBL(input_channels=4 * in_channels,output_channels=out_channels)
    def forward(self,x):
        y = self.CBL(x)
        m1 = self.parallel1(y)
        m2 = self.parallel2(m1)
        m3 = self.parallel3(m2)
        y = torch.cat([y,m1,m2,m3],1)
        y_weight = self.conv(y)
        y_weight = self.sigmoid(y_weight)
        y = y*y_weight
        y = self.CBL2(y)
        y = y+x
        return y
class DepthWiseConv(torch.nn.Module):
    def __init__(self,in_channels,out_channels, kernel_size, padding = 1, stride = 1) -> None:
        super(DepthWiseConv,self).__init__()
        self.separate_conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=in_channels,kernel_size=kernel_size,padding=padding,stride = stride,groups=in_channels)
        self.bn1 = torch.nn.BatchNorm2d(in_channels)
        self.point_conv = torch.nn.Conv2d(in_channels = in_channels,out_channels=out_channels,kernel_size=(1,1))
    
    def forward(self,x):
        y = self.separate_conv(x)
        y = self.point_conv(x)
        return y
def adjusted_padding_stride(kernel_size):
    if kernel_size == 1:
        return 1,0
    elif kernel_size == 3:
        return 1,1
    elif kernel_size == 5:
        return 2,1
    else:
        return None



class CBAM(nn.Module):
    def __init__(self, in_channel):
        super(CBAM, self).__init__()
        self.Cam = ChannelAttentionModul(in_channel=in_channel)  # 通道注意力模块
        self.Sam = SpatialAttentionModul(in_channel=in_channel)  # 空间注意力模块

    def forward(self, x):
        x = self.Cam(x)
        x = self.Sam(x)
        return x


class ChannelAttentionModul(nn.Module):  # 通道注意力模块
    def __init__(self, in_channel, r=0.5):  # channel为输入的维度, r为全连接层缩放比例->控制中间层个数
        super(ChannelAttentionModul, self).__init__()
        # 全局最大池化
        self.MaxPool = nn.AdaptiveMaxPool2d(1)

        self.fc_MaxPool = nn.Sequential(
            nn.Linear(in_channel, int(in_channel * r)),  # int(channel * r)取整数, 中间层神经元数至少为1, 如有必要可设为向上取整
            nn.ReLU(),
            nn.Linear(int(in_channel * r), in_channel),
            nn.Sigmoid(),
        )

        # 全局均值池化
        self.AvgPool = nn.AdaptiveAvgPool2d(1)

        self.fc_AvgPool = nn.Sequential(
            nn.Linear(in_channel, int(in_channel * r)),  # int(channel * r)取整数, 中间层神经元数至少为1, 如有必要可设为向上取整
            nn.ReLU(),
            nn.Linear(int(in_channel * r), in_channel),
            nn.Sigmoid(),
        )

        # 激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1.最大池化分支
        max_branch = self.MaxPool(x)
        # 送入MLP全连接神经网络, 得到权重
        max_in = max_branch.view(max_branch.size(0), -1)
        max_weight = self.fc_MaxPool(max_in)

        # 2.全局池化分支
        avg_branch = self.AvgPool(x)
        # 送入MLP全连接神经网络, 得到权重
        avg_in = avg_branch.view(avg_branch.size(0), -1)
        avg_weight = self.fc_AvgPool(avg_in)

        # MaxPool + AvgPool 激活后得到权重weight
        weight = max_weight + avg_weight
        weight = self.sigmoid(weight)

        # 将维度为b, c的weight, reshape成b, c, 1, 1 与 输入x 相乘
        h, w = weight.shape
        # 通道注意力Mc
        Mc = torch.reshape(weight, (h, w, 1, 1))

        # 乘积获得结果
        x = Mc * x

        return x


class SpatialAttentionModul(nn.Module):  # 空间注意力模块
    def __init__(self, in_channel):
        super(SpatialAttentionModul, self).__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x维度为 [N, C, H, W] 沿着维度C进行操作, 所以dim=1, 结果为[N, H, W]
        MaxPool = torch.max(x, dim=1).values  # torch.max 返回的是索引和value， 要用.values去访问值才行！
        AvgPool = torch.mean(x, dim=1)

        # 增加维度, 变成 [N, 1, H, W]
        MaxPool = torch.unsqueeze(MaxPool, dim=1)
        AvgPool = torch.unsqueeze(AvgPool, dim=1)

        # 维度拼接 [N, 2, H, W]
        x_cat = torch.cat((MaxPool, AvgPool), dim=1)  # 获得特征图

        # 卷积操作得到空间注意力结果
        x_out = self.conv(x_cat)
        Ms = self.sigmoid(x_out)

        # 与原图通道进行乘积
        x = Ms * x

        return x
    