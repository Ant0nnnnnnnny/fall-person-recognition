import math
from copy import copy
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d
from PIL import Image
from torch.cuda import amp



##### basic ####

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class MP(nn.Module):
    def __init__(self, k=2):
        super(MP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=k)

    def forward(self, x):
        return self.m(x)


class Concat(nn.Module):
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class CBM(nn.Module):
    # conv+bn+sigmoid
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(CBM, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.Sigmoid() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class DownC(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, n=1, k=2):
        super(DownC, self).__init__()
        c_ = int(c1)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2//2, 3, k)
        self.cv3 = Conv(c1, c2//2, 1, 1)
        self.mp = nn.MaxPool2d(kernel_size=k, stride=k)

    def forward(self, x):
        return torch.cat((self.cv2(self.cv1(x)), self.cv3(self.mp(x))), dim=1)



##### cspnet #####

class SPPCSPC(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(SPPCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))



##### repvgg #####

class RepConv(nn.Module):
    # Represented convolution
    # https://arxiv.org/abs/2101.03697

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True, deploy=False):
        super(RepConv, self).__init__()

        self.deploy = deploy
        self.groups = g
        self.in_channels = c1
        self.out_channels = c2

        assert k == 3
        assert autopad(k, p) == 1

        padding_11 = autopad(k, p) - k // 2

        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

        if deploy:
            self.rbr_reparam = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=True)

        else:
            self.rbr_identity = (nn.BatchNorm2d(num_features=c1) if c2 == c1 and s == 1 else None)

            self.rbr_dense = nn.Sequential(
                nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False),
                nn.BatchNorm2d(num_features=c2),
            )

            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d( c1, c2, 1, s, padding_11, groups=g, bias=False),
                nn.BatchNorm2d(num_features=c2),
            )

    def forward(self, inputs):
        if hasattr(self, "rbr_reparam"):
            return self.act(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.act(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)
    
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return (
            kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid,
            bias3x3 + bias1x1 + biasid,
        )

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch[0].weight
            running_mean = branch[1].running_mean
            running_var = branch[1].running_var
            gamma = branch[1].weight
            beta = branch[1].bias
            eps = branch[1].eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros(
                    (self.in_channels, input_dim, 3, 3), dtype=np.float32
                )
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def repvgg_convert(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        return (
            kernel.detach().cpu().numpy(),
            bias.detach().cpu().numpy(),
        )

    def fuse_conv_bn(self, conv, bn):

        std = (bn.running_var + bn.eps).sqrt()
        bias = bn.bias - bn.running_mean * bn.weight / std

        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        weights = conv.weight * t

        bn = nn.Identity()
        conv = nn.Conv2d(in_channels = conv.in_channels,
                              out_channels = conv.out_channels,
                              kernel_size = conv.kernel_size,
                              stride=conv.stride,
                              padding = conv.padding,
                              dilation = conv.dilation,
                              groups = conv.groups,
                              bias = True,
                              padding_mode = conv.padding_mode)

        conv.weight = torch.nn.Parameter(weights)
        conv.bias = torch.nn.Parameter(bias)
        return conv

    def fuse_repvgg_block(self):    
        if self.deploy:
            return
        print(f"RepConv.fuse_repvgg_block")
                
        self.rbr_dense = self.fuse_conv_bn(self.rbr_dense[0], self.rbr_dense[1])
        
        self.rbr_1x1 = self.fuse_conv_bn(self.rbr_1x1[0], self.rbr_1x1[1])
        rbr_1x1_bias = self.rbr_1x1.bias
        weight_1x1_expanded = torch.nn.functional.pad(self.rbr_1x1.weight, [1, 1, 1, 1])
        
        # Fuse self.rbr_identity
        if (isinstance(self.rbr_identity, nn.BatchNorm2d) or isinstance(self.rbr_identity, nn.modules.batchnorm.SyncBatchNorm)):
            identity_conv_1x1 = nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    groups=self.groups, 
                    bias=False)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.to(self.rbr_1x1.weight.data.device)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.squeeze().squeeze()
            identity_conv_1x1.weight.data.fill_(0.0)
            identity_conv_1x1.weight.data.fill_diagonal_(1.0)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.unsqueeze(2).unsqueeze(3)

            identity_conv_1x1 = self.fuse_conv_bn(identity_conv_1x1, self.rbr_identity)
            bias_identity_expanded = identity_conv_1x1.bias
            weight_identity_expanded = torch.nn.functional.pad(identity_conv_1x1.weight, [1, 1, 1, 1])            
        else:
            bias_identity_expanded = torch.nn.Parameter( torch.zeros_like(rbr_1x1_bias) )
            weight_identity_expanded = torch.nn.Parameter( torch.zeros_like(weight_1x1_expanded) )            

        self.rbr_dense.weight = torch.nn.Parameter(self.rbr_dense.weight + weight_1x1_expanded + weight_identity_expanded)
        self.rbr_dense.bias = torch.nn.Parameter(self.rbr_dense.bias + rbr_1x1_bias + bias_identity_expanded)
                
        self.rbr_reparam = self.rbr_dense
        self.deploy = True

        if self.rbr_identity is not None:
            del self.rbr_identity
            self.rbr_identity = None

        if self.rbr_1x1 is not None:
            del self.rbr_1x1
            self.rbr_1x1 = None

        if self.rbr_dense is not None:
            del self.rbr_dense
            self.rbr_dense = None

class ELAN(nn.Module):
    def __init__(self, c1, out_dim, expand_ratio=0.5):
        super(ELAN, self).__init__()
        c_ = int(c1*expand_ratio)
        self.cv1 = Conv(c1, c_, k=1)
        self.cv2 = Conv(c1, c_, k=1)
        self.cv3 = nn.Sequential(
            Conv(c_, c_, k=3, p=1,s = 1),
            Conv(c_, c_, k=3, p=1,s = 1)
        )
        self.cv4 = nn.Sequential(
            Conv(c_, c_, k=3, p=1,s = 1),
            Conv(c_, c_, k=3, p=1,s = 1)
        )
        assert c_*4 == out_dim
        self.out = Conv(c_*4, out_dim, k=1)

    def forward(self, x):
        """
        Input:
            x: [B, C, H, W]
        Output:
            out: [B, 2C, H, W]
        """
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x3 = self.cv3(x2)
        x4 = self.cv4(x3)
        # [B, C, H, W] -> [B, 2C, H, W]
        out = self.out(torch.cat([x1, x2, x3, x4], dim=1))
        return out

class MP_1(nn.Module):
    def __init__(self, c1):
        super().__init__()
        c_ = c1 // 2
        self.mp = nn.MaxPool2d((2, 2), 2)
        self.cv1 = Conv(c1, c_, k=1)
        self.cv2 = nn.Sequential(
            Conv(c1, c_, k=1),
            Conv(c_, c_, k=3, p=1, s=2)
        )

    def forward(self, x):
        """
        Input:
            x: [B, C, H, W]
        Output:
            out: [B, C, H//2, W//2]
        """
        # [B, C, H, W] -> [B, C//2, H//2, W//2]
        x1 = self.cv1(self.mp(x))
        x2 = self.cv2(x)

        # [B, C, H//2, W//2]
        out = torch.cat([x1, x2], dim=1)

        return out

class ELAN_W(nn.Module):
    """
    ELAN BLock of YOLOv7's head
    """
    def __init__(self, c1, out_dim, expand_ratio=0.5):
        super(ELAN_W, self).__init__()
        c_ = int(c1 * expand_ratio)
        c_2 = int(c_ * expand_ratio)
        self.cv1 = Conv(c1, c_, k=1, s = 1)
        self.cv2 = Conv(c1, c_, k=1, s = 1)
        self.cv3 = Conv(c_, c_2, k=3, p=1, s = 1)
        self.cv4 = Conv(c_2, c_2, k=3, p=1, s = 1)
        self.cv5 = Conv(c_2, c_2, k=3, p=1, s = 1)
        self.cv6 = Conv(c_2, c_2, k=3, p=1, s = 1)

        self.out = Conv(c_*2+c_2*4, out_dim, k=1)


    def forward(self, x):
        """
        Input:
            x: [B, C_in, H, W]
        Output:
            out: [B, C_out, H, W]
        """
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x3 = self.cv3(x2)
        x4 = self.cv4(x3)
        x5 = self.cv5(x4)
        x6 = self.cv6(x5)

        # [B, C_in, H, W] -> [B, C_out, H, W]
        out = self.out(torch.cat([x1, x2, x3, x4, x5, x6], dim=1))

        return out


class MP_2(nn.Module):
    def __init__(self, c1):
        super().__init__()
        c_ = c1
        self.mp = nn.MaxPool2d((2, 2), 2)
        self.cv1 = Conv(c1, c_, k=1)
        self.cv2 = nn.Sequential(
            Conv(c1, c_, k=1),
            Conv(c_, c_, k=3, p=1, s=2)
        )

    def forward(self, x):
        """
        Input:
            x: [B, C, H, W]
        Output:
            out: [B, 2C, H//2, W//2]
        """
        # [B, C, H, W] -> [B, C//2, H//2, W//2]
        x1 = self.cv1(self.mp(x))
        x2 = self.cv2(x)

        # [B, C, H//2, W//2]
        out = torch.cat([x1, x2], dim=1)

        return out


class YOLOV7_backbone(nn.Module):
    def __init__(self):
        super(YOLOV7_backbone, self).__init__()

        self.layer_1 = nn.Sequential(
            Conv(3, 32, k=3, p=1),      
            Conv(32, 64, k=3, p=1, s=2),
            Conv(64, 64, k=3, p=1)                                                   # P1/2
        )
        self.layer_2 = nn.Sequential(   
            Conv(64, 128, k=3, p=1, s=2),             
            ELAN(c1=128, out_dim=256, expand_ratio=0.5)                     # P2/4
        )
        self.layer_3 = nn.Sequential(
            MP_1(c1=256),             
            ELAN(c1=256, out_dim=512, expand_ratio=0.5)                     # P3/8
        )
        self.layer_4 = nn.Sequential(
            MP_1(c1=512),             
            ELAN(c1=512, out_dim=1024, expand_ratio=0.5)                    # P4/16
        )
        self.layer_5 = nn.Sequential(
            MP_1(c1=1024),             
            ELAN(c1=1024, out_dim=1024, expand_ratio=0.25)                  # P5/32
        )

        self.Conv1 = Conv(512,256,1,1)
        self.Conv2 = Conv(1024,512,1,1)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x1 = self.layer_3(x)
        x2 = self.layer_4(x1)
        x3 = self.layer_5(x2)

        x1 = self.Conv1(x1)
        x2 = self.Conv2(x2)

        return [x1,x2,x3]
        # return [256,512,1024]




class YOLOV7_Head(nn.Module):
    def __init__(self, 
                 c1=[256,512,1024],
                 out_dim=[256, 512, 1024],
                 depthwise=False,
                 norm_type='BN',
                 act_type='silu'):
        super(YOLOV7_Head, self).__init__()
        self.c1s = c1
        self.out_dim = out_dim
        c3, c4, c5 = c1

        # SPPCSPC
        self.SPPCSPC = SPPCSPC(c1 = c5, c2 = 512)
        # top dwon
        ## P5 -> P4
        self.cv1 = Conv(int(c5/2), 256, k=1, s = 1)
        self.cv2 = Conv(c4, 256, k=1,s = 1)
        self.head_elan_1 = ELAN_W(c1=256 + 256,out_dim=256)

        # P4 -> P3
        self.cv3 = Conv(256, 128, k=1, s = 1)
        self.cv4 = Conv(c3, 128, k=1,s = 1)
        self.head_elan_2 = ELAN_W(c1=128 + 128, out_dim=128)

        # bottom up
        # P3 -> P4
        self.mp1 = MP_2(128)
        self.head_elan_3 = ELAN_W(c1=256 + 256,out_dim=256)

        # P4 -> P5
        self.mp2 = MP_2(256)
        self.head_elan_4 = ELAN_W(c1=512 + 512,out_dim=512)

        # RepConv
        self.repconv_1 = RepConv(128, out_dim[0], k=3, s=1, p=1)
        self.repconv_2 = RepConv(256, out_dim[1], k=3, s=1, p=1)
        self.repconv_3 = RepConv(512, out_dim[2], k=3, s=1, p=1)

        # CBM
        self.CBM_1 = CBM(out_dim[0],256,k=1,s=1)
        self.CBM_2 = CBM(out_dim[1],256,k=1,s=1)
        self.CBM_3 = CBM(out_dim[2],256,k=1,s=1)

    def forward(self, features):
        c3, c4, c5 = features

        # Top down
        ## P5 -> P4
        c5 = self.SPPCSPC(c5)
        c6 = self.cv1(c5)
        c7 = F.interpolate(c6, scale_factor=2.0)
        c8 = torch.cat([c7, self.cv2(c4)], dim=1)
        c9 = self.head_elan_1(c8)
        ## P4 -> P3
        c10 = self.cv3(c9)
        c11 = F.interpolate(c10, scale_factor=2.0)
        c12 = torch.cat([c11, self.cv4(c3)], dim=1)
        c13 = self.head_elan_2(c12)

        # Bottom up
        # p3 -> P4
        c14 = self.mp1(c13)
        c15 = torch.cat([c14, c9], dim=1)
        c16 = self.head_elan_3(c15)
        # P4 -> P5
        c17 = self.mp2(c16)
        c18 = torch.cat([c17, c5], dim=1)
        c19 = self.head_elan_4(c18)

        # RepCpnv
        c20 = self.repconv_1(c13)
        c21 = self.repconv_2(c16)
        c22 = self.repconv_3(c19)

        # CBM
        c23 = self.CBM_1(c20)
        c24 = self.CBM_2(c21)
        c25 = self.CBM_3(c22)


        out_feats = [c23, c24, c25] # [P3, P4, P5]

        return out_feats