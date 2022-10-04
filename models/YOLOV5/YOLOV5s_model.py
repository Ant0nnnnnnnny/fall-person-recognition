
import torch
import torch.nn as nn
from models.YOLOV5.backbone import autopad,Focus,BottleneckCSP,SPP,SPPF
from models.YOLOV5.head import Concat, Detect
from models.YOLOV5.experimental import check_anchor_order, initialize_weights
import math

class YoloModel(nn.Module):
    anchors = [[116, 90, 156, 198, 373, 326],
               [30, 61, 62, 45, 59, 119],
               [10, 13, 16, 30, 33, 23]]

    def __init__(self, class_num=1, input_ch=3):
        super(YoloModel, self).__init__()
        self.build_model(class_num)

        # Build strides, anchors
        s = 128  # 2x min stride
        self.Detect.stride = torch.tensor(
            [s / x.shape[-2] for x in self.forward(torch.zeros(1, input_ch, s, s))])  # forward
        self.Detect.anchors /= self.Detect.stride.view(-1, 1, 1)
        check_anchor_order(self.Detect)
        self.stride = self.Detect.stride
        # print('Strides: %s' % self.Detect.stride.tolist())  # [8.0, 16.0, 32.0]
        print("Input size must be multiple of", self.stride.max().item())

        initialize_weights(self)
        self._initialize_biases()  # only run once
        # model_info(self)

    def build_model(self, class_num):
        # output channels
        self.class_num = class_num
        self.anchors_num = len(self.anchors[0]) // 2
        self.output_ch = self.anchors_num * (5 + class_num)
        # backbone
        self.Focus = Focus(c1=3, c2=32, k=3, s=1)
        self.CBL_1 = self.CBL(c1=32, c2=64, k=3, s=2)
        self.CSP_1 = BottleneckCSP(c1=64, c2=64, n=1)
        self.CBL_2 = self.CBL(c1=64, c2=128, k=3, s=2)
        self.CSP_2 = BottleneckCSP(c1=128, c2=128, n=3)
        self.CBL_3 = self.CBL(c1=128, c2=256, k=3, s=2)
        self.CSP_3 = BottleneckCSP(c1=256, c2=256, n=3)
        self.CBL_4 = self.CBL(c1=256, c2=512, k=3, s=2)
        self.SPP = SPP(c1=512, c2=512, k=(5, 9, 13))

        # head
        self.CSP_4 = BottleneckCSP(c1=512, c2=512, n=1, shortcut=False)

        self.CBL_5 = self.CBL(c1=512, c2=256, k=1, s=1)
        self.Upsample_5 = nn.Upsample(size=None, scale_factor=2, mode="nearest")
        self.Concat_5 = Concat(dimension=1)
        self.CSP_5 = BottleneckCSP(c1=512, c2=256, n=1, shortcut=False)

        self.CBL_6 = self.CBL(c1=256, c2=128, k=1, s=1)
        self.Upsample_6 = nn.Upsample(size=None, scale_factor=2, mode="nearest")
        self.Concat_6 = Concat(dimension=1)
        self.CSP_6 = BottleneckCSP(c1=256, c2=128, n=1, shortcut=False)
        self.Conv_6 = nn.Conv2d(in_channels=128, out_channels=self.output_ch, kernel_size=1, stride=1)

        self.CBL_7 = self.CBL(c1=128, c2=128, k=3, s=2)
        self.Concat_7 = Concat(dimension=1)
        self.CSP_7 = BottleneckCSP(c1=256, c2=256, n=1, shortcut=False)
        self.Conv_7 = nn.Conv2d(in_channels=256, out_channels=self.output_ch, kernel_size=1, stride=1)

        self.CBL_8 = self.CBL(c1=256, c2=256, k=3, s=2)
        self.Concat_8 = Concat(dimension=1)
        self.CSP_8 = BottleneckCSP(c1=512, c2=512, n=1, shortcut=False)
        self.Conv_8 = nn.Conv2d(in_channels=512, out_channels=self.output_ch, kernel_size=1, stride=1)

        # detection
        self.Detect = Detect(nc=self.class_num, anchors=self.anchors)

    def forward(self, x):
        # backbone
        x = self.Focus(x)  # 0
        x = self.CBL_1(x)
        x = self.CSP_1(x)
        x = self.CBL_2(x)
        y1 = self.CSP_2(x)  # 4
        x = self.CBL_3(y1)
        y2 = self.CSP_3(x)  # 6
        x = self.CBL_4(y2)
        x = self.SPP(x)

        # head
        x = self.CSP_4(x)

        y3 = self.CBL_5(x)  # 10
        x = self.Upsample_5(y3)
        x = self.Concat_5([x, y2])
        x = self.CSP_5(x)

        y4 = self.CBL_6(x)  # 14
        x = self.Upsample_6(y4)
        x = self.Concat_6([x, y1])
        y5 = self.CSP_6(x)  # 17
        output_1 = self.Conv_6(y5)  # 18 output_1

        x = self.CBL_7(y5)
        x = self.Concat_7([x, y4])
        y6 = self.CSP_7(x)  # 21
        output_2 = self.Conv_7(y6)  # 22 output_2

        x = self.CBL_8(y6)
        x = self.Concat_8([x, y3])
        x = self.CSP_8(x)
        output_3 = self.Conv_8(x)  # 26 output_3

        output = self.Detect([output_1, output_2, output_3])
        return output

    @staticmethod
    def CBL(c1, c2, k, s):
        return nn.Sequential(
            nn.Conv2d(c1, c2, k, s, autopad(k), bias=False),
            nn.BatchNorm2d(c2),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        conv_layers = [self.Conv_6, self.Conv_7, self.Conv_8]
        for conv_layer, s in zip(conv_layers, self.Detect.stride):
            bias = conv_layer.bias.view(self.anchors_num, -1)
            bias[:, 4] += math.log(8 / (640 / s) ** 2)  # initialize confidence
            bias[:, 5:] += math.log(0.6 / (self.class_num - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            conv_layer.bias = torch.nn.Parameter(bias.view(-1), requires_grad=True)
