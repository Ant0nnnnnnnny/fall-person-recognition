
import math
from numpy import require
import torch
import torch.nn as nn

from models.YOLOV5.backbone import BottleneckCSP, Conv
from models.YOLOV5.experimental import check_anchor_order, initialize_weights

## 拼接
class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)



class head(nn.Module):
    def __init__(self,output_ch):
        super().__init__()
        self.CSP_4 = BottleneckCSP(c1=512, c2=512, n=1, shortcut=False)
        self.CBL_5 = Conv(c1=512, c2=256, k=1, s=1)
        self.Upsample_5 = nn.Upsample(size=None, scale_factor=2, mode="nearest")
        self.Concat_5 = Concat(dimension=1)
        self.CSP_5 = BottleneckCSP(c1=512, c2=256, n=1, shortcut=False)

        self.CBL_6 = Conv(c1=256, c2=128, k=1, s=1)
        self.Upsample_6 = nn.Upsample(size=None, scale_factor=2, mode="nearest")
        self.Concat_6 = Concat(dimension=1)
        self.CSP_6 = BottleneckCSP(c1=256, c2=128, n=1, shortcut=False)
        self.Conv_6 = nn.Conv2d(in_channels=128, out_channels=output_ch, kernel_size=1, stride=1)

        self.CBL_7 = Conv(c1=128, c2=128, k=3, s=2)
        self.Concat_7 = Concat(dimension=1)
        self.CSP_7 = BottleneckCSP(c1=256, c2=256, n=1, shortcut=False)
        self.Conv_7 = nn.Conv2d(in_channels=256, out_channels=output_ch, kernel_size=1, stride=1)

        self.CBL_8 = Conv(c1=256, c2=256, k=3, s=2)
        self.Concat_8 = Concat(dimension=1)
        self.CSP_8 = BottleneckCSP(c1=512, c2=512, n=1, shortcut=False)
        self.Conv_8 = nn.Conv2d(in_channels=512, out_channels=output_ch, kernel_size=1, stride=1)


    def forward(self, y1,y2,x):
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
        return [output_1, output_2, output_3]
    
