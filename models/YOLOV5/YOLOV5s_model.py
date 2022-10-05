
import torch
import torch.nn as nn
from models.YOLOV5.backbone import backbone
from models.YOLOV5.head import head
from models.YOLOV5.experimental import initialize_weights
import math

class YoloModel(nn.Module):
    anchors = [[116, 90, 156, 198, 373, 326],
               [30, 61, 62, 45, 59, 119],
               [10, 13, 16, 30, 33, 23]]

    def __init__(self, args) -> None:
        super().__init__()
        self.nc = args.yolov5_class_num
        self.anchors_num = len(self.anchors[0]) // 2
        self.output_ch = self.anchors_num * (5 + self.nc)
        self.backbone = backbone()
        self.head = head(output_ch = self.output_ch)
        initialize_weights(self)

    def forward(self, x):
        y1,y2,x = self.backbone(x)
        output = self.head(y1,y2,x)
        return output
