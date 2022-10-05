
import torch
import torch.nn as nn
from models.YOLOV5.backbone import autopad,backbone
from models.YOLOV5.head import head
from models.YOLOV5.experimental import check_anchor_order, initialize_weights
import math

class YoloModel(nn.Module):
    anchors = [[116, 90, 156, 198, 373, 326],
               [30, 61, 62, 45, 59, 119],
               [10, 13, 16, 30, 33, 23]]

    def __init__(self, args):
        super(YoloModel, self).__init__()
        self.build_model(args.yolov5_class_num)

        # Build strides, anchors
        s = 128  # 2x min stride
        self.Detect.stride = torch.tensor(
            [s / x.shape[-2] for x in self.forward(torch.zeros(1, args.yolov5_input_ch, s, s))])  # forward
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
        self.backbone = backbone

        # head
        self.head = head

    def forward(self, x):
        # backbone
        y1,y2,x = self.backbone(x)
        # head
        output = self.head(y1,y2,x)
        return output

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        conv_layers = [self.Conv_6, self.Conv_7, self.Conv_8]
        for conv_layer, s in zip(conv_layers, self.Detect.stride):
            bias = conv_layer.bias.view(self.anchors_num, -1)
            bias[:, 4] += math.log(8 / (640 / s) ** 2)  # initialize confidence
            bias[:, 5:] += math.log(0.6 / (self.class_num - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            conv_layer.bias = torch.nn.Parameter(bias.view(-1), requires_grad=True)
