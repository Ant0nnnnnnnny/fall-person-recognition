import torch.nn as nn
import torch
from models.MFNet.backbone.MFNet import MobileNetV3
from models.MFNet.head.SimpleHead import SimpleHead
class MFNet(nn.Module):
    def __init__(self,args,multiplier =  1) -> None:
        super().__init__()
        self.backbone = MobileNetV3(output_channels=args.neck_channels,multiplier = multiplier)
        self.head = SimpleHead(args)
    def forward(self,x,target,target_weight):
        x = torch.Tensor.float(x)
        y = self.backbone(x)
        y = self.head(y)
        return y