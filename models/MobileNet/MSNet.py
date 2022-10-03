import torch.nn as nn
import torch
from models.MobileNet.backbone.MobileNet import MobileNetV3
from models.MobileNet.head.SimpleHead import SimpleHead
class MSNet(nn.Module):
    def __init__(self,args) -> None:
        super().__init__()
        self.backbone = MobileNetV3(output_channels=args.neck_channels)
        self.head = SimpleHead(args)
    def forward(self,x,target,target_weight):
        x = torch.Tensor.float(x)
        y = self.backbone(x)

        y = self.head(y)
        return y