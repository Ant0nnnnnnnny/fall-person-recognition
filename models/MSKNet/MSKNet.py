import torch.nn as nn
import torch
from .backbone.backbone import SKBackbone
from .head.head import SKHead
import torch.nn.functional as F
class MSKNet(nn.Module):
    def __init__(self,
               args):

        super(MSKNet, self).__init__()

        self.backbone = SKBackbone(args=args)
        self.head = SKHead(args=args)
    def forward(self, x,target,target_weight):
        x = torch.Tensor.float(x)
        x0,x1,x2,y = self.backbone(x)
        y = self.head(x0,x1,x2,y)
        return y

if __name__ == "__main__":
    MSKNet((256, 256), 18).init_weights()
    model = MSKNet((256, 256), 18)
    test_data = torch.rand(1, 3, 256, 256)
    test_outputs = model(test_data)
    print(test_outputs.size())