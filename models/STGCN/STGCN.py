from torch import nn
from .st_gcn import STGCN as Backbone
from .head import SkeletonHead
class STGCN(nn.Module):
    def __init__(self, args) -> None:

        super().__init__()
        self.backbone = Backbone(args=args)
        self.head = SkeletonHead(args)

    def forward(self, X):

        y = self.backbone(X)

        # No need to softmax before torch.nn.CrossEntropy.
        # y = self.head(y)
        if self.training:
            return y
        else:
            return self.head(y)