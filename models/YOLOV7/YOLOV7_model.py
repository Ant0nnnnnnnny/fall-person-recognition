from common import *

class YOLOV7_Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = YOLOV7_backbone()
        self.head = YOLOV7_Head()

    def forward(self, x):
        x = self.backbone(x)
        output = self.head(x)
        return output






