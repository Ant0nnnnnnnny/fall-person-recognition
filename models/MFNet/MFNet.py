import torch.nn as nn
from models.MFNet.backbone.MFNet import mobilenetv2_ed
from models.MFNet.head.SimpleHead import SimpleHead
import dsntnn
class MFNet(nn.Module):
    def __init__(self,args,multiplier =  1) -> None:
        super().__init__()
        self.resnet = mobilenetv2_ed(width_mult=multiplier)
        self.outsize = 32
        self.hm_conv = nn.Conv2d(self.outsize, args.num_keypoints, kernel_size=1, bias=False)
    def forward(self,x,target,target_weight):
        
        resnet_out = self.resnet(x)

        unnormalized_heatmaps = self.hm_conv(resnet_out)
     
        heatmaps = dsntnn.flat_softmax(unnormalized_heatmaps)
    
        # coords = dsntnn.dsnt(heatmaps)

        return  heatmaps