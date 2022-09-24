from turtle import forward
import torch
from torch import nn

from models.TCFormer.backbone.tcformer import tcformer
from models.TCFormer.neck.mta import MTA
from models.TCFormer.head.pose_resnet import TopdownHeatmapSimpleHead

class TCFormerPose(torch.nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.tcformer = tcformer()
        self.mta = MTA(
            in_channels=args.mta_in_channels,
            out_channels=args.mta_out_channels,
            start_level=args.mta_start_level,
            add_extra_convs=args.mta_add_extra_convs,
            num_heads=args.mta_num_heads,
            mlp_ratios=args.mta_mlp_ratios,
            num_outs=args.mta_num_outs,
            use_sr_conv=args.mta_use_sr_layer,
        )
        self.resnet = TopdownHeatmapSimpleHead(
            in_channels=args.head_in_channels,
            out_channels= args.num_keypoints,
            num_deconv_layers= args.head_num_deconv_layers,
            extra=args.head_extra,
            loss_keypoint=args.head_loss_keypoint,
        )

    def forward(self, x,target,target_weight):
        x = torch.Tensor.float(x)
        y = self.tcformer(x)
        y = self.mta(y)
        y = self.resnet(y)
        return y
