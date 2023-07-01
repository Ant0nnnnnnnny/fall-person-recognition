import torch.nn as nn
import torch

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo. It ensures that all layers have a channel number that is divisible by 8
    It can be seen here: https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class DeConv(nn.Sequential):
    def __init__(self, in_ch, mid_ch, out_ch, ):
        super(DeConv, self).__init__(
            nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=1),
            nn.BatchNorm2d(mid_ch),
            nn.PReLU(mid_ch),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.PReLU(out_ch),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.PReLU(out_planes)
        )
class SKHead(nn.Module):
    def __init__(self,
                args):

        super(SKHead, self).__init__()

        assert args.input_size[1] in [256]

        inverted_residual_setting = [
            [1, 64, 1, 2],  #[-1, 48, 256, 256]
                [6, 48, 2, 2],  #[-1, 48, 128, 128]
                [6, 48, 3, 2],  #[-1, 48, 64, 64]
                [6, 64, 4, 2],  #[-1, 64, 32, 32]
                [6, 96, 3, 2],  #[-1, 96, 16, 16]
                [6, 160, 3, 1], #[-1, 160, 8, 8]
                [6, 320, 1, 1], #[-1, 320, 8, 8]
                ]

        # building first layer

        self.deconv0 = DeConv(args.embedding_size, _make_divisible(inverted_residual_setting[-3][-3] * args.width_mult, args.round_nearest), 256)
        self.deconv1 = DeConv(256, _make_divisible(inverted_residual_setting[-4][-3] * args.width_mult, args.round_nearest), 256)
        self.deconv2 = DeConv(256, _make_divisible(inverted_residual_setting[-5][-3] * args.width_mult, args.round_nearest), 256)

        self.final_layer = nn.Conv2d(
            in_channels=256,
            out_channels= args.num_keypoints*32,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.final_layer2 = nn.Conv2d(
            in_channels=args.num_keypoints*32,
            out_channels= args.num_keypoints,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def forward(self, x0,x1,x2,x):
     
        y = torch.cat([x0, x], dim=1)
        y = self.deconv0(y)
        y = torch.cat([x1, y], dim=1)
        y = self.deconv1(y)
        y = torch.cat([x2, y], dim=1)
        y = self.deconv2(y)
        y = self.final_layer(y)
        y = self.final_layer2(y)
        
        return y

    def init_weights(self):
        for i in [self.deconv0, self.deconv1, self.deconv2]:
            for name, m in i.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight, std=0.001)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        for j in [self.final_layer]:
            for m in j.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.001)
                    if hasattr(m, 'bias'):
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)
