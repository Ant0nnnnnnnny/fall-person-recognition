import argparse
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

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class SKBackbone(nn.Module):
    def __init__(self,
                 args):
        super(SKBackbone, self).__init__()

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
        input_channel = _make_divisible(args.input_channel * args.width_mult, args.round_nearest)

        self.first_conv = ConvBNReLU(3, input_channel, stride=2)

        inv_residual = []
        # building inverted residual InvertedResiduals
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * args.width_mult, args.round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                inv_residual.append(InvertedResidual(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # make it nn.Sequential
        self.inv_residual = nn.Sequential(*inv_residual)

        self.last_conv = ConvBNReLU(input_channel, args.embedding_size, kernel_size=1)
        self.final_layer = nn.Conv2d(
            in_channels=256,
            out_channels= args.num_keypoints * 32,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def forward(self, x):
        x = self.first_conv(x)
        x = self.inv_residual[0:6](x)
        x2 = x
        x = self.inv_residual[6:10](x)
        x1 = x
        x = self.inv_residual[10:13](x)
        x0 = x
        x = self.inv_residual[13:16](x)
        x = self.inv_residual[16:](x)
        y = self.last_conv(x)

        return x0,x1,x2,y

    def init_weights(self):
        for j in [self.first_conv, self.inv_residual, self.last_conv]:
            for m in j.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.001)
                    if hasattr(m, 'bias'):
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)

def msknet_config(parser):

    '''
    input_size,
                 joint_num,
                 input_channel = 48,
                 embedding_size = 2048,
                 width_mult=1.0,
                 round_nearest=8,
                 InvertedResidual=None,
                 norm_layer=None,
                 nn.PReLU=None,
                 inverted_residual_setting=None
    '''
    parser.add_argument('--num_keypoints', type=int,
                        default=16)
    parser.add_argument('--input_size', type=tuple,
                        default=(256,256))
    parser.add_argument('--input_channel', type=int,
                        default=48)
    parser.add_argument('--embedding_size', type = int ,default = 2048)
    parser.add_argument('--width_mult', type = float ,default = 1.0)

    parser.add_argument('--round_nearest', type = int ,default = 8)
        
    return parser

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Arguments for person pose estimation tester')
    
    model = SKBackbone(msknet_config(parser).parse_args(args=[]))
    test_data = torch.rand(1, 3, 256, 256)
    test_outputs = model(test_data)
    x,y,z,w = test_outputs
    print(x.size())
    print(y.size())
    print(z.size())
    print(w.size())

