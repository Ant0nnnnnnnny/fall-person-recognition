import torch.nn as nn

class SimpleHead(nn.Module):
    def __init__(self,args) -> None:
        super(SimpleHead, self).__init__()
        self.inplanes = args.neck_channels
        self.deconv_layers = self._make_deconv_layer(args.deconv_num_layers,
                                                    args.deconv_num_filters,
                                                    args.deconv_num_kernels,args.bn_momentum )
        
        self.final_layer = nn.Conv2d(
            in_channels=args.deconv_num_filters[-1],
            out_channels=args.num_keypoints,
            kernel_size=args.final_conv_kernel,
            stride=1,
            padding=1 if args.final_conv_kernel == 3 else 0
        )

    
    def _get_deconv_cfg(self, deconv_kernel):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding
    
    def _make_deconv_layer(self, num_layers, num_filters, num_kernels, bn_momentum):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False))
            layers.append(nn.BatchNorm2d(planes, momentum=bn_momentum))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.deconv_layers(x)
        x = self.final_layer(x)

        return x