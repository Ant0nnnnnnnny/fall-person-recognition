import torch.nn as nn
import torch

class SimpleHead(nn.Module):
    def __init__(self,args) -> None:
        super(SimpleHead, self).__init__()
        self.inplanes = args.neck_channels
        self.deconv_layers = self._make_deconv_layer(args.deconv_num_layers,
                                                    args.deconv_num_filters,
                                                    args.deconv_num_kernels,args.bn_momentum )
        
        self.neck = nn.Conv2d(in_channels=args.deconv_num_filters[-1],
            out_channels=args.num_keypoints,
            kernel_size=args.final_conv_kernel,
            stride=1,
            padding=1 if args.final_conv_kernel == 3 else 0)
        
        self.CSA = CrossSpatialAttention(args.num_keypoints)
        self.final_layer =  nn.Conv2d(in_channels=args.num_keypoints,
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
        y = self.deconv_layers(x)
        y = self.neck(y)
        y = self.CSA(y)
        y = self.final_layer(y)
        return y

class CrossSpatialAttention(nn.Module):  
    def __init__(self, in_channel,r = 0.5):
        super(CrossSpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channel, in_channel, 7, padding=3,groups=in_channel)
        self.sigmoid = nn.Sigmoid()
        self.MaxPool = nn.AdaptiveMaxPool2d(1)

        self.fc_MaxPool = nn.Sequential(
            nn.Linear(in_channel, int(in_channel * r)),  # int(channel * r)取整数, 中间层神经元数至少为1, 如有必要可设为向上取整
            nn.ReLU(),
            nn.Linear(int(in_channel * r), in_channel),
            nn.Sigmoid(),
        )

        # 全局均值池化
        self.AvgPool = nn.AdaptiveAvgPool2d(1)

        self.fc_AvgPool = nn.Sequential(
            nn.Linear(in_channel, int(in_channel * r)),  # int(channel * r)取整数, 中间层神经元数至少为1, 如有必要可设为向上取整
            nn.ReLU(),
            nn.Linear(int(in_channel * r), in_channel),
            nn.Sigmoid(),
        )
        

    def forward(self, x):
         # 1.最大池化分支
        max_branch = self.MaxPool(x)
        # 送入MLP全连接神经网络, 得到权重
        max_in = max_branch.view(max_branch.size(0), -1)
        max_weight = self.fc_MaxPool(max_in)

        # 2.全局池化分支
        avg_branch = self.AvgPool(x)
        # 送入MLP全连接神经网络, 得到权重
        avg_in = avg_branch.view(avg_branch.size(0), -1)
        avg_weight = self.fc_AvgPool(avg_in)

        # MaxPool + AvgPool 激活后得到权重weight
        weight = max_weight + avg_weight
        weight = self.sigmoid(weight)

        # 将维度为b, c的weight, reshape成b, c, 1, 1 与 输入x 相乘
        h, w = weight.shape
        # 通道注意力Mc
        Mc = torch.reshape(weight, (h, w, 1, 1))
        x = x * Mc
        # 卷积操作得到交叉注意力结果
        x_out = self.conv(x)
        Ms = self.sigmoid(x_out)

        # 与原图通道进行乘积
        x = Ms * x

        return x