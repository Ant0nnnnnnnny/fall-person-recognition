def msnet_config(parser):
    parser.add_argument('--num_keypoints', type=int,
                        default=16)
    parser.add_argument('--neck_channels', type = int ,default = 64)
    parser.add_argument('--deconv_num_layers', type=int,
                        default=3)
    parser.add_argument('--deconv_num_filters', type=list,
                        default=[256,256,256])
    parser.add_argument('--deconv_num_kernels', type=list,
                        default=[4,4,4])
    parser.add_argument('--final_conv_kernel', type=int,
                        default=1)
    parser.add_argument('--bn_momentum', type=float,
                        default=0.1)                
    return parser