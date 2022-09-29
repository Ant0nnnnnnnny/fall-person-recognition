def add_config(parser):
    parser.add_argument('--num_keypoints', type=int,
                        default=16)

    parser.add_argument('--heatmap_size', type=list,
                        default=[48,64])      

    parser.add_argument('--sigma', type=float,
                        default=2.0)                    

    parser.add_argument('--mta_in_channels', type=list,
                        default=[64, 128, 320, 512])

    parser.add_argument('--mta_out_channels', type=int, default=256)

    parser.add_argument('--mta_start_level', type=int, default=0)

    parser.add_argument('--mta_add_extra_convs', type=str, default='on_input')

    parser.add_argument('--mta_num_heads', type = list, default=[4,4,4,4])

    parser.add_argument('--mta_mlp_ratios', type = list, default=[4,4,4,4])

    parser.add_argument('--mta_num_outs', type = int, default=4)

    parser.add_argument('--mta_use_sr_layer', type = bool, default=False)



    parser.add_argument('--head_in_channels', type = int, default=256)

    # parser.add_argument('--head_out_channels', type = int, default=16)

    parser.add_argument('--head_num_deconv_layers', type = int, default=0)
    
    parser.add_argument('--head_extra', type = dict, default=dict(final_conv_kernel=1, ))

    parser.add_argument('--head_loss_keypoint', type = dict, default=dict(type='JointsMSELoss', use_target_weight=True))

    return parser