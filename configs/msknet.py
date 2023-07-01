def msknet_config(parser):

    '''
    input_size,
                 joint_num,
                 input_channel = 48,
                 embedding_size = 2048,
                 width_mult=1.0,
                 round_nearest=8,
                 block=None,
                 norm_layer=None,
                 activation_layer=None,
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