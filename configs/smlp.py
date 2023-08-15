def smlp_gcn_config(parser):
    parser.add_argument('--num_class', type=int,
                        default=120)
    parser.add_argument('--in_channels', type=int,
                        default=2) 
    parser.add_argument('--mlp_dims', type=list, default = [64,128,256,256,128,64,32])
    return parser