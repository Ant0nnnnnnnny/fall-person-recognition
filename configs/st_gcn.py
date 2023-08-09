def st_gcn_config(parser):
    parser.add_argument('--num_class', type=int,
                        default=4)
    parser.add_argument('--edge_importance_weighting', type = bool ,default = True)
    parser.add_argument('--in_channels', type=int,
                        default=2) 
    return parser