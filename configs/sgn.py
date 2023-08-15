def sgn_config(parser):
    parser.add_argument('--num_classes', type=int,
                        default=120)
    parser.add_argument('--seg', type = int ,default = 20)
    parser.add_argument('--bias', type=bool, default=True) 
    parser.add_argument('--num_joint', type = int, default = 17)
    parser.add_argument('--channels', type = int, default = 2)
    return parser