import os
def sgn_config(parser):
    parser.add_argument('--num_classes', type=int,
                        default=2)
    parser.add_argument('--seg', type = int ,default = 30)
    parser.add_argument('--bias', type=bool, default=True) 
    parser.add_argument('--num_joint', type = int, default = 17)
    parser.add_argument('--channels', type = int, default = 2)
    parser.add_argument('--classifier_weight_path', type=str, default=os.path.join('checkpoints','sgn','c0833seg30.pth')) #os.path.join('labels.txt')

    parser.add_argument('--pretrained_weight',type = str,default = os.path.join('checkpoints','sgn','c0833seg30.pth')) #ntu pretrained weight.

    return parser