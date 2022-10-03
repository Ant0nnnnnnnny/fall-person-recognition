import torch
import random
import numpy as np
import logging
from configs.msnet import msnet_config
from configs.tcformer import tcformers_config

def setup_device(args):
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.n_gpu = torch.cuda.device_count()


def setup_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def setup_logging():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)

    return logger

def setup(args_parser,args):
    setup_logging()
    if args.model_name == 'msnet':
        args = msnet_config(args_parser)
    elif args.model_name == 'tcformer':
        args = tcformers_config(args_parser)
    args = args.parse_args()
    setup_device(args)
    setup_seed(args)
    return args