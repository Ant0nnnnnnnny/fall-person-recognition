import os
import torch

from models.MFNet.MFNet import MFNet
from models.MobileNet.MSNet import MSNet
from models.MSKNet.MSKNet import MSKNet
from models.STGCN.STGCN import STGCN
from models.SMLP.backbone import SMLP
from models.SGN.backbone import SGN
import time
import matplotlib.pyplot as plt
import cv2
import numpy as np
import logging

def save_checkpoint(states, is_best, output_dir,model_name,
                    filename='checkpoint.pth'):
    torch.save(states, os.path.join(output_dir,model_name,filename))
    if not os.path.exists( os.path.join(output_dir,model_name)):
        os.mkdir( os.path.join( output_dir,model_name))
   
def load_model(args):

    if args.model_name == 'msknet':
        model = MSKNet(args)
    elif args.model_name == 'msnet':
        model = MSNet(args)
    elif args.model_name == 'mfnet':
        model = MFNet(args)
    elif args.model_name == 'st-gcn':
        model = STGCN(args)
    elif args.model_name == 'smlp':
        model = SMLP(args)
    elif args.model_name == 'sgn':
        model = SGN(args)
    else:
        raise 'Unknown model name.'
    return model.to(args.device)
