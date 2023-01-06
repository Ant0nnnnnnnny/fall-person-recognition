
import logging
import time


from utils.config import parse_args
import cv2
from utils.data_loader import get_dataloaders, get_inference_dataloader
from utils.loss import JointsMSELoss
from utils.optimizer import build_optimizer
from utils.tools import inference, save_checkpoint

from utils.train import validate
from utils.setup import setup
from utils.train import train


from models.TCFormer.pose_model import TCFormerPose
from models.MobileNet.MSNet import MSNet


import torch

import numpy as np
import os

import warnings
warnings.filterwarnings("ignore")

from tensorboardX import SummaryWriter

def main(args):

    train_dataloader,val_dataloader,train_dataset, val_data_set = get_dataloaders(args)
    writer_dict = {
        'writer': SummaryWriter(log_dir=args.log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    loss_func = JointsMSELoss(True).cuda()
    if args.model_name =='tcformer':
        model = TCFormerPose(args)
    elif args.model_name =='msnet':
        model = MSNet(args)
    model = torch.nn.DataParallel(model).cuda()
    best_perf = 0.0
    best_model = False
    last_epoch = -1
    optimizer,scheduler = build_optimizer(args, model)
    begin_epoch = 0
    checkpoint_file = os.path.join(
        args.ckpg_dir, args.model_name,'checkpoint.pth'
    )

    if args.auto_resume and os.path.exists(checkpoint_file):
        logging.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        begin_epoch = checkpoint['epoch']
        best_perf = checkpoint['perf']
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer'])
        logging.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file, checkpoint['epoch']))


    for epoch in range(begin_epoch, args.max_epochs):

        # train for one epoch
        train(args, train_dataloader, model, optimizer, epoch, loss_func,
              writer_dict)


        # evaluate on validation set
        perf_indicator,val_loss = validate(
            args, val_dataloader, val_data_set, model,loss_func,
           writer_dict
        )
        scheduler.step(val_loss)
        if perf_indicator >= best_perf:
            best_perf = perf_indicator
            best_model = True
        else:
            best_model = False

        logging.info('=> saving checkpoint to {}'.format(args.ckpg_dir))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': args.model_name,
            'state_dict': model.state_dict(),
            'best_state_dict': model.module.state_dict(),
            'perf': perf_indicator,
            'optimizer': optimizer.state_dict(),
        }, best_model, args.ckpg_dir, args.model_name)

    final_model_state_file = os.path.join(
        args.ckpg_dir, 'final_state.pth'
    )
    logging.info('=> saving final model state to {}'.format(
        final_model_state_file)
    )
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()


def inf(args,mode = 'offline',video_path = None, camera_id = None,pic_path = None,use_dataset = False):
    if mode == 'offline':
        if args.model_name =='tcformer':
            model = TCFormerPose(args)
        elif args.model_name =='msnet':
            model = MSNet(args)
        model = torch.nn.DataParallel(model).cuda()
        checkpoint_file = os.path.join(
            args.ckpg_dir, args.model_name,'checkpoint.pth'
        )
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['state_dict'])
        dataset = get_inference_dataloader(args)
        start_time = time.time()

        if pic_path!=None:
            img = cv2.imread(pic_path)
            h,w = img.shape[:2]
            img = cv2.resize(img,tuple(args.img_shape))

            result = inference(model,args,img,0,None,'offline',use_dataset)
            result = cv2.resize(result,(w,h))
            cv2.imshow('image', result) 
            k = cv2.waitKey(0) 
            #q键退出
            return  

        for i, (x, _, _, meta) in enumerate(dataset):
            inference(model,args,x,i,meta,'offline',use_dataset)
            end_time = time.time()
            logging.info('{0}/{1} finished, {2} seconds/sample.{3} fps'.format(i,len(dataset),(end_time - start_time)/args.test_batch_size,args.test_batch_size/(end_time - start_time)))
            start_time = time.time()
    else:
        video_capture = None
        if video_path !=None:
            video_capture = video_path
        if camera_id != None:
            video_capture = camera_id
        if video_capture == None:
            raise TypeError(
            "One of camera id and video path should not be none in online mode."
        )
        if args.model_name =='tcformer':
            model = TCFormerPose(args)
        elif args.model_name =='msnet':
            model = MSNet(args)
        model = torch.nn.DataParallel(model).cuda()
        checkpoint_file = os.path.join(
            args.ckpg_dir, args.model_name,'checkpoint.pth'
        )
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['state_dict'])
        cap = cv2.VideoCapture(video_capture) 
        while(cap.isOpened()): 
            ret, frame = cap.read() 
            h,w = frame.shape[:2]
            frame = cv2.resize(frame,(tuple(args.img_shape)))
            result = inference(model,args,frame,0,None,'online',use_dataset)
            result = cv2.resize(result,(w,h))
            cv2.imshow('image', result) 
            k = cv2.waitKey(20) 
            #q键退出
            if (k & 0xff == ord('q')): 
                break 
        cap.release() 
        

if __name__ == '__main__':

    args_parser,args = parse_args()
    args = setup(args_parser,args)
    # main(args)
    inf(args,mode = 'offline',pic_path=os.path.join('examples','pic1.jpg'))
    # inf(args,mode = 'online',video_path=os.path.join('examples','demo2s.mp4'))



