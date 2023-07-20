
from torch.utils.tensorboard import SummaryWriter
import logging
import time


from utils.config import parse_args
import cv2
from utils.data_loader import get_dataloaders, get_inference_dataloader
from utils.loss import JointsMSELoss
from utils.optimizer import build_optimizer
from utils.tools import Rescale, inference, save_checkpoint

from utils.train import validate
from utils.setup import setup
from utils.train import train

from models.MFNet.MFNet import MFNet
from models.MobileNet.MSNet import MSNet
from models.MSKNet.MSKNet import MSKNet
import matplotlib.pyplot as plt

import torch
import dsntnn
from utils.detector import detect_person
import os

import warnings

from utils.vis import display_pose


warnings.filterwarnings("ignore")


def main(args):

    train_dataloader, val_dataloader, train_dataset, val_data_set = get_dataloaders(
        args)
    writer_dict = {
        'writer': SummaryWriter(log_dir=args.log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    loss_func = JointsMSELoss(True).to(args.device)
    if args.model_name == 'msknet':
        model = MSKNet(args)
    elif args.model_name == 'msnet':
        model = MSNet(args)
    elif args.model_name == 'mfnet':
        model = MFNet(args)
    model = torch.nn.DataParallel(model).to(args.device)
    best_perf = 0.0
    best_model = False
    last_epoch = -1
    optimizer, scheduler = build_optimizer(args, model)
    begin_epoch = 0
    checkpoint_file = os.path.join(
        args.ckpg_dir, args.model_name, 'checkpoint.pth'
    )

    if args.auto_resume and os.path.exists(checkpoint_file):
        logging.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file,map_location=args.device)
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
        perf_indicator, val_loss = validate(
            args, val_dataloader, val_data_set, model, loss_func,
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


def inf(args, mode='offline', video_path=None, camera_id=None, pic_path=None, use_dataset=False):
    if mode == 'offline':
        if args.model_name == 'msknet':
            model = MSKNet(args)
        elif args.model_name == 'msnet':
            model = MSNet(args,1.5)
        elif args.model_name == 'mfnet':
            model = MFNet(args,1)
        model = torch.nn.DataParallel(model).to(args.device)
        checkpoint_file = os.path.join(
            args.ckpg_dir, args.model_name, 'checkpoint.pth'
        )
        
        checkpoint = torch.load(checkpoint_file,map_location=args.device)
        model.load_state_dict(checkpoint['state_dict'])
        dataset = get_inference_dataloader(args)
        start_time = time.time()

        if pic_path != None:
            img = cv2.imread(pic_path)
          
            result = inference(model, args, img, 0, None,
                               'offline', use_dataset)
            plt.savefig(pic_path.split('.')[0]+'_result.png')
            plt.show()
            return

        for i, (x, _, _, meta) in enumerate(dataset):
            inference(model, args, x, i, meta, 'offline', use_dataset)
            end_time = time.time()
            logging.info('{0}/{1} finished, {2} seconds/sample.{3} fps'.format(i, len(dataset),
                         (end_time - start_time)/args.test_batch_size, args.test_batch_size/(end_time - start_time)))
            start_time = time.time()
    else:
        video_capture = None
        if video_path != None:
            video_capture = video_path
        if camera_id != None:
            video_capture = camera_id
        if video_capture == None:
            raise TypeError(
                "One of camera id and video path should not be none in online mode."
            )
        if args.model_name == 'msknet':
            model = MSKNet(args)
        elif args.model_name == 'msnet':
            model = MSNet(args)
        elif args.model_name == 'mfnet':
            model = MFNet(args)
        model = torch.nn.DataParallel(model).to(args.device)
        checkpoint_file = os.path.join(
            args.ckpg_dir, args.model_name, 'checkpoint.pth'
        )
        checkpoint = torch.load(checkpoint_file,map_location=args.device)
        model.load_state_dict(checkpoint['state_dict'])
        cap = cv2.VideoCapture(video_capture)
        while(cap.isOpened()):
            ret, frame = cap.read()
            h, w = frame.shape[:2]
            frame = cv2.resize(frame, (tuple(args.img_shape)))
            result = inference(model, args, frame, 0,
                               None, 'online', use_dataset)
            result = cv2.resize(result, (w, h))
            cv2.imshow('image', result)
            k = cv2.waitKey(20)
            # q键退出
            if (k & 0xff == ord('q')):
                break
        cap.release()

def inference_with_detector(args, img_path):

    assert args.model_name =='mfnet' 
    'Detecting mode only supports MFNet'
    model = MFNet(args)
    checkpoint_file = os.path.join(
            args.ckpg_dir, args.model_name, 'checkpoint.pth'
        )
    model = torch.nn.DataParallel(model).to(args.device)
    checkpoint_file = os.path.join(
            args.ckpg_dir, args.model_name, 'checkpoint.pth'
        )
    checkpoint = torch.load(checkpoint_file,map_location=args.device)
    model.load_state_dict(checkpoint['state_dict'])

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    before = time.time()
    boxes = detect_person(img_path)
    mid = time.time()
    person_count = len(boxes)
    plt.figure(figsize=(10,8))
    plt.subplot(3,1,1)
    plt.imshow(img)
    for i in range(person_count):
        plt.subplot(3,person_count,person_count + i+1)
        plt.imshow(img[int(boxes[i][1]):int(boxes[i][3]), int(boxes[i][0]):int(boxes[i][2]),:])
        model.eval()
        reshape = Rescale(tuple(args.img_shape))
        x = torch.tensor(reshape(img[int(boxes[i][1]):int(boxes[i][3]), int(boxes[i][0]):int(boxes[i][2]),:])).float().to(args.device)
            
        x = torch.Tensor.float(x)
        x = x.unsqueeze(0)
        x = x.permute(0,3,1,2)
        y = model(x,None,None)
 
        y = dsntnn.dsnt(y)
        plt.subplot(3,person_count,2*person_count + i+1)
        display_pose(x[0][:3,:,:],y[0])
    after = time.time()
    plt.subplot(3,1,1)
    plt.title('Total {:.2f}s,detect:{:.2f}s. \n Pose estimation:{:.2f}s, avg: {:.2f}s/person.'.format(after - before,mid - before,after - mid,(after - mid)/person_count))
    plt.savefig(img_path.split('.')[0]+'-estimation.png')
    plt.show()


if __name__ == '__main__':

    args_parser, args = parse_args()
    args = setup(args_parser, args)
    # main(args)
    # inf(args,mode = 'offline',use_dataset=True)
    # inf(args,mode = 'offline',pic_path=os.path.join('examples','sit_pic1.jpg'))
    # inf(args,mode = 'online',video_path=os.path.join('examples','video1.mp4'))
    inference_with_detector(args=args, img_path=os.path.join('examples','fallen-pose.jpg'))
    inference_with_detector(args=args, img_path=os.path.join('examples','stand-pose.jpg'))
    inference_with_detector(args=args, img_path=os.path.join('examples','sit-pose.jpg'))
