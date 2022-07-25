
import logging

from utils.config import parse_args
from utils.data_loader import get_dataloaders
from utils.loss import JointsMSELoss
from utils.optimizer import build_optimizer
from utils.utils import save_checkpoint
from utils.train import validate
from utils.setup import setup
from utils.train import train

from models.pose_model import TCFormerPose

import torch

import numpy as np

import os

from tensorboardX import SummaryWriter

def main(args):

    train_dataloader,val_dataloader,train_dataset, val_data_set = get_dataloaders(args)
    writer_dict = {
        'writer': SummaryWriter(log_dir=args.log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }
    loss_func = JointsMSELoss(True).cuda()
    model = TCFormerPose(args)
    model = torch.nn.DataParallel(model).cuda()
    best_perf = 0.0
    best_model = False
    last_epoch = -1
    optimizer,scheduler = build_optimizer(args, model)
    begin_epoch = 0
    checkpoint_file = os.path.join(
        args.ckpg_dir, 'checkpoint.pth'
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
        }, best_model, args.ckpg_dir)

    final_model_state_file = os.path.join(
        args.ckpg_dir, 'final_state.pth'
    )
    logging.info('=> saving final model state to {}'.format(
        final_model_state_file)
    )
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()

if __name__ == '__main__':

    args = parse_args()
    setup(args)
    main(args)
 