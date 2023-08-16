
from torch.utils.tensorboard import SummaryWriter

from utils.config import parse_args
from utils.data_loader import get_dataloaders
from utils.loss import get_loss
from utils.optimizer import build_optimizer
from utils.tools import save_checkpoint

from utils.train import validate
from utils.setup import setup
from utils.train import train

from utils.train_skeleton import train as skeleton_train
from utils.train_skeleton import validate as skeleton_validate

from utils.tools import load_model,inference_with_detector, inference_with_tracker

import torch
import os
import logging


import warnings


warnings.filterwarnings("ignore")


def main(args):

    train_dataloader, val_dataloader, _, val_data_set = get_dataloaders(
        args)
    writer_dict = {
        'writer': SummaryWriter(log_dir=args.log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    loss_func = get_loss(args)

    model = load_model(args)
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
        model.load_state_dict(checkpoint['state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer'])
        logging.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file, checkpoint['epoch']))
    for epoch in range(begin_epoch, args.max_epochs):

        if args.model_name in ['st-gcn','smlp','sgn'] :

            skeleton_train(args,train_dataloader,model, optimizer, epoch, loss_func,writer_dict)
    
            val_loss = skeleton_validate(args,  val_dataloader, model, loss_func,scheduler.get_last_lr()[0], writer_dict)

            scheduler.step(val_loss)

            logging.info('=> saving checkpoint to {}'.format(args.ckpg_dir))

            save_checkpoint({
                'epoch': epoch + 1,
                'model': args.model_name,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, best_model, args.ckpg_dir, args.model_name)


        else:
            # train for one epoch
            train(args, train_dataloader, model, optimizer, epoch, loss_func,
                writer_dict)
        
            # evaluate on validation set
            perf_indicator, val_loss = validate(
                args, val_dataloader, val_data_set, model, loss_func,
                writer_dict
            )
            scheduler.step(val_loss)

            logging.info('=> saving checkpoint to {}'.format(args.ckpg_dir))
            save_checkpoint({
                'epoch': epoch + 1,
                'model': args.model_name,
                'state_dict': model.state_dict(),
                'best_state_dict': model.module.state_dict(),
                'perf': perf_indicator,
                'optimizer': optimizer.state_dict(),
            },  args.ckpg_dir, args.model_name)

    final_model_state_file = os.path.join(
        args.ckpg_dir, 'final_state.pth'
    )
    logging.info('=> saving final model state to {}'.format(
        final_model_state_file)
    )
    torch.save(model.state_dict(), final_model_state_file)

    writer_dict['writer'].close()

if __name__ == '__main__':

    args_parser, args = parse_args()
    args = setup(args_parser, args)
    main(args)
    # import os
    # os.system('/root/upload.sh')

    # inference_with_detector(args=args, img_path=os.path.join('examples','multi-pose.jpg'))

    # inference_with_detector(args=args, img_path=os.path.join('examples','fallen-pose.jpg'))
    # inference_with_detector(args=args, img_path=os.path.join('examples','stand-pose.jpg'))
    # inference_with_detector(args=args, img_path=os.path.join('examples','sit-pose.jpg'))


    # inference_with_tracker(args=args,video_path=os.path.join('examples','video3.mp4'))