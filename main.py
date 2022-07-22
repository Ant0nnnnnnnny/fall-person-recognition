import time
import logging

from matplotlib import cm
import matplotlib.pyplot as plt
from utils.config import parse_args
from utils.data_loader import get_dataloaders
from utils.optimizer import build_optimizer
from utils.validate import validate
from utils.setup import setup

from models.TCFormer.tcformer import tcformer

import torch

import numpy as np

def train(args):
    # 1. load data
    train_dataloader, val_dataloader = get_dataloaders(args)
    # 2. build model and optimizers
    model = tcformer(img_size=args.img_shape,return_map  =True)
    logging.info('Model initialization finished.')
    optimizer, scheduler = build_optimizer(args, model)
    if torch.cuda.is_available():
        logging.info('model will be running on gpu.')
        model = torch.nn.parallel.DataParallel(model.cuda())
    model.eval()
    logging.info('Training begin.')
    # 3. training
    step = 0
    start_time = time.time()
    num_total_steps = len(train_dataloader) * args.max_epochs
    for epoch in range(args.max_epochs):
        for batch in train_dataloader:
            model.train()
            name,result = model(batch)
            logging.info(name)
            r = np.array(result)
            logging.info("batch result0: "+ str(r[0].shape)+",batch result1: "+ str(r[1].shape)+",batch result2: "+ str(r[2].shape)+",batch result3: "+ str(r[3].shape))
            ax=plt.subplot()
            logging.info(torch.Tensor.cpu(r[0][0][0]).detach().numpy().shape)
            im=ax.imshow(torch.Tensor.cpu(r[0][0][0]).detach().numpy(),cmap=cm.get_cmap())#绘制 可通过更改cmap改变颜色
            plt.show()
            # logging.info(loss)
            # loss = loss.mean()
            # accuracy = accuracy.mean()
            # loss.backward()
            optimizer.step()
            optimizer.zero_grad()


            step += 1
            if step % args.print_steps == 0:
                time_per_step = (time.time() - start_time) / max(1, step)
                remaining_time = time_per_step * (num_total_steps - step)
                remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                logging.info(f"Epoch {epoch} step {step} eta {remaining_time}: loss {loss:.3f}")

        # 4. validation

        loss = validate(model, val_dataloader)
        scheduler.step(loss)
        results = {k: round(v, 4) for k, v in results.items()}
        logging.info(f"Epoch {epoch} step {step}: loss {loss:.3f}")

        # 5. save checkpoint

        state_dict = model.module.state_dict() if args.device == 'cuda' else model.state_dict()
        torch.save({'epoch': epoch, "step":step,'model_state_dict': state_dict,'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict()},
                       f'{args.savedmodel_path}/model_epoch_{epoch}.bin')
if __name__ == '__main__':
    args = parse_args()
    setup(args)
    train(args)