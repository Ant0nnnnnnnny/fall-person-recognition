import time
import logging
from data_loader import get_dataloaders

from models.TCFormer import tcformer

import torch

from utils.optimizer import build_optimizer

from utils.validate import validate
def train(args,model):
    # 1. load data
    train_dataloader, val_dataloader = get_dataloaders(args)

    # 2. build model and optimizers
    model = tcformer()
    checkpoint = torch.load(args.ckpt_file, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer, scheduler = build_optimizer(args, model)

    if torch.cuda.is_available():
        model = torch.nn.parallel.DataParallel(model.cuda())
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    model.eval()
    patience = args.patience
    epoch_counter = 0
    f1_scores = []
    # 3. training
    step = checkpoint['step']
    best_score = args.best_score
    start_time = time.time()
    num_total_steps = len(train_dataloader) * args.max_epochs
    for epoch in range(checkpoint['epoch']+1,args.max_epochs):
        if epoch_counter > patience:
            logging.info(f"Epoch {epoch} step {step} ---------------early stopping.-------------")
            break
        for batch in train_dataloader:
            model.train()
            
            loss, accuracy, _, _ = model(batch)
            loss = loss.mean()
            accuracy = accuracy.mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            step += 1
            if step % args.print_steps == 0:
                time_per_step = (time.time() - start_time) / max(1, step)
                remaining_time = time_per_step * (num_total_steps - step)
                remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                logging.info(f"Epoch {epoch} step {step} eta {remaining_time}: loss {loss:.3f}, accuracy {accuracy:.3f}")

        # 4. validation

        loss = validate(model, val_dataloader)
        results = {k: round(v, 4) for k, v in results.items()}
        logging.info(f"Epoch {epoch} step {step}: loss {loss:.3f}")

        # 5. save checkpoint

        state_dict = model.module.state_dict() if args.device == 'cuda' else model.state_dict()
        torch.save({'epoch': epoch, "step":step,'model_state_dict': state_dict,'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict()},
                       f'{args.savedmodel_path}/model_epoch_{epoch}.bin')