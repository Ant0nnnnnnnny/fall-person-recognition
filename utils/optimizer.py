from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
def build_optimizer(args,model):

    optimizer = AdamW(model.parameters(),args.learning_rate)
    
    scheduler = CosineAnnealingLR(optimizer=optimizer, 
    T_max=args.max_epochs/2)

    return optimizer,scheduler