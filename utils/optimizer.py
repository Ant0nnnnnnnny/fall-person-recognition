from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
def build_optimizer(args,model):

    optimizer = AdamW(model.parameters(),args.learning_rate)
    
    scheduler = ReduceLROnPlateau(optimizer=optimizer, 
    factor=args.scheduler_factor, 
    patience=args.scheduler_patience,
    verbose=True,
    min_lr=args.scheduler_min_lr)

    return optimizer,scheduler