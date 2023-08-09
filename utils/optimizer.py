from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
def build_optimizer(args,model):

    if args.model_name == 'st-gcn':
        
        optimizer = AdamW(model.parameters(),args.learning_rate)
    
        scheduler = CosineAnnealingLR(optimizer=optimizer, 
    T_max=args.max_epochs/2)
        

    optimizer = AdamW(model.parameters(),args.learning_rate, betas=args.adamw_betas,
        weight_decay=args.adamw_weight_decay,
        amsgrad=args.adamw_amsgrad,)
    
    scheduler = CosineAnnealingLR(optimizer=optimizer, 
    T_max=args.max_epochs/2,eta_min = args.scheduler_min_lr)

        
    return optimizer,scheduler