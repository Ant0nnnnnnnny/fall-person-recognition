from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts

def build_optimizer(args,model):

    if args.model_name == 'st-gcn':
        
        optimizer = AdamW(model.parameters(),args.learning_rate)
    
        scheduler = CosineAnnealingWarmRestarts(optimizer=optimizer, 
                                                T_0=args.scheduler_cosine_T_miu ,
        T_mult=args.scheduler_cosine_T_miu,eta_min = args.scheduler_min_lr)
    else:
        optimizer = AdamW(model.parameters(),args.learning_rate, betas=args.adamw_betas,
            weight_decay=args.adamw_weight_decay,
            amsgrad=args.adamw_amsgrad,)
        
        scheduler = CosineAnnealingWarmRestarts(optimizer=optimizer, 
                                                T_0=args.scheduler_cosine_T_miu ,
        T_mult=args.scheduler_cosine_T_miu,eta_min = args.scheduler_min_lr)

        
    return optimizer,scheduler