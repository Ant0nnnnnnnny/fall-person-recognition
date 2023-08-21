import logging
from torch.utils.data import DataLoader, RandomSampler
from functools import partial

from utils.skeleton_dataset import SkeletonTrainDataset, SkeletonValDataset
from utils.fall_dataset import FallTrainDataset, FallValDataset
from utils.mpii import MPIIDataset
def get_inference_dataloader(args):
    dataset = MPIIDataset(args,args.dataset_root,'test',False)
    sampler = RandomSampler(dataset)
    dataloader_class = partial(DataLoader, pin_memory=True, num_workers=0)
    test_dataloader = dataloader_class(dataset,
                                        batch_size=args.test_batch_size,
                                        sampler=sampler,
                                        drop_last=True)
    return test_dataloader

def get_dataloaders(args, split_mode = 'x-sub'):
    
    train_dataset = None
    val_dataset = None

    if args.model_name in ['st-gcn', 'smlp','sgn']:
        
        if args.dataset_name == 'fall':
            train_dataset = FallTrainDataset(args=args)
            val_dataset = FallValDataset(args=args)
        else:
            train_dataset = SkeletonTrainDataset(args=args, mode=split_mode)
            val_dataset = SkeletonValDataset(args=args, mode=split_mode)
        
    else:
        train_dataset = MPIIDataset(args,args.dataset_root,'train',True)
        val_dataset = MPIIDataset(args,args.dataset_root,'valid',False)

    if args.num_workers > 0:
        dataloader_class = partial(
            DataLoader, pin_memory=True, num_workers=args.num_workers, prefetch_factor=args.prefetch)
    else:
        # single-thread reading does not support prefetch_factor arg
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=0)

    train_sampler = RandomSampler(train_dataset)

    val_sampler = RandomSampler(val_dataset)

    train_dataloader = dataloader_class(train_dataset,
                                        batch_size=args.batch_size,
                                        sampler=train_sampler,
                                        collate_fn=train_dataset.sgn_transform if args.dataset_name == 'ntu' else None,
                                        drop_last=True)
    val_dataloader = dataloader_class(val_dataset,
                                      batch_size=args.val_batch_size,
                                       collate_fn=val_dataset.sgn_transform if args.dataset_name == 'ntu' else None,
                                      sampler=val_sampler,
                                      drop_last=True)
    
    logging.info("DataLoader finished.")

    return train_dataloader, val_dataloader,train_dataset,val_dataset