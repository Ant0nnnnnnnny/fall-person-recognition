import logging
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, random_split
from functools import partial

from utils.skeleton_dataset import SkeletonDataset
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

def get_dataloaders(args):
    
    train_dataset = None
    val_dataset = None

    if args.model_name == 'st-gcn':

        dataset = SkeletonDataset(args)
        train_length = int(len(dataset)*0.7 )
        val_length = int( len(dataset)*0.3 )
        train_dataset,val_dataset = random_split(dataset = dataset,lengths = [train_length,val_length+1])
        
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
                                        drop_last=True)
    val_dataloader = dataloader_class(val_dataset,
                                      batch_size=args.val_batch_size,
                                      sampler=val_sampler,
                                      drop_last=False)
    
    logging.info("DataLoader finished.")

    return train_dataloader, val_dataloader,train_dataset,val_dataset