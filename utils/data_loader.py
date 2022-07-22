import logging
import os
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
import torch
import json
import tarfile
import cv2
import numpy as np
from functools import partial


def get_dataloaders(args):
    dataset = PoseDataSet(args, args.train_annotation, args.train_image_feat)
    size = len(dataset)
    val_size = int(size * args.val_ratio)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [size - val_size, val_size],
                                                               generator=torch.Generator().manual_seed(args.seed))

    if args.num_workers > 0:
        dataloader_class = partial(
            DataLoader, pin_memory=True, num_workers=args.num_workers, prefetch_factor=args.prefetch)
    else:
        # single-thread reading does not support prefetch_factor arg
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=0)
    train_sampler = RandomSampler(train_dataset)

    val_sampler = SequentialSampler(val_dataset)

    train_dataloader = dataloader_class(train_dataset,
                                        batch_size=args.batch_size,
                                        sampler=train_sampler,
                                        drop_last=True)
    val_dataloader = dataloader_class(val_dataset,
                                      batch_size=args.val_batch_size,
                                      sampler=val_sampler,
                                      drop_last=False)
    logging.info("DataLoader finished.")
    return train_dataloader, val_dataloader


class PoseDataSet(Dataset):
    def __init__(self, args, ann_path, img_path) -> None:
        
        self.args = args
        self.img_feat_path = img_path
        self.num_workers = args.num_workers
        self.img_shape = args.img_shape
            
        with open(ann_path, 'r') as f:
            self.anns = json.load(f)['train']

    def __len__(self) -> int:
        return len(self.anns)

    def get_img(self, idx) -> np.array:
        file_name = self.anns[idx]['filename']
        img = cv2.imread(os.path.join(self.img_feat_path,file_name))
        img = cv2.resize(img,(self.img_shape,self.img_shape))
        return file_name,img

    def get_head_position(self, idx) -> tuple:

        x1 = self.anns[idx]['head_rect'][0]
        y1 = self.anns[idx]['head_rect'][1]
        x2 = self.anns[idx]['head_rect'][2]
        y2 = self.anns[idx]['head_rect'][3]

        return x1, y1, x2, y2

    def get_annopoints(self, idx) -> tuple:

        points = self.anns[idx]['joint_pos']

        return tuple(points.values())

    def __getitem__(self, idx: int) -> dict:
        points = self.get_annopoints(idx)
        if len(points) != 16:
            return self.__getitem__(idx+1)
        
        name,img = self.get_img(idx)
        head_position = self.get_head_position(idx)

        data = dict(
            img=img,
            head_position=head_position,
            points=points
        )

        # logging.info('img size: '+str(img.shape) + ', head_position size: '+ str(len(head_position))+ ' points size: ' + str(len(points)))
        # return data
        img = np.transpose(img,(2,0,1))
        return name, torch.FloatTensor(img)
