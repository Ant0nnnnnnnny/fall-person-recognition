from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
import torch
import json
import tarfile
import cv2
import numpy as np
from functools import partial


def get_dataloaders(args):
    dataset = PoseDataSet(args, args.train_annotation, args.train_zip_feat)
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
                                      
    return train_dataloader, val_dataloader


class PoseDataSet(Dataset):
    def __init__(self, args, ann_path, zip_path) -> None:
        pass
        self.args = args
        self.zip_feat_path = zip_path
        self.num_workers = args.num_workers
        self.img_shape = args.img_shape
        with open(ann_path, 'r') as f:
            self.anns = json.load(f)['train']

    def __len__(self) -> int:
        return len(self.anns)

    def get_img(self, idx) -> np.array:
        file_name = self.anns[idx]['filename']
        if self.num_workers > 0:
            worker_id = torch.utils.data.get_worker_info().id
            if self.handles[worker_id] is None:
                self.handles[worker_id] = tarfile.TarFile(
                    self.zip_feat_path, 'r')
            handle = self.handles[worker_id]
        else:
            handle = self.handles
        img = np.frombuffer(handle.read(name=file_name), np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_ANYCOLOR)
        img = cv2.resize(img, (self.img_shape, self.img_shape))
        return img

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

        img = self.get_img(idx)
        head_position = self.get_head_position(idx)
        points = self.get_annopoints(idx)

        data = dict(
            img=img,
            head_position=head_position,
            points=points
        )
        return data
