from torch.utils.data import Dataset
import logging
import pickle
import torch
import numpy as np

class SkeletonDataset(Dataset):
    def __init__(self, args, train = True):
        
        super().__init__()
        self.original_data = None
        
        with open(args.skeleton_dataset_dir,'rb') as f:

            self.original_data = pickle.load(f)
            
        if train:
            self.original_data = self.original_data[:int(len(self.original_data) * 0.7)]
        else:
            self.original_data = self.original_data[int(len(self.original_data) * 0.7):]
            
            
        logging.info('Skeleton dataset loading complete.')
        
    def __getitem__(self, idx):
        
        shape = self.original_data[idx]['img_shape']

        keypoints = (self.original_data[idx]['keypoint']/shape).transpose(3,1,2,0)

        keypoints = np.pad(keypoints,((0,0),(0,300 - keypoints.shape[1]), (0,0),(0,2 - keypoints.shape[-1])))
        
        X = torch.tensor(keypoints,dtype = torch.float32)

        if self.class_range == None:
            y = torch.tensor(self.original_data[idx]['label'],dtype = torch.long)
        else:
            y =torch.tensor( self.class_range.index(self.original_data[idx]['label']) if self.original_data[idx]['label'] in self.class_range else 0 ,dtype = torch.long)
        
        feats = {'img_shape':shape, 'time_length':self.original_data[idx]['total_frames']}
        
        return X, y, feats
    
    
    def __len__(self):
        
        return len(self.original_data)
    