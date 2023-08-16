from torch.utils.data import Dataset
import logging
import pickle
import torch
import numpy as np

class SkeletonTrainDataset(Dataset):
    def __init__(self, args, mode = 'x-sub'):
        '''
        Initialize Skeleton train dataset. mode should be 'x-sub' or 'x-set'.

        x-sub: Use `cross subject` to split dataset, which means different performers are used in train dataset and validation dataset.
        x-view:  Use `cross view` to split dataset, which means different camera views are used in train dataset and validation dataset.

        '''
        super().__init__()
        
        self.seg = args.seg
        dataset_path = args.skeleton_dataset_xsub_train_dir if mode =='x-sub' else args.skeleton_dataset_xset_train_dir

        with open(args.skeleton_label_dir,'r') as f:
            self.labels = f.readlines()
        
        with open(dataset_path,'rb') as f:

            self.original_data = pickle.load(f)

        logging.info('Skeleton train dataset loading complete. Split mode: '+mode)

    def sgn_transform(self, batch):

        X,y,feats = zip(*batch)
  
        N = len(X)

        C, T, V, M = X[0].size()
        new_X = torch.tensor([])
        
        for i in range(N):
            # shape: C,T,V,M -> M,T,V,C
            Xt = X[i].permute(3,1,2,0).contiguous()
            # shape: C,T,V,M -> M,T,V*C
            Xt = Xt.view(M,T,V*C)
            
            # 对于M维度的数据进行判断，如果是被填充的数据(0)则去掉，否则在T维度拼接。
            # Trick: 由于填充的数据均在原数据之后，所以只需判断第二组是否全为0
            
            if not (Xt[1,:,:] == torch.zeros(T,V*C)).all():
                
                Xt_concat = torch.concat([Xt[0,:,:], Xt[1,:,:]],dim = 0) # shape: 2*T, V*C
            
            else:
                Xt_concat = Xt[0,:,:]# shape: T, V*C
                
            # 去除T维度上的0填充，并根据设定的时间间隔(seg)抽帧采样或0填充
            idx = torch.all(Xt_concat,dim = 1)
            
            Xt_nonzero = Xt_concat[idx,:]
            # 对于不足seg的0填充
            Xt_nonzero_pad = torch.nn.functional.pad(Xt_nonzero,(0,0,0,self.seg - Xt_nonzero.shape[0]))# shape: >=seg,V*C
            # 抽帧采样到seg
            Xt_sampled  = torch.nn.functional.interpolate(Xt_nonzero_pad.unsqueeze(0).unsqueeze(0),size = (self.seg, V*C))[0]
            new_X = torch.concat([new_X,Xt_sampled],dim = 0)
            
            # 堆叠feats
            fs = {'img_shape': torch.tensor([[*(i['img_shape'])] for i in feats]) ,'real_label':[i['real_label'][:-1] for i in feats] }

        # 数据增强
        new_X = _transform(new_X, 0.3)

        return [new_X,torch.tensor([*y]), fs]
            
    def __getitem__(self, idx):
        
        shape = self.original_data[idx]['img_shape']

        keypoints = (self.original_data[idx]['keypoint']/shape).transpose(3,1,2,0)

        keypoints = np.pad(keypoints,((0,0),(0,300 - keypoints.shape[1]), (0,0),(0,2 - keypoints.shape[-1])))
        
        X = torch.tensor(keypoints,dtype = torch.float32)

        y = torch.tensor(self.original_data[idx]['label'],dtype = torch.long)
       
        feats = {'img_shape':shape,'real_label':self.labels[self.original_data[idx]['label']]}
        
        return X, y, feats
    
    
    def __len__(self):
        
        return len(self.original_data)


class SkeletonValDataset(Dataset):
    def __init__(self, args, mode = 'x-sub'):
        '''
        Initialize Skeleton validation dataset. mode should be 'x-sub' or 'x-set'.

        x-sub: Use `cross subject` to split dataset, which means different performers are used in train dataset and validation dataset.
        x-view:  Use `cross view` to split dataset, which means different camera views are used in train dataset and validation dataset.

        '''
        super().__init__()
        self.seg = args.seg
        dataset_path = args.skeleton_dataset_xsub_val_dir if mode =='x-sub' else args.skeleton_dataset_xsub_val_dir

        with open(args.skeleton_label_dir,'r') as f:
            self.labels = f.readlines()
        
        with open(dataset_path,'rb') as f:

            self.original_data = pickle.load(f)

        logging.info('Skeleton validation dataset loading complete. Split mode: '+mode)
        
    def sgn_transform(self, batch):
        
        X,y,feats = zip(*batch)
  
        N = len(X)

        C, T, V, M = X[0].size()
        new_X = torch.tensor([])
        
        for i in range(N):
            # shape: C,T,V,M -> M,T,V,C
            Xt = X[i].permute(3,1,2,0).contiguous()
            # shape: C,T,V,M -> M,T,V*C
            Xt = Xt.view(M,T,V*C)
            
            # 对于M维度的数据进行判断，如果是被填充的数据(0)则去掉，否则在T维度拼接。
            # Trick: 由于填充的数据均在原数据之后，所以只需判断第二组是否全为0
            
            if not (Xt[1,:,:] == torch.zeros(T,V*C)).all():
                
                Xt_concat = torch.concat([Xt[0,:,:], Xt[1,:,:]],dim = 0) # shape: 2*T, V*C
            
            else:
                Xt_concat = Xt[0,:,:]# shape: T, V*C
                
            # 去除T维度上的0填充，并根据设定的时间间隔(seg)抽帧采样或0填充
            idx = torch.all(Xt_concat,dim = 1)
            
            Xt_nonzero = Xt_concat[idx,:]
            # 对于不足seg的0填充
            Xt_nonzero_pad = torch.nn.functional.pad(Xt_nonzero,(0,0,0,self.seg - Xt_nonzero.shape[0]))# shape: >=seg,V*C
            # 抽帧采样到seg
            Xt_sampled  = torch.nn.functional.interpolate(Xt_nonzero_pad.unsqueeze(0).unsqueeze(0),size = (self.seg, V*C))[0]
            new_X = torch.concat([new_X,Xt_sampled],dim = 0)
            
            # 堆叠feats
            fs = {'img_shape': torch.tensor([[*(i['img_shape'])] for i in feats]) ,'real_label':[i['real_label'][:-1] for i in feats] }

        return [new_X,torch.tensor([*y]), fs]
    
        
    def __getitem__(self, idx):
        
        shape = self.original_data[idx]['img_shape']

        keypoints = (self.original_data[idx]['keypoint']/shape).transpose(3,1,2,0)

        keypoints = np.pad(keypoints,((0,0),(0,300 - keypoints.shape[1]), (0,0),(0,2 - keypoints.shape[-1])))
        
        X = torch.tensor(keypoints,dtype = torch.float32)
    
        y = torch.tensor(self.original_data[idx]['label'],dtype = torch.long)
       
        feats = {'img_shape':shape,'real_label':self.labels[self.original_data[idx]['label']]}
        
        return X, y, feats
    
    
    def __len__(self):
        
        return len(self.original_data)
    
def _rot(rot):
    cos_r, sin_r = rot.cos(), rot.sin()
    zeros = rot.new(rot.size()[:2] + (1,)).zero_()
    ones = rot.new(rot.size()[:2] + (1,)).fill_(1)

    r1 = torch.stack((ones, zeros),dim=-1)
    rx2 = torch.stack((zeros, cos_r[:,:,0:1]), dim = -1)
    rx = torch.cat((r1, rx2), dim = 2)

    ry1 = torch.stack((cos_r[:,:,1:2], zeros), dim =-1)
    r2 = torch.stack((zeros, ones),dim=-1)

    ry = torch.cat((ry1, r2), dim = 2)

    rot = ry.matmul(rx)

    return rot

def _transform(x, theta, rate = 0.3):
    x = x.contiguous().view(x.size()[:2] + (-1, 2))
    rot = x.new(x.size()[0],2).uniform_(-theta, theta)
    rot = rot.repeat(1, x.size()[1])
    rot = rot.contiguous().view((-1, x.size()[1], 2))
    rot = _rot(rot)
    x = torch.transpose(x, 2, 3)
    x = torch.matmul(rot, x)
    x = torch.transpose(x, 2, 3)

    x = x.contiguous().view(x.size()[:2] + (-1,))
    return x