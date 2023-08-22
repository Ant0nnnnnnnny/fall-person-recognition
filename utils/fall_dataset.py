from torch.utils.data import Dataset
import logging
import pickle
import torch
import numpy as np

class FallTrainDataset(Dataset):
    def __init__(self, args):
        '''
        Initialize Fall train dataset. 
        '''
        super().__init__()
        
        self.seg = args.seg
        X_path = args.fall_train_dataset_dir
        y_path = args.fall_train_label_dir

        with open(X_path,'rb') as f:

            self.X = pickle.load(f)

        with open(y_path,'rb') as f:

            self.y = pickle.load(f)
        
        logging.info('Fall train dataset loading complete. Total: '+ str(len(self.X))+' samples')

    def sgn_transform(self, batch):

        X,y,feats= zip(*batch)
  
        N = len(X)

        T,V,C = X[0].size()
        new_X = torch.tensor([])
        
        for i in range(N):

            Xt_nonzero_pad = torch.nn.functional.pad(new_X[i],(0,0,0,self.seg - new_X[i].shape[0]))# shape: >=seg,V*C
            # 抽帧采样到seg
            Xt_sampled  = torch.nn.functional.interpolate(Xt_nonzero_pad.unsqueeze(0).unsqueeze(0),size = (self.seg, V*C))[0]
            new_X = torch.concat([new_X,Xt_sampled],dim = 0)
        
        # 数据增强
        # new_X = _transform(new_X, 0.3)
        fs = {'real_label':[i['real_label'][:-1] for i in feats] }

        return [new_X,torch.tensor([*y]),fs]
    
    def __getitem__(self, idx):

        item = self.X[idx]

        shape = np.max(np.max(item,axis = 1),axis = 0)

        keypoints = item/shape

        X = torch.tensor(keypoints,dtype = torch.float32)

        y = torch.tensor(self.y[idx],dtype = torch.long)

        feat = {'real_label':'fall' if y == 1 else 'other'}
       
        return X, y,feat
    
    
    def __len__(self):
        
        return len(self.X)


class FallValDataset(Dataset):
    def __init__(self, args):
        '''
        Initialize Fall validate dataset.  
        '''
        super().__init__()
        
        self.seg = args.seg
        X_path = args.fall_val_dataset_dir
        y_path = args.fall_val_label_dir

        with open(X_path,'rb') as f:

            self.X = pickle.load(f)

        with open(y_path,'rb') as f:

            self.y = pickle.load(f)
        
        logging.info('Fall val dataset loading complete. Total: '+ str(len(self.X))+' samples')

    def sgn_transform(self, batch):

        X,y,feats= zip(*batch)
  
        N = len(X)

        T,V,C = X[0].size()
        new_X = torch.tensor([])
        
        for i in range(N):

            Xt = X[0].view(T,V*C)

            idx = torch.all(Xt,dim = 1)
            
            Xt_nonzero = Xt[idx,:]
            # 对于不足seg的0填充
            Xt_nonzero_pad = torch.nn.functional.pad(Xt_nonzero,(0,0,0,self.seg - Xt_nonzero.shape[0]))# shape: >=seg,V*C
            # 抽帧采样到seg
            Xt_sampled  = torch.nn.functional.interpolate(Xt_nonzero_pad.unsqueeze(0).unsqueeze(0),size = (self.seg, V*C))[0]
            new_X = torch.concat([new_X,Xt_sampled],dim = 0)
        
        fs = {'real_label':[i['real_label'][:-1] for i in feats] }

        return [new_X,torch.tensor([*y]),fs]
            
    def __getitem__(self, idx):

        item = self.X[idx]

        shape = np.max(np.max(item,axis = 1),axis = 0)

        keypoints = item/shape

        X = torch.tensor(keypoints,dtype = torch.float32)

        y = torch.tensor(self.y[idx],dtype = torch.long)
        
        feat = {'real_label':'fall' if y == 1 else 'other'}

        return X, y,feat
    
    
    def __len__(self):
        
        return len(self.X)


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