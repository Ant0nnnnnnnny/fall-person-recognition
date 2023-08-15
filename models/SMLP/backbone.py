import torch.nn as nn
import torch

class SMLP(nn.Module):

    def __init__(self, args) -> None:
        '''
        input shape:[N, T, V, C, 2]
        '''
        super().__init__()

        self.initial_adj()
        self.first_layer = torch.nn.Linear(in_features=2, out_features=16)
        self.layers = nn.ModuleList([
            MLPBlock(32, self.A, [64,128,256,256,128,64,32,64]),
            MLPBlock(32, self.A,[32,128,256,256,128,64,32,32]),
            MLPBlock(32, self.A,[32,128,256,256,128,64,32,32]),
            MLPBlock(32, self.A,[32,128,256,256,128,64,32,32]),
            MLPBlock(32, self.A,[32,128,256,256,128,64,32,32]),
            MLPBlock(32, self.A,[32,128,256,256,128,64,32,32]),
            MLPBlock(32, self.A, [32,128,32])
        ])
        self.data_bn = nn.BatchNorm1d(51)
        self.first_layer = torch.nn.Linear(in_features=3, out_features=32)
        
        self.final_layer = torch.nn.Linear(in_features=32, out_features=120)
        
        self.time_embedding = nn.ModuleList([nn.Linear(in_features = 300,out_features = 64),nn.Linear(in_features = 64,out_features = 1)])
        
        self.spatial_embedding = nn.ModuleList([nn.Linear(in_features = 17,out_features = 8),nn.Linear(in_features = 8,out_features = 1)])
        
        self.activation = nn.Hardswish(inplace=True)

    def forward(self, x):
      
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)
       
        y = self.time_embedding[0](x.permute(0,1,3,2))
       
        y = self.time_embedding[1](y)
        
        y = torch.squeeze(y, dim = 3)
        
        # y = torch.matmul(y,self.A)
        
        y = self.spatial_embedding[0](y)
        
        y = self.spatial_embedding[1](y).squeeze(dim = 2)
      
        y = self.first_layer(y)
        
        for layer in self.layers:
        
            y = layer(y)
            y = self.activation(y)
        y = self.final_layer(y)
        y = torch.matmul(torch.ones((N,2*N)) * 0.5, y)
        return y
    def initial_adj(self) -> None:

        self.register_buffer('A', torch.tensor( [[0.3333, 0.2500, 0.2500, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.3333, 0.2500, 0.2500, 0.3333, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.3333, 0.2500, 0.2500, 0.0000, 0.3333, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.2500, 0.0000, 0.3333, 0.0000, 0.2000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.2500, 0.0000, 0.3333, 0.0000, 0.2000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.3333, 0.0000, 0.2000, 0.2000, 0.3333, 0.0000,
         0.0000, 0.0000, 0.2500, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.3333, 0.2000, 0.2000, 0.0000, 0.3333,
         0.0000, 0.0000, 0.0000, 0.2500, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.2000, 0.0000, 0.3333, 0.0000,
         0.5000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.2000, 0.0000, 0.3333,
         0.0000, 0.5000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.3333, 0.0000,
         0.5000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.3333,
         0.0000, 0.5000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.2000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.2500, 0.2500, 0.3333, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.2000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.2500, 0.2500, 0.0000, 0.3333, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.2500, 0.0000, 0.3333, 0.0000, 0.5000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.2500, 0.0000, 0.3333, 0.0000, 0.5000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.3333, 0.0000, 0.5000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.3333, 0.0000, 0.5000]]))
        
        
class MLPBlock(torch.nn.Module):
    def __init__(self, input_shape, A, squeeze_dims = []) -> None:
        '''
        input shape:[N, T, V, C, 2]
        '''
        super().__init__()
        self.register_buffer('A',A)
        
        self.layers = nn.ModuleList(
            [torch.nn.Linear(in_features = input_dim,out_features = output_dim,) for input_dim,output_dim in zip([input_shape,* squeeze_dims],  squeeze_dims)])
        self.final_layer = torch.nn.Linear(in_features = squeeze_dims[-1], out_features = input_shape)
        self.activation = torch.nn.Hardswish(inplace=True)
    def forward(self, x):
       
        y = x
        
        for layer in self.layers:
            y = layer(y)
            y = self.activation(y)
        y = self.final_layer(y)
        y = y + x
        return y