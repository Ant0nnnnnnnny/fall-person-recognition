import torch
import torch.nn as nn

from .st_gcn import STGCN

class Model(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

        self.origin_stream = STGCN(*args, **kwargs)
        self.motion_stream = STGCN(*args, **kwargs)

    def forward(self, x):
        N, C, T, V, M = x.size()
        m = torch.cat((torch.cuda.FloatTensor(N, C, 1, V, M).zero_(),
                        x[:, :, 1:-1] - 0.5 * x[:, :, 2:] - 0.5 * x[:, :, :-2],
                        torch.cuda.FloatTensor(N, C, 1, V, M).zero_()), 2)

        res = self.origin_stream(x) + self.motion_stream(m)
        return res