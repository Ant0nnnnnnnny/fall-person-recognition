import torch
class SkeletonHead(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim = 1)
        
    def forward(self,x):
        return self.softmax(x)