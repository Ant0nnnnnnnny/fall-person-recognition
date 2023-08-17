import torch
class OneEuroFilter:
    def __init__(self, te, mincutoff=1.0, beta=0.007, dcutoff=1.0):
        self.x = None
        self.dx = 0
        self.te = te
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.alpha = self._alpha(self.mincutoff)
        self.dalpha = self._alpha(self.dcutoff)

    def _alpha(self, cutoff):
        tau = 1.0 / (2 * torch.pi * cutoff)
        return 1.0 / (1.0 + tau / self.te)

    def predict(self, x, te):
        result = x
        if self.x is not None:
            edx = (x - self.x) / te
            self.dx = self.dx + (self.dalpha * (edx - self.dx))
            cutoff = self.mincutoff + self.beta * abs(self.dx)
            self.alpha = self._alpha(cutoff)
            result = self.x + self.alpha * (x - self.x)
        self.x = result
        return result