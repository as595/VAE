import torch
import torch.nn as nn

# --------------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, dim):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)
        
# --------------------------------------------------------------------------
        
class OordEncoder(nn.Module):
    def __init__(self, n_chan, zdim, hidden=128):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(n_chan, hidden, 4, 2, 1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(True),
            nn.Conv2d(hidden, zdim, 4, 2, 1),
            nn.BatchNorm2d(zdim),
            ResBlock(zdim),
            ResBlock(zdim)
        )

        self.to_latent = nn.Conv2d(zdim, zdim, 1, 1)
        
    def forward(self, x):

        x = self.layers(x)
        
        mu = self.to_latent(x)
        logvar = self.to_latent(x)
        
        return mu, logvar
        
# --------------------------------------------------------------------------

class OordDecoder(nn.Module):
    def __init__(self, n_chan, zdim, hidden=128):
        super().__init__()

        self.layers = nn.Sequential(
            ResBlock(zdim),
            ResBlock(zdim),
            nn.ReLU(True),
            nn.ConvTranspose2d(zdim, hidden, 4, 2, 1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden, n_chan, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.layers(x)

# --------------------------------------------------------------------------