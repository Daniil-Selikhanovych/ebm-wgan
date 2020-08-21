import numpy as np
import random
import torch, torch.nn as nn

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
device_default = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Generator_2d(nn.Module):
    def __init__(self, 
                 n_dim = 2,
                 n_hidden = 256, 
                 device = device_default, 
                 non_linear = nn.ReLU()):
        super(Generator_2d, self).__init__()
        self.non_linear = non_linear
        self.device = device
        self.n_hidden = n_hidden
        self.n_dim = n_dim
        self.layers = nn.ModuleList([nn.Linear(self.n_dim, self.n_hidden),
                                     nn.Linear(self.n_hidden, self.n_hidden), 
                                     nn.Linear(self.n_hidden, self.n_hidden),
                                     nn.Linear(self.n_hidden, 2)])
        self.num_layer = len(self.layers)
        #for i in range(4):
        #    std_init = 0.8 * (2/self.layers[i].in_features)**0.5
        #    torch.nn.init.normal_(self.layers[i].weight, std = std_init)
            
    def make_hidden(self, batch_size):
        return torch.randn(batch_size, self.n_dim, device = self.device)

    def forward(self, z):
        for i in range(self.num_layer - 1):
            z = self.non_linear((self.layers[i])(z))
        z = (self.layers[self.num_layer - 1])(z)
        return z

    def sampling(self, batch_size):
        z = self.make_hidden(batch_size)
        #print(z.detach().cpu())
        return self.forward(z)
        
class Discriminator_2d(nn.Module):
    def __init__(self, 
                 n_hidden = 512,
                 device = device_default, 
                 non_linear = nn.ReLU()):
        super(Discriminator_2d, self).__init__()
        self.non_linear = non_linear
        self.device = device
        self.n_hidden = n_hidden
        self.layers = nn.ModuleList([nn.Linear(2, self.n_hidden),
                                     nn.Linear(self.n_hidden, self.n_hidden), 
                                     nn.Linear(self.n_hidden, self.n_hidden),
                                     nn.Linear(self.n_hidden, 1)])
        self.num_layer = len(self.layers)
        #for i in range(4):
        #    std_init = 0.8 * (2/self.layers[i].in_features)**0.5
        #    torch.nn.init.normal_(self.layers[i].weight, std = std_init)
      
    def forward(self, z):
        for i in range(self.num_layer - 1):
            z = self.non_linear((self.layers[i])(z))
        z = (self.layers[self.num_layer - 1])(z)
        return z
        
def weights_init_1(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        
def weights_init_2(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        std_init = 0.8 * (2/m.in_features)**0.5
        m.weight.data.normal_(0.0, std = std_init)
