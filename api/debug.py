import numpy as np
import sklearn.datasets
import time
import random

from matplotlib import pyplot as plt

import torch, torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch import autograd

from paths import path_to_save
from utils import prepare_25gaussian_data, prepare_train_batches
from wgan_2d_models import Generator_2d, Discriminator_2d, weights_init_1, weights_init_2
from wgan_train import train_wgan

one = torch.tensor(1, dtype=torch.float)
mone = one * -1

device = torch.device('cpu')
train_dataset_size = 100000
BATCH_SIZE = 256            
X_train = prepare_25gaussian_data(train_dataset_size)
X_train_batches = prepare_train_batches(X_train, BATCH_SIZE) 

G = Generator_2d(n_dim = 2, device = device).to(device)
D = Discriminator_2d(device = device).to(device)

_data = next(X_train_batches)
real_data = torch.Tensor(_data)
real_data_v = autograd.Variable(real_data)

D.zero_grad()

        # train with real
D_real = D(real_data_v)
D_real = D_real.mean()
D_real.backward(mone)

noise = G.make_hidden(BATCH_SIZE)
with torch.no_grad():
   noise = autograd.Variable(noise)
#print(noise.size())
fake_data = G(noise).to(device)
