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


torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

train_dataset_size = 100000
BATCH_SIZE = 256            
X_train = prepare_25gaussian_data(train_dataset_size)
X_train_batches = prepare_train_batches(X_train, BATCH_SIZE) 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
G = Generator_2d(n_dim = 2).to(device)
D = Discriminator_2d().to(device)
G.apply(weights_init_2)
D.apply(weights_init_2)

lr_init = 1e-4
d_optimizer = torch.optim.Adam(D.parameters(), betas = (0.5, 0.9), lr = lr_init)
g_optimizer = torch.optim.Adam(G.parameters(), betas = (0.5, 0.9), lr = lr_init)
use_gradient_penalty = True
Lambda = 0.1
num_epochs = 20000
num_epoch_for_save = 500
batch_size_sample = 5000     

print("Start to train WGAN")

train_wgan(X_train,
           X_train_batches, 
           G, g_optimizer, 
           D, d_optimizer,
           path_to_save,
           BATCH_SIZE,
           device,
           use_gradient_penalty,
           Lambda,
           num_epochs, 
           num_epoch_for_save,
           batch_size_sample)
