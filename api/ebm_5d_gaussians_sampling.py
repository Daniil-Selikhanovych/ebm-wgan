import torch
import numpy as np
import random
import time
import os

from utils import (prepare_gaussians, 
                   prepare_train_batches, 
                   visualize_fake_data_projection)

from paths import (path_to_save, 
                   path_to_5d_gaussian_discriminator,
                   path_to_5d_gaussian_generator)

from ebm_sampling import Langevin_sampling, MALA_sampling, NUTS_sampling

from wgan_fully_connected_models import (Generator_fully_connected, 
                                         Discriminator_fully_connected)

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

BATCH_SIZE = 256
num_samples_in_cluster = 1000
dim = 5
num_gaussian_per_dim = 3
coord_limits = 4.0
std = 0.05
X_train = prepare_gaussians(num_samples_in_cluster, dim,
                            num_gaussian_per_dim, coord_limits,
                            std)

z_dim = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
G = Generator_fully_connected(n_dim = z_dim, n_output = z_dim).to(device)
D = Discriminator_fully_connected(n_in = z_dim).to(device)

print("Load models")
G.load_state_dict(torch.load(path_to_5d_gaussian_generator))
D.load_state_dict(torch.load(path_to_5d_gaussian_discriminator))

for p in D.parameters():  
    p.requires_grad = False
for p in G.parameters():  
    p.requires_grad = False

batch_size_sample = 5000

proj_list = [[0, 1], [2, 3], [0, 4]]

title = f"Training and generated samples, num samples = {batch_size_sample}"
mode = f"downloaded_5d_gaussians"

fake_data = G.sampling(batch_size_sample).data.cpu().numpy()
path_to_save_plots = os.path.join(path_to_save, 'plots')

print("Start to sample from simple generator")
for i in range(len(proj_list)):
    visualize_fake_data_projection(fake_data, X_train, path_to_save_plots, 
                                   proj_list[i][0], proj_list[i][1],
                                   title,
                                   mode)

print("Start to sample from Langevin dynamics")
eps_arr = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
for i in range(len(eps_arr)):
    cur_eps = eps_arr[i]
    print(f"Langevin dynamics with eps = {cur_eps}")
    start_time = time.time()
    latent_arr = Langevin_sampling(G, D, 
                                   z_dim, cur_eps, 
                                   batch_size_sample, device)
    end_time = time.time()
    calc_time = end_time - start_time
    print(f"Time calculation = {round(calc_time, 2)} seconds")
    fake_data = G(latent_arr).data.cpu().numpy()
    title = fr"Langevin sampling, $\varepsilon$ = {cur_eps}, num samples = {batch_size_sample}"
    mode = f"langevin_eps_{cur_eps}_5d_gaussians"
    for j in range(len(proj_list)):
        visualize_fake_data_projection(fake_data, X_train, path_to_save_plots,
                                       proj_list[j][0], proj_list[j][1],
                                       title,
                                       mode)

print("Start to sample with MALA")
num_iter = 2*batch_size_sample
for i in range(len(eps_arr)):
    cur_eps = eps_arr[i]      
    print(f"MALA with eps = {cur_eps}")
    start_time = time.time()
    latent_arr = MALA_sampling(G, D,
                               z_dim, cur_eps,
                               num_iter, device)
    num_mala_samples = latent_arr.size()[0]
    end_time = time.time()
    calc_time = end_time - start_time 
    print(f"For {num_iter} iterations accepted samples = {num_mala_samples}")
    print(f"Time calculation = {round(calc_time, 2)} seconds")
    fake_data = G(latent_arr).data.cpu().numpy()
    title = fr"MALA, $\varepsilon$ = {cur_eps}, iterations = {num_iter}, samples = {num_mala_samples}"
    mode = f"mala_eps_{cur_eps}_5d_gaussians"
    for j in range(len(proj_list)):
        visualize_fake_data_projection(fake_data, X_train, path_to_save_plots,
                                       proj_list[j][0], proj_list[j][1],
                                       title,
                                       mode)

print("Start to sample with NUTS")
start_time = time.time()
latent_arr, mcmc = NUTS_sampling(G, D, z_dim, batch_size_sample, device)
end_time = time.time()
calc_time = end_time - start_time
print(f"Time calculation = {round(calc_time, 2)} seconds")
fake_data = G(latent_arr).data.cpu().numpy()
title = fr"NUTS, samples = {batch_size_sample}"
mode = f"nuts_5d_gaussians"
for j in range(len(proj_list)):
    visualize_fake_data_projection(fake_data, X_train, path_to_save_plots,
                                   proj_list[j][0], proj_list[j][1],
                                   title,
                                   mode)
