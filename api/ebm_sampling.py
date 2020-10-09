import torch, torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Normal, Independent, Uniform

import pyro
from pyro.infer import MCMC, NUTS, HMC

from functools import partial

def Langevin_sampling(generator, discriminator, 
                      z_dim, eps, num_iter, device):
   loc = torch.zeros(z_dim).to(device)
   scale = torch.ones(z_dim).to(device)
   normal = Normal(loc, scale)
   diagn = Independent(normal, 1)
   cur_z = diagn.sample()
   cur_z.requires_grad_(True)
   latent_arr = [cur_z.clone()]
   for i in range(num_iter - 1):
      GAN_part = -discriminator(generator(cur_z))
      latent_part = -diagn.log_prob(cur_z)
      energy = GAN_part + latent_part 
      energy.backward()
      noise = diagn.sample()
      with torch.no_grad():
         cur_z -= (eps/2*cur_z.grad - (eps ** 0.5)*noise)
      latent_arr.append(cur_z.clone())
   latent_arr = torch.stack(latent_arr, dim = 0)
   return latent_arr

def MALA_sampling(generator, discriminator, 
                  z_dim, eps, num_iter, device):
   loc = torch.zeros(z_dim).to(device)
   scale = torch.ones(z_dim).to(device)
   normal = Normal(loc, scale)
   diagn = Independent(normal, 1)
   uniform_sampler = Uniform(low = 0.0, high = 1.0)
   cur_z = diagn.sample()
   cur_z.requires_grad_(True)
   latent_arr = [cur_z.clone()]
   for i in range(num_iter):
      GAN_part = -discriminator(generator(cur_z))
      latent_part = -diagn.log_prob(cur_z)
      cur_energy = GAN_part + latent_part 
      cur_energy.backward()
      noise = diagn.sample()
      gamma = eps/2
      with torch.no_grad():
         new_z = (cur_z - gamma*cur_z.grad + (eps ** 0.5)*noise)
      new_z = new_z.clone()
      new_z.requires_grad_(True)
      new_energy = -discriminator(generator(new_z)) - diagn.log_prob(new_z)
      new_energy.backward()
      energy_part = cur_energy - new_energy
      with torch.no_grad():
         vec_for_propose_2 = cur_z - new_z + gamma*new_z.grad
      propose_part_2 = (vec_for_propose_2 @ vec_for_propose_2)/4.0/gamma
      propose_part = (noise @ noise)/2.0 - propose_part_2
      log_accept_prob = propose_part + energy_part
      generate_uniform_var = uniform_sampler.sample().to(device)
      log_generate_uniform_var = torch.log(generate_uniform_var)
      if log_generate_uniform_var < log_accept_prob:
          latent_arr.append(new_z.clone())
          cur_z = new_z
          cur_z.grad.data.zero_()

   latent_arr = torch.stack(latent_arr, dim = 0)
   return latent_arr

def calculate_energy(params, generator, discriminator):
   if params is not None:
      cur_params = params['points']
      return -discriminator(generator(cur_params)) + (cur_params @ cur_params)/2
   else:
      return torch.tensor([0.0])

def NUTS_sampling(generator, discriminator, z_dim, num_samples, device):
   cur_calculate_energy = partial(calculate_energy, 
                                  generator = generator,
                                  discriminator = discriminator)
   kernel = NUTS(potential_fn = cur_calculate_energy)
   loc = torch.zeros(z_dim).to(device)
   scale = torch.ones(z_dim).to(device)
   normal = Normal(loc, scale)
   diagn = Independent(normal, 1)
   init_params = diagn.sample()
   init_params = {'points': init_params}
   mcmc = MCMC(kernel = kernel, 
               num_samples = num_samples, 
               initial_params = init_params,
               num_chains = 1)
   mcmc.run()
   latent_arr = mcmc.get_samples()['points']
   return latent_arr, mcmc
