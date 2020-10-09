import numpy as np
import random
import torch, torch.nn as nn
from torch.autograd import Variable
from torch import autograd
import time
import datetime
import os

from utils import prepare_train_batches, epoch_visualization

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
device_default = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calc_gradient_penalty(D, real_data, fake_data, batch_size, Lambda = 0.1,
                          device = device_default):
    #print(real_data.shape)
    #print(fake_data.shape)
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(real_data.size()).to(device)

    interpolates = alpha * real_data + (1 - alpha) * fake_data
    interpolates = interpolates.to(device)
    interpolates = autograd.Variable(interpolates, 
                                     requires_grad = True)
    discriminator_interpolates = D(interpolates)
    ones = torch.ones(discriminator_interpolates.size()).to(device)
    gradients = autograd.grad(outputs = discriminator_interpolates, 
                              inputs = interpolates,
                              grad_outputs = ones,
                              create_graph = True, 
                              retain_graph = True, 
                              only_inputs = True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1.0) ** 2).mean() * Lambda
    return gradient_penalty
    
def train_wgan(X_train,
               X_train_batches, 
               generator, g_optimizer, 
               discriminator, d_optimizer,
               path_to_save,
               batch_size = 256,
               device = device_default,
               use_gradient_penalty = True,
               Lambda = 0.1,
               num_epochs = 20000, 
               num_epoch_for_save = 100,
               batch_size_sample = 5000,
               proj_list = None):

    k_g = 1
    generator_loss_arr = []
    generator_mean_loss_arr = []
    discriminator_loss_arr = []
    discriminator_mean_loss_arr = []
    one = torch.tensor(1, dtype = torch.float).to(device)
    mone = one * -1
    mone = mone.to(device)  
    path_to_save_models = os.path.join(path_to_save, 'models')
    path_to_save_plots = os.path.join(path_to_save, 'plots')

    try:
        for epoch in range(num_epochs):
            print(f"Start epoch = {epoch}")
            if epoch < 25:
                k_d = 100
            else: 
                k_d = 10

            for p in discriminator.parameters():  # reset requires_grad
                p.requires_grad = True
            
            start_time = time.time()
            # Optimize D
            # discriminator.train(True)
            # generator.train(False)
            for _ in range(k_d):
                # Sample noise
                # real_data = sample_data_batch(batch_size, 
                #                               X_train, 
                #                               device)
                discriminator.zero_grad()
                real_data = next(X_train_batches)
                if (real_data.shape[0] != batch_size):
                   continue

                real_data = torch.Tensor(real_data)
                real_data = autograd.Variable(real_data).to(device)
                
                d_real_data = discriminator(real_data).mean()
                d_real_data.backward(mone)   
                
                noise = generator.make_hidden(batch_size)
                #with torch.no_grad():
                noise = autograd.Variable(noise).to(device)
                #print(noise.size())
                fake_data = generator(noise)
                d_fake_data = discriminator(fake_data).mean()
                d_fake_data.backward(one)

                d_loss = d_fake_data - d_real_data
                #print("OK")
                if use_gradient_penalty:
                    gradient_penalty = calc_gradient_penalty(discriminator, 
                                                             real_data.data, 
                                                             fake_data.data, 
                                                             batch_size,
                                                             Lambda)
                    gradient_penalty.backward()
                    d_loss += gradient_penalty
                d_optimizer.step()
                discriminator_loss_arr.append(d_loss.data.cpu().numpy())

            #discriminator.train(False)
            #generator.train(True)
            # Optimize G
            for p in discriminator.parameters():  # to avoid computation
                p.requires_grad = False

            for _ in range(k_g):
                g_optimizer.zero_grad()

                # Do an update
                noise = generator.make_hidden(batch_size)
                noise = autograd.Variable(noise).to(device)
                fake_data = generator(noise)

                generator_loss = discriminator(fake_data).mean()
                generator_loss.backward(mone)
                generator_loss = -generator_loss
                g_optimizer.step()
                generator_loss_arr.append(generator_loss.data.cpu().numpy())
           
            end_time = time.time()
            calc_time = end_time - start_time
            discriminator_mean_loss_arr.append(np.mean(discriminator_loss_arr[-k_d :]))
            generator_mean_loss_arr.append(np.mean(generator_loss_arr[-k_g :]))
            print("Epoch {} of {} took {:.3f}s".format(
                   epoch + 1, num_epochs, calc_time))
            print("Discriminator last mean loss: \t{:.6f}".format(
                   discriminator_mean_loss_arr[-1]))
            print("Generator last mean loss: \t{:.6f}".format(
                   generator_mean_loss_arr[-1])) 
            if epoch % num_epoch_for_save == 0:
               # Visualize
               epoch_visualization(X_train, generator, 
                                   use_gradient_penalty, 
                                   discriminator_mean_loss_arr, 
                                   epoch, Lambda,
                                   generator_mean_loss_arr, 
                                   path_to_save_plots,
                                   batch_size_sample,
                                   proj_list)
               cur_time = datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')

               discriminator_model_name = cur_time + '_discriminator.pth'
               generator_model_name = cur_time + '_generator.pth'

               path_to_discriminator = os.path.join(path_to_save_models, discriminator_model_name)
               path_to_generator = os.path.join(path_to_save_models, generator_model_name)

               torch.save(discriminator.state_dict(), path_to_discriminator)
               torch.save(generator.state_dict(), path_to_generator)
                

    except KeyboardInterrupt:
        pass
