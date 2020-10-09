import numpy as np
import sklearn.datasets
import os
import matplotlib.pyplot as plt
import datetime

def prepare_swissroll_data(BATCH_SIZE=1000):
    data = sklearn.datasets.make_swiss_roll(
                    n_samples=BATCH_SIZE,
                    noise=0.25
                )[0]
    data = data.astype('float32')[:, [0, 2]]
    data /= 7.5 # stdev plus a little
    return data

def prepare_25gaussian_data(BATCH_SIZE=1000):
    dataset = []
    for i in range(BATCH_SIZE//25):
        for x in range(-2, 3):
            for y in range(-2, 3):
                point = np.random.randn(2)*0.05
                point[0] += 2*x
                point[1] += 2*y
                dataset.append(point)
    dataset = np.array(dataset, dtype=np.float32)
    np.random.shuffle(dataset)
    dataset /= 2.828 # stdev
    return dataset 

def prepare_gaussians(num_samples_in_cluster, dim, 
                      num_gaussian_per_dim, coord_limits, 
                      std = 0.1, random_state = 42,
                      scale = 2.828):
    num_clusters = num_gaussian_per_dim ** dim
    num_samples = num_samples_in_cluster * num_clusters
    coords_per_dim = np.linspace(-coord_limits, 
                                 coord_limits, 
                                 num = num_gaussian_per_dim)
    copy_coords = list(np.tile(coords_per_dim, (dim, 1)))
    centers = np.array(np.meshgrid(*copy_coords)).T.reshape(-1, dim)
    dataset = sklearn.datasets.make_blobs(n_samples = num_samples, 
                                          n_features = dim, 
                                          centers = centers, 
                                          cluster_std = std,
                                          random_state = random_state)[0]
    dataset /= scale
    return dataset

def prepare_train_batches(dataset, BATCH_SIZE):
    while True:
        for i in range(len(dataset) // BATCH_SIZE):
            yield dataset[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]

def sample_fake_data(generator, X_train, epoch, path_to_save, batch_size_sample = 5000):
    fake_data = generator.sampling(batch_size_sample).data.cpu().numpy()
    plt.figure(figsize=(8, 8))
    plt.xlim(-2., 2.)
    plt.ylim(-2., 2.)
    plt.title("Training and generated samples", fontsize=20)
    plt.scatter(X_train[:,:1], X_train[:,1:], alpha=0.5, color='gray', 
                marker='o', label = 'training samples')
    plt.scatter(fake_data[:,:1], fake_data[:,1:], alpha=0.5, color='blue', 
                marker='o', label = 'samples by G')
    plt.legend()
    plt.grid(True)
    if path_to_save is not None:
       cur_time = datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
       plot_name = cur_time + f'_wgan_sampling_{epoch}_epoch.pdf'
       path_to_plot = os.path.join(path_to_save, plot_name)
       plt.savefig(path_to_plot)

    else:
       plt.show()

def visualize_fake_data_projection(fake_data, X_train, path_to_save, proj_1, proj_2, 
                                   title,
                                   mode):
    fake_data_proj = fake_data[:, [proj_1, proj_2]]
    X_train_proj = X_train[:, [proj_1, proj_2]]

    plt.figure(figsize=(8, 8))
    plt.xlim(-2., 2.)
    plt.ylim(-2., 2.)
    plt.title(title, fontsize=20)
    X_train_proj = X_train[:, [proj_1, proj_2]]


    plt.scatter(X_train_proj[:, 0], X_train_proj[:, 1], alpha=0.5, color='gray',
                marker='o', label = 'training samples')
    plt.scatter(fake_data_proj[:, 0], fake_data_proj[:, 1], alpha=0.5, color='blue',
                marker='o', label = 'samples by G')
    plt.xlabel(f"proj ind = {proj_1 + 1}")
    plt.ylabel(f"proj ind = {proj_2 + 1}")
    plt.legend()
    plt.grid(True)
    if path_to_save is not None:
       cur_time = datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
       plot_name = cur_time + f"_wgan_sampling_" + mode + f"_proj1_{proj_1}_proj2_{proj_2}.pdf"
       path_to_plot = os.path.join(path_to_save, plot_name)
       plt.savefig(path_to_plot)

    else:
       plt.show()

def epoch_visualization(X_train, generator, 
                        use_gradient_penalty, 
                        discriminator_mean_loss_arr, 
                        epoch, Lambda,
                        generator_mean_loss_arr, 
                        path_to_save,
                        batch_size_sample = 5000,
                        proj_list = None):
    subtitle_for_losses = "Training process for discriminator and generator"
    if (use_gradient_penalty):
        subtitle_for_losses += f" with gradient penalty, $\lambda = {Lambda}$"
    fig, axs = plt.subplots(1, 2, figsize = (20, 5))
    fig.suptitle(subtitle_for_losses)
    axs[0].set_xlabel("#epoch")
    axs[0].set_ylabel("loss")
    axs[1].set_xlabel("#epoch")
    axs[1].set_ylabel("loss")
    axs[0].grid(True)
    axs[1].grid(True)
    axs[0].set_title('D-loss')
    axs[1].set_title('G-loss')
    axs[0].plot(discriminator_mean_loss_arr, 'b', 
                label = f'discriminator loss = Wasserstein')
    axs[1].plot(generator_mean_loss_arr, 'r', label = 'generator loss')
    axs[0].legend()
    axs[1].legend()
    cur_time = datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
    plot_name = cur_time + f'_wgan_losses_{epoch}_epoch.pdf'
    path_to_plot = os.path.join(path_to_save, plot_name)
    fig.savefig(path_to_plot)

    if proj_list is None:
        sample_fake_data(generator, X_train, epoch, path_to_save, batch_size_sample)

    else:
        fake_data = generator.sampling(batch_size_sample).data.cpu().numpy()
        title = "Training and generated samples"
        mode = f"{epoch}_epoch"
        for i in range(len(proj_list)):
            visualize_fake_data_projection(fake_data, X_train, path_to_save, 
                                           proj_list[i][0], proj_list[i][1],
                                           title,
                                           mode)
