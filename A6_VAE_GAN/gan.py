from __future__ import print_function

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F

NOISE_DIM = 96


def hello_gan():
    print("Hello from gan.py!")


def sample_noise(batch_size, noise_dim, dtype=torch.float, device="cpu"):
    """
    Generate a PyTorch Tensor of uniform random noise.

    Input:
    - batch_size: Integer giving the batch size of noise to generate.
    - noise_dim: Integer giving the dimension of noise to generate.

    Output:
    - A PyTorch Tensor of shape (batch_size, noise_dim) containing uniform
      random noise in the range (-1, 1).
    """
    noise = None
    ##############################################################################
    # TODO: Implement sample_noise.                                              #
    ##############################################################################
    # Since "torch.rand" generates random values in [0,1), we transform
    # the value to be between [-1, 1)
    noise = 2 * torch.rand((batch_size, noise_dim), dtype=dtype, device=device) - 1
    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################

    return noise


def discriminator():
    """
    Build and return a PyTorch nn.Sequential model implementing the architecture
    in the notebook.
    """
    model = None
    ############################################################################
    # TODO: Implement discriminator.                                           #
    ############################################################################
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256), 
        nn.LeakyReLU(0.01),
        nn.Linear(256, 256),
        nn.LeakyReLU(0.01),
        nn.Linear(256, 1)   
    )
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    return model


def generator(noise_dim=NOISE_DIM):
    """
    Build and return a PyTorch nn.Sequential model implementing the architecture
    in the notebook.
    """
    model = None
    ############################################################################
    # TODO: Implement generator.                                               #
    ############################################################################
    model = nn.Sequential(
        nn.Linear(noise_dim, 1024), 
        nn.ReLU(),
        nn.Linear(1024, 1024), 
        nn.ReLU(),
        nn.Linear(1024, 784),
        nn.Tanh()
    )
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return model


def discriminator_loss(logits_real, logits_fake):
    """
    Computes the discriminator loss described above.

    Inputs:
    - logits_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Returns:
    - loss: PyTorch Tensor containing (scalar) the loss for the discriminator.
    """
    loss = None
    ##############################################################################
    # TODO: Implement discriminator_loss.                                        #
    ##############################################################################
    '''
    Update the discriminator (𝐷), to maximize the probability of the 
    discriminator making the correct choice on real and generated data.

    Note that we can use binary_cross_entropy_with_logits here, because
    all targets label are either 0 or 1
    '''
    # Given a real image, the target output of discriminator should be 1
    targets_real = torch.ones_like(logits_real) # (N, )
    loss_real = F.binary_cross_entropy_with_logits(logits_real, targets_real)
    
    # Given a fake image, the target output of discriminator should be 0
    targets_fake = torch.zeros_like(logits_fake) # (N, )
    loss_fake = F.binary_cross_entropy_with_logits(logits_fake, targets_fake)

    # BCE has already taken into account the negative sign and the mean, given
    # that default reduction="mean"
    loss = loss_real + loss_fake
    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################
    return loss


def generator_loss(logits_fake):
    """
    Computes the generator loss described above.

    Inputs:
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Returns:
    - loss: PyTorch Tensor containing the (scalar) loss for the generator.
    """
    loss = None
    ##############################################################################
    # TODO: Implement generator_loss.                                            #
    ##############################################################################
    '''
    Update the generator (𝐺) to maximize the probability of the discriminator 
    making the incorrect choice on generated data.
    '''
    # We want the generator to generator to generate good image which fools discriminator
    targets_fake = torch.ones_like(logits_fake) # (N, )
    
    # BCE has already taken into account the negative sign and the mean, given
    # that default reduction="mean"
    loss = F.binary_cross_entropy_with_logits(logits_fake, targets_fake)
    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################
    return loss


def get_optimizer(model):
    """
    Construct and return an Adam optimizer for the model with learning rate 1e-3,
    beta1=0.5, and beta2=0.999.

    Input:
    - model: A PyTorch model that we want to optimize.

    Returns:
    - An Adam optimizer for the model with the desired hyperparameters.
    """
    optimizer = None
    ##############################################################################
    # TODO: Implement optimizer.                                                 #
    ##############################################################################
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.5, 0.999))
    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################
    return optimizer


def ls_discriminator_loss(scores_real, scores_fake):
    """
    Compute the Least-Squares GAN loss for the discriminator.

    Inputs:
    - scores_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """
    loss = None
    ##############################################################################
    # TODO: Implement ls_discriminator_loss.                                     #
    ##############################################################################
    '''
    Update the discriminator (𝐷), to maximize the probability of the 
    discriminator making the correct choice on real and generated data:
    '''
    loss_real = (scores_real - 1) ** 2
    loss_real = 0.5 * loss_real.mean()

    loss_fake = scores_fake ** 2
    loss_fake = 0.5 * loss_fake.mean()

    loss = loss_real + loss_fake
    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################
    return loss


def ls_generator_loss(scores_fake):
    """
    Computes the Least-Squares GAN loss for the generator.

    Inputs:
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """
    loss = None
    ##############################################################################
    # TODO: Implement ls_generator_loss.                                         #
    ##############################################################################
    '''
    Update the generator (𝐺) to maximize the probability of the discriminator 
    making the incorrect choice on generated data.
    '''
    loss_fake = (scores_fake - 1) ** 2
    loss_fake = 0.5 * loss_fake.mean()

    loss = loss_fake
    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################
    return loss


def build_dc_classifier():
    """
    Build and return a PyTorch nn.Sequential model for the DCGAN discriminator
    implementing the architecture in the notebook.
    """
    model = None
    ############################################################################
    # TODO: Implement build_dc_classifier.                                     #
    ############################################################################
    # model input size: (N, 784)
    model = nn.Sequential(
        nn.Unflatten(dim=1, unflattened_size=(1, 28, 28)), # (N, 1, 28, 28)
        nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1), # (N, 32, 24, 24)
        nn.LeakyReLU(0.01),
        nn.MaxPool2d(kernel_size=2, stride=2), # (N, 32, 12, 12)
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1), # (N, 64, 8, 8)
        nn.LeakyReLU(0.01),
        nn.MaxPool2d(kernel_size=2, stride=2), # (N, 64, 4, 4)
        nn.Flatten(), # (N, 64 * 4 * 4)
        nn.Linear(4 * 4 * 64, 4 * 4 * 64),
        nn.LeakyReLU(0.01),
        nn.Linear(4 * 4 * 64, 1)
    )
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return model


def build_dc_generator(noise_dim=NOISE_DIM):
    """
    Build and return a PyTorch nn.Sequential model implementing the DCGAN
    generator using the architecture described in the notebook.
    """
    model = None
    ############################################################################
    # TODO: Implement build_dc_generator.                                      #
    ############################################################################
    # model input size: (N, noise_dim)
    model = nn.Sequential(
        nn.Linear(noise_dim, 1024),
        nn.ReLU(),
        nn.BatchNorm1d(1024),
        nn.Linear(1024, 7 * 7 * 128),
        nn.ReLU(),
        nn.BatchNorm1d(7 * 7 * 128),
        nn.Unflatten(dim=1, unflattened_size=(128, 7, 7)), # (N, 128, 7, 7)
        nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4,
          stride=2, padding=1), # (N, 64, 14, 14), 'same padding'
                                # 14 = 7 + 6*(stride-1) + 2 * (dilation*(kernel_size-1)-padding)
                                #      - (kernel_size-1)
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=4,
          stride=2, padding=1), # (N, 1, 28, 28), use 'same padding'
        nn.Tanh(),
        nn.Flatten() # (N, 784)
    )
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return model
