from __future__ import print_function

import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F


def hello_vae():
    print("Hello from vae.py!")


class VAE(nn.Module):
    def __init__(self, input_size, latent_size=15):
        super(VAE, self).__init__()
        self.input_size = input_size  # H*W
        self.latent_size = latent_size  # Z
        self.hidden_dim = None  # H_d
        self.encoder = None
        self.mu_layer = None
        self.logvar_layer = None
        self.decoder = None

        ###########################################################################
        # TODO: Implement the fully-connected encoder architecture described in   #
        # the notebook. Specifically, self.encoder should be a network that       #
        # inputs a batch of input images of shape (N, 1, H, W) into a batch of    #
        # hidden features of shape (N, H_d). Set up self.mu_layer and             #
        # self.logvar_layer to be a pair of linear layers that map the hidden     #
        # features into estimates of the mean and log-variance of the posterior   #
        # over the latent vectors; the mean and log-variance estimates will both  #
        # be tensors of shape (N, Z).                                             #
        ###########################################################################
        self.hidden_dim = 400

        # Define encoder: a three Linear + ReLU neural network.
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.input_size, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU()
        )
                                  
        # Define linear layers that map the hidden features into estimatation
        # Input (N, H_d) and output (N, Z)
        self.mu_layer = nn.Linear(self.hidden_dim, self.latent_size)
        self.logvar_layer = nn.Linear(self.hidden_dim, self.latent_size)

        ###########################################################################
        # TODO: Implement the fully-connected decoder architecture described in   #
        # the notebook. Specifically, self.decoder should be a network that inputs#
        # a batch of latent vectors of shape (N, Z) and outputs a tensor of       #
        # estimated images of shape (N, 1, H, W).                                 #
        ###########################################################################
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_size, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.input_size),
            nn.Sigmoid(),
            nn.Unflatten(dim=1, unflattened_size=(1, 28, 28))
        )

        ###########################################################################
        #                                      END OF YOUR CODE                   #
        ###########################################################################

    def forward(self, x):
        """
        Performs forward pass through FC-VAE model by passing image through
        encoder, reparametrize trick, and decoder models

        Inputs:
        - x: Batch of input images of shape (N, 1, H, W)

        Returns:
        - x_hat: Reconstruced input data of shape (N, 1, H, W)
        - mu: Matrix representing estimated posterior mu (N, Z), with Z latent
          space dimension
        - logvar: Matrix representing estimataed variance in log-space (N, Z),
          with Z latent space dimension
        """
        x_hat = None
        mu = None
        logvar = None
        ###########################################################################
        # TODO: Implement the forward pass by following these steps               #
        # (1) Pass the input batch through the encoder model to get posterior     #
        #     mu and logvariance                                                  #
        # (2) Reparametrize to compute  the latent vector z                       #
        # (3) Pass z through the decoder to resconstruct x                        #
        ###########################################################################
        # Step 1: Pass the input batch through the encoder model to get posterior
        enc_out = self.encoder(x) # (N, hidden_dim)
        mu = self.mu_layer(enc_out) # (N, Z)
        logvar = self.logvar_layer(enc_out) # (N, Z)

        # Step 2: Reparametrize to compute  the latent vector z
        z = reparametrize(mu, logvar) # (N, Z)

        # Step 3: Pass z through the decoder to resconstruct x 
        x_hat = self.decoder(z) # (N, 1, H, W)

        ###########################################################################
        #                                      END OF YOUR CODE                   #
        ###########################################################################
        return x_hat, mu, logvar


class CVAE(nn.Module):
    def __init__(self, input_size, num_classes=10, latent_size=15):
        super(CVAE, self).__init__()
        self.input_size = input_size  # H*W
        self.latent_size = latent_size  # Z
        self.num_classes = num_classes  # C
        self.hidden_dim = None  # H_d
        self.encoder = None
        self.mu_layer = None
        self.logvar_layer = None
        self.decoder = None

        ###########################################################################
        # TODO: Define a FC encoder as described in the notebook that transforms  #
        # the image--after flattening and now adding our one-hot class vector (N, #
        # H*W + C)--into a hidden_dimension (N, H_d) feature space, and a final   #
        # two layers that project that feature space to posterior mu and posterior#
        # log-variance estimates of the latent space (N, Z)                       #
        ###########################################################################
        self.hidden_dim = 400

        # Define encoder: a three Linear + ReLU neural network.
        self.encoder = nn.Sequential(
            nn.Linear(self.input_size + self.num_classes, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU()
        )
                                  
        # Define linear layers that map the hidden features into estimatation
        # Input (N, H_d) and output (N, Z)
        self.mu_layer = nn.Linear(self.hidden_dim, self.latent_size)
        self.logvar_layer = nn.Linear(self.hidden_dim, self.latent_size)

        ###########################################################################
        # TODO: Define a fully-connected decoder as described in the notebook that#
        # transforms the latent space (N, Z + C) to the estimated images of shape #
        # (N, 1, H, W).                                                           #
        ###########################################################################
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_size + self.num_classes, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.input_size),
            nn.Sigmoid(),
            nn.Unflatten(dim=1, unflattened_size=(1, 28, 28))
        )
        ###########################################################################
        #                                      END OF YOUR CODE                   #
        ###########################################################################

    def forward(self, x, c):
        """
        Performs forward pass through FC-CVAE model by passing image through
        encoder, reparametrize trick, and decoder models

        Inputs:
        - x: Input data for this timestep of shape (N, 1, H, W)
        - c: One hot vector representing the input class (0-9) (N, C)

        Returns:
        - x_hat: Reconstructed input data of shape (N, 1, H, W)
        - mu: Matrix representing estimated posterior mu (N, Z), with Z latent
          space dimension
        - logvar: Matrix representing estimated variance in log-space (N, Z),  with
          Z latent space dimension
        """
        x_hat = None
        mu = None
        logvar = None
        ###########################################################################
        # TODO: Implement the forward pass by following these steps               #
        # (1) Pass the concatenation of input batch and one hot vectors through   #
        #     the encoder model to get posterior mu and logvariance               #
        # (2) Reparametrize to compute the latent vector z                        #
        # (3) Pass concatenation of z and one hot vectors through the decoder to  #
        #     resconstruct x                                                      #
        ###########################################################################
        '''
        Add one-hot label vector to encoder input, so that the latent features are
        not only extracted from the image feature, but are also label-specialized;
        Add one-hot label vector to latent space, so that the trained decoder learns
        how to generate target digit based on both latent features and also labels.
        '''
        # Step 1: Pass the input batch through the encoder model to get posterior
        #         Note that we need to flatten images and concatenate one-hot vector
        x_flatten = torch.flatten(x, start_dim=1) # (N, H * W)
        enc_in = torch.cat((x_flatten, c), dim=1) # (N, H * W + C)
        enc_out = self.encoder(enc_in) # (N, hidden_dim)
        mu = self.mu_layer(enc_out) # (N, Z)
        logvar = self.logvar_layer(enc_out) # (N, Z)

        # Step 2: Reparametrize to compute  the latent vector z
        #         And concatenate one-hot vectors
        z = reparametrize(mu, logvar) # (N, Z)
        z = torch.cat((z, c), dim=1) # (N, Z + C)

        # Step 3: Pass z through the decoder to resconstruct x 
        x_hat = self.decoder(z) # (N, 1, H, W)
        ###########################################################################
        #                                      END OF YOUR CODE                   #
        ###########################################################################
        return x_hat, mu, logvar


def reparametrize(mu, logvar):
    """
    Differentiably sample random Gaussian data with specified mean and variance
    using the reparameterization trick.

    Suppose we want to sample a random number z from a Gaussian distribution with
    mean mu and standard deviation sigma, such that we can backpropagate from the
    z back to mu and sigma. We can achieve this by first sampling a random value
    epsilon from a standard Gaussian distribution with zero mean and unit variance,
    then setting z = sigma * epsilon + mu.

    For more stable training when integrating this function into a neural network,
    it helps to pass this function the log of the variance of the distribution from
    which to sample, rather than specifying the standard deviation directly.

    Inputs:
    - mu: Tensor of shape (N, Z) giving means
    - logvar: Tensor of shape (N, Z) giving log-variances

    Returns:
    - z: Estimated latent vectors, where z[i, j] is a random value sampled from a
      Gaussian with mean mu[i, j] and log-variance logvar[i, j].
    """
    z = None
    ###############################################################################
    # TODO: Reparametrize by initializing epsilon as a normal distribution and    #
    # scaling by posterior mu and sigma to estimate z                             #
    ##############################################################################
    # Convert "logvar" to "sigma" (standard deviation).
    sigma = torch.sqrt(torch.exp(logvar))

    # randn_like samples from a standard normal distribution (mu=0, std=1)
    epsilon = torch.randn_like(mu)
    z = sigma * epsilon + mu
    ###############################################################################
    #                              END OF YOUR CODE                               #
    ###############################################################################
    return z


def loss_function(x_hat, x, mu, logvar):
    """
    Computes the negative variational lower bound loss term of the VAE (refer to
    formulation in notebook).

    Inputs:
    - x_hat: Reconstruced input data of shape (N, 1, H, W)
    - x: Input data for this timestep of shape (N, 1, H, W)
    - mu: Matrix representing estimated posterior mu (N, Z), with Z latent space
      dimension
    - logvar: Matrix representing estimated variance in log-space (N, Z), with Z
      latent space dimension

    Returns:
    - loss: Tensor containing the scalar loss for the negative variational
      lowerbound
    """
    loss = None
    ###############################################################################
    # TODO: Compute negative variational lowerbound loss as described in the      #
    # notebook                                                                    #
    ###############################################################################
    ''' 
    Reconstruction loss indicates the different between reconstructed pixel and
    the ground truth pixel; KL Divergence term force the latent space distribution
    "z" to be close to a prior distribution, which in our case is a standard Gaussian
    '''
    # BCE loss have to be adapted to reconstruction loss by:
    # - Changing the reduction mode from 'mean' (default) to 'sum'
    # - The input to the BCE is 'x_hat' and the target is 'x'. This can be done 
    # because each pixel in MNIST dataset is either 0 or 1
    # Note that the minus sign is handled by the BCE loss itself
    recon_loss = F.binary_cross_entropy(x_hat, x, reduction="sum")

    # Get the vectorized KL divergence
    kldiv_term = 1 + logvar - mu**2 - torch.exp(logvar) # (N, Z)
    kl_loss = -0.5 * kldiv_term.sum() # (1)

    # Average the loss across samples in the minibatch
    N = x.shape[0]
    loss = (recon_loss + kl_loss) / N
    ###############################################################################
    #                            END OF YOUR CODE                                 #
    ###############################################################################
    return loss
