"""
Implements a style transfer in PyTorch.
Make sure to write device-agnostic code. For any function, initialize new tensors
on the same device as input tensors
"""

import torch
import torch.nn as nn


def hello():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Hello from style_transfer.py!")


def content_loss(content_weight, content_current, content_original):
    """
    Compute the content loss for style transfer.

    Inputs:
    - content_weight: Scalar giving the weighting for the content loss.
    - content_current: features of the current image; this is a PyTorch Tensor
      of shape (1, C_l, H_l, W_l).
    - content_original: features of the content image, Tensor with shape
      (1, C_l, H_l, W_l).

    Returns:
    - scalar content loss
    """
    ############################################################################
    # TODO: Compute the content loss for style transfer.                       #
    ############################################################################
    '''
    Content loss indicates how much the feature map of the generated image 
    differs from the feature map of the source image
    '''
    loss = content_weight * torch.sum((content_current - content_original) ** 2)

    return loss
    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################


def gram_matrix(features, normalize=True):
    """
    Compute the Gram matrix from features.

    Inputs:
    - features: PyTorch Tensor of shape (N, C, H, W) giving features for
      a batch of N images.
    - normalize: optional, whether to normalize the Gram matrix
        If True, divide the Gram matrix by the number of neurons (H * W * C)

    Returns:
    - gram: PyTorch Tensor of shape (N, C, C) giving the
      (optionally normalized) Gram matrices for the N input images.
    """
    gram = None
    ############################################################################
    # TODO: Compute the Gram matrix from features.                             #
    # Don't forget to implement for both normalized and non-normalized version #
    ############################################################################
    '''
    Gram matrix G which represents the correlations between the 
    responses of each filter
    '''
    N, C, H, W = features.shape

    f = features.view(N, C, H * W)
    tr_f = torch.transpose(f, 1, 2) # (N, H * W, C)

    gram = f @ tr_f # (N, C, C)

    # Optionally, normalize the Gram matrix.
    if normalize:
        gram /= (H * W * C)
    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################
    return gram


def style_loss(feats, style_layers, style_targets, style_weights):
    """
    Computes the style loss at a set of layers.

    Inputs:
    - feats: list of the features at every layer of the current image, as produced
      by the extract_features function.
    - style_layers: List of layer indices into feats giving the layers to include
      in the style loss.
    - style_targets: List of the same length as style_layers, where
      style_targets[i] is a PyTorch Tensor giving the Gram matrix of the source
      style image computed at layer style_layers[i].
    - style_weights: List of the same length as style_layers, where
      style_weights[i] is a scalar giving the weight for the style loss at layer
      style_layers[i].

    Returns:
    - style_loss: A PyTorch Tensor holding a scalar giving the style loss.
    """
    ############################################################################
    # TODO: Computes the style loss at a set of layers.                        #
    # Hint: you can do this with one for loop over the style layers, and       #
    # should not be very much code (~5 lines).                                 #
    # You will need to use your gram_matrix function.                          #
    ############################################################################
    '''
    We want the activation statistics of our generated image to match the 
    activation statistics of our style image, and matching the (approximate) 
    covariance is one way to do that.
    '''
    loss = 0.0

    for idx, layer_idx in enumerate(style_layers):
      # Compute Gram matrix of features of current layer
      current_gram = gram_matrix(feats[layer_idx])

      # Add current layer style loss.
      loss += style_weights[idx] \
                    * torch.sum((current_gram - style_targets[idx]) ** 2)

    return loss

    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################


def tv_loss(img, tv_weight):
    """
    Compute total variation loss.

    Inputs:
    - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.

    Returns:
    - loss: PyTorch Variable holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    ############################################################################
    # TODO: Compute total variation loss.                                      #
    # Your implementation should be vectorized and not require any loops!      #
    ############################################################################
    '''
    Encourage smoothness in the image
    '''
    verti = torch.sum((img[..., 1:, :] - img[..., :-1, :]) ** 2)
    hori = torch.sum((img[..., 1:] - img[..., :-1]) ** 2)

    loss = tv_weight * (verti + hori)

    return loss
    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################


def guided_gram_matrix(features, masks, normalize=True):
    """
    Inputs:
      - features: PyTorch Tensor of shape (N, R, C, H, W) giving features for
        a batch of N images.
      - masks: PyTorch Tensor of shape (N, R, H, W)
      - normalize: optional, whether to normalize the Gram matrix
          If True, divide the Gram matrix by the number of neurons (H * W * C)

      Returns:
      - gram: PyTorch Tensor of shape (N, R, C, C) giving the
        (optionally normalized) guided Gram matrices for the N input images.
    """
    guided_gram = None
    ##############################################################################
    # TODO: Compute the guided Gram matrix from features.                        #
    # Apply the regional guidance mask to its corresponding feature and          #
    # calculate the Gram Matrix. You are allowed to use one for-loop in          #
    # this problem.                                                              #
    ##############################################################################
    N, R, C, H, W = features.shape

    # Convert mask to be the same dimension as features
    Guidance = masks.unsqueeze(dim=2).repeat(1, 1, C, 1, 1) # (N, R, C, H, W)

    # Get spatially guided feature map
    F = (features * Guidance).view(N, R, C, H*W) # (N, R, C, H*W)
    F_t = torch.transpose(F, dim0=2, dim1=3)

    # Get guided Gram Matrix
    guided_gram = F @ F_t # (N, R, C, C)

    # Optionally, normalize the Gram matrix.
    if normalize:
        guided_gram /= (H * W * C)
    return guided_gram
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################


def guided_style_loss(
    feats, style_layers, style_targets, style_weights, content_masks
):
    """
    Computes the style loss at a set of layers.

    Inputs:
    - feats: list of the features at every layer of the current image, as produced
      by the extract_features function.
    - style_layers: List of layer indices into feats giving the layers to include
      in the style loss.
    - style_targets: List of the same length as style_layers, where
      style_targets[i] is a PyTorch Tensor giving the guided Gram matrix of the
      source style image computed at layer style_layers[i].
    - style_weights: List of the same length as style_layers, where
      style_weights[i] is a scalar giving the weight for the style loss at layer
      style_layers[i].
    - content_masks: List of the same length as "feats", where
      content_masks[i] is a PyTorch Tensor giving the binary masks of the content
      image.

    Returns:
    - style_loss: A PyTorch Tensor holding a scalar giving the style loss.
    """
    ############################################################################
    # TODO: Computes the guided style loss at a set of layers.                 #
    ############################################################################
    '''
    We want the activation statistics of our generated image to match the 
    activation statistics of our style image, and matching the (approximate) 
    covariance is one way to do that.
    '''
    loss = 0.0

    for idx, layer_idx in enumerate(style_layers):
      # Compute Gram matrix of features of current layer
      current_guided_gram = guided_gram_matrix(feats[layer_idx], content_masks[layer_idx])

      # Add current layer style loss.
      loss += style_weights[idx] \
                    * torch.sum((current_guided_gram - style_targets[idx]) ** 2)

    return loss

    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################
