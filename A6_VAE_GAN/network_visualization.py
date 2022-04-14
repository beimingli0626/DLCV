"""
Implements a network visualization in PyTorch.
Make sure to write device-agnostic code. For any function, initialize new tensors
on the same device as input tensors
"""

import torch


def hello():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Hello from network_visualization.py!")


def compute_saliency_maps(X, y, model):
    """
    Compute a class saliency map using the model for images X and labels y.

    Input:
    - X: Input images; Tensor of shape (N, 3, H, W)
    - y: Labels for X; LongTensor of shape (N,)
    - model: A pretrained CNN that will be used to compute the saliency map.

    Returns:
    - saliency: A Tensor of shape (N, H, W) giving the saliency maps for the input
    images.
    """
    # Make input tensor require gradient
    X.requires_grad_()

    saliency = None
    ##############################################################################
    # TODO: Implement this function. Perform a forward and backward pass through #
    # the model to compute the gradient of the correct class score with respect  #
    # to each input image. You first want to compute the loss over the correct   #
    # scores (we'll combine losses across a batch by summing), and then compute  #
    # the gradients with a backward pass.                                        #
    # Hint: X.grad.data stores the gradients                                     #
    ##############################################################################
    # Make a forward pass of X through the model.
    scores = model(X) # (N, C)

    # Get the scores for correct class for each image.
    # torch.gather gathers values along an axis specified by dim.
    scores = torch.gather(scores, dim=1, index=y.unsqueeze(-1)).squeeze() # (N)

    # Compute the loss over the correct scores, combine across a batch by summing
    loss = torch.sum(scores) # (1)

    # Apply the backward pass, which computes the gradient of the loss
    loss.backward()

    saliency = X.grad.data.abs() # (N, 3, H, W)
    saliency = saliency.max(dim=1).values # (N, H, W)
    ##############################################################################
    #               END OF YOUR CODE                                             #
    ##############################################################################
    return saliency


def make_adversarial_attack(X, target_y, model, max_iter=100, verbose=True):
    """
    Generate an adversarial attack that is close to X, but that the model classifies
    as target_y.

    Inputs:
    - X: Input image; Tensor of shape (1, 3, 224, 224)
    - target_y: An integer in the range [0, 1000)
    - model: A pretrained CNN
    - max_iter: Upper bound on number of iteration to perform
    - verbose: If True, it prints the pogress (you can use this flag for debugging)

    Returns:
    - X_adv: An image that is close to X, but that is classifed as target_y
    by the model.
    """
    # Initialize our adversarial attack to the input image, and make it require
    # gradient
    X_adv = X.clone()
    X_adv = X_adv.requires_grad_()

    learning_rate = 1
    ##############################################################################
    # TODO: Generate an adversarial attack X_adv that the model will classify    #
    # as the class target_y. You should perform gradient ascent on the score     #
    # of the target class, stopping when the model is fooled.                    #
    # When computing an update step, first normalize the gradient:               #
    #   dX = learning_rate * g / ||g||_2                                         #
    #                                                                            #
    # You should write a training loop.                                          #
    #                                                                            #
    # HINT: For most examples, you should be able to generate an adversarial     #
    # attack in fewer than 100 iterations of gradient ascent.                    #
    # You can print your progress over iterations to check your algorithm.       #
    #       You have to perform inplace operations on img.data to update         #
    # the generated image using gradient ascent & reset img.grad to zero         #
    # after each step.                                                           #
    ##############################################################################
    for epoch in range(max_iter):
        score = model(X_adv) # (1, C)
        max_score, label = torch.max(score, axis=1) # get classified label of the image
        target_score = score[0, target_y].squeeze()
        print('Iteration %2d: target score %.3f, max score %.3f' \
                % (epoch + 1, target_score.item(), max_score.item()))
        
        if label.item() == target_y: # if the image classified to target label, stop attack
            break

        target_score.backward()
        
        # Normalize the gradient (Note that "L2 norm" was used in the division).
        X_adv.grad /= torch.linalg.norm(X_adv.grad)
        # X_adv.grad /= (torch.sum(X_adv.grad ** 2) ** 0.5) # implementation of L2 norm

        # Compute an update step: Apply the gradient ascent
        X_adv.data += learning_rate * X_adv.grad.data

        # Re-initialize the gradient of "X_adv" to zero
        X_adv.grad.data.zero_()
        
    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################
    return X_adv


def class_visualization_step(img, target_y, model, **kwargs):
    """
    Performs gradient step update to generate an image that maximizes the
    score of target_y under a pretrained model.

    Inputs:
    - img: random image with jittering as a PyTorch tensor
    - target_y: Integer in the range [0, 1000) giving the index of the class
    - model: A pretrained CNN that will be used to generate the image

    Keyword arguments:
    - l2_reg: Strength of L2 regularization on the image
    - learning_rate: How big of a step to take
    """

    l2_reg = kwargs.pop("l2_reg", 1e-3)
    learning_rate = kwargs.pop("learning_rate", 25)
    ########################################################################
    # TODO: Use the model to compute the gradient of the score for the     #
    # class target_y with respect to the pixels of the image, and make a   #
    # gradient step on the image using the learning rate. Don't forget the #
    # L2 regularization term!                                              #
    # Be very careful about the signs of elements in your code.            #
    # Hint: You have to perform inplace operations on img.data to update   #
    # the generated image using gradient ascent & reset img.grad to zero   #
    # after each step.                                                     #
    ########################################################################
    score = model(img) # (1, C)
    target_score = score[0, target_y].squeeze() # Sy(I)
    target_score -= l2_reg * torch.square(torch.linalg.norm(img)) # Sy(I)-R(I)
    # target_score -= l2_reg * torch.sum(img.grad ** 2)
    
    target_score.backward()

    # Compute an update step: Apply the gradient ascent
    img.data += learning_rate * img.grad.data

    # Re-initialize the gradient of "X_adv" to zero
    img.grad.data.zero_()
    ########################################################################
    #                             END OF YOUR CODE                         #
    ########################################################################
    return img
