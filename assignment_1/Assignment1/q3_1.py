import numpy as np


def compute_gradient_simple(X, y, w, b):
    """
    Compute the gradients of the loss function with respect to w and b for simple linear regression.

    Args:
        X (np.ndarray): Input features matrix of shape (n, m).
        y (np.ndarray): Target vector of shape (n, ).
        w (np.ndarray): Weights vector of shape (m, ).
        b (float): Bias term.

    Returns:
        grad_w (np.ndarray): Gradient with respect to weights.
        grad_b (float): Gradient with respect to bias.
    """
    # Write your code here ...

    return grad_w, grad_b


def compute_gradient_ridge(X, y, w, b, lambda_reg):
    """
    Compute the gradients of the loss function with respect to w and b for ridge regression.

    Args:
        X (np.ndarray): Input features matrix of shape (n, m).
        y (np.ndarray): Target vector of shape (n, ).
        w (np.ndarray): Weights vector of shape (m, ).
        b (float): Bias term.
        lambda_reg (float): Regularization parameter.

    Returns:
        grad_w (np.ndarray): Gradient with respect to weights.
        grad_b (float): Gradient with respect to bias.
    """
    # Write your code here ...

    return grad_w, grad_b
