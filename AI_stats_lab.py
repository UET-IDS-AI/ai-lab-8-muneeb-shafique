import numpy as np


def sigmoid(z):
    """
    sigmoid(z) = 1 / (1 + exp(-z))
    """
    return 1 / (1 + np.exp(-z))


def forward_pass(X, W1, W2, W3):
    """
    3-layer neural network forward pass.
    """
    h1 = sigmoid(X.dot(W1))
    h2 = sigmoid(h1.dot(W2))
    y = sigmoid(h2.dot(W3))

    return h1, h2, y


def backward_pass(X, h1, h2, y, label, W1, W2, W3):
    """
    Backpropagation for a 3-layer sigmoid neural network.
    """

    # ----- LOSS -----
    if label == 1:
        loss = -np.log(y)
        dJ_dy = -1 / y
    else:
        loss = -np.log(1 - y)
        dJ_dy = 1 / (1 - y)

    # ----- OUTPUT LAYER -----
    dy_dz3 = y * (1 - y)
    grad3 = dJ_dy * dy_dz3

    dW3 = h2.T.dot(grad3)

    # ----- LAYER 2 -----
    dJ_dh2 = grad3.dot(W3.T)
    dh2_dz2 = h2 * (1 - h2)
    grad2 = dJ_dh2 * dh2_dz2

    dW2 = h1.T.dot(grad2)

    # ----- LAYER 1 -----
    dJ_dh1 = grad2.dot(W2.T)
    dh1_dz1 = h1 * (1 - h1)
    grad1 = dJ_dh1 * dh1_dz1

    dW1 = X.T.dot(grad1)

    return dW1, dW2, dW3, loss