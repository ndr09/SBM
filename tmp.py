import numpy as np
import torch
import torch.nn as nn
import time
from scipy.stats import kstest
import gymnasium as gym

from utility.NHNN import NHNN
from utility.HebbianNetworkClassifier import HebbianNetworkClassifier

use_old = True 

# train the model
def tr(model, loss, optimizer, iteration, input):
    """
    Train the model for one iteration
    :param model: The model to be trained
    :param loss: The loss function
    :param optimizer: The optimizer to be used
    :param iteration: The iteration number
    :param input: The input data
    :return: None
    """
    torch.manual_seed(42)


    # Reset the weights of the model
    model.reset_weights('uni')

    # Set size of the input
    s=input.size()[0]

    # Set the model to training mode
    model.train()

    # Shuffle the input data
    idx = torch.randperm(input.nelement())
    inputs = input.view(-1)[idx].view(input.size())

    # Define the target values, in this case we want to predict a linear function
    # that is simply twice the input values
    y = inputs * 2

    # Initialize the predicted values and target values tensors
    yh = torch.zeros((s, 1)) # results of the hebbian

    if use_old:
        for i in range(s):
            a = model.forward(torch.tensor([inputs[i, 0]]))
            # print(a)
            yh[i, 0] = a
    else:
        # Forward pass through the model to obtain predicted values
        for i in range(s):
            a = model.learn(torch.tensor([inputs[i, 0]]))
            a = model.forward(torch.tensor([inputs[i, 0]]))
            yh[i, 0] = a


    # Compute the loss between the predicted values and target values
    loss = loss(yh, y)

    # Print the iteration number and loss
    print(iteration, " ", loss)

    # Print the target values
    # print(y.flatten())

    # Backward pass to compute gradients and update weights
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


if __name__ == "__main__":
    loss_fn = torch.nn.MSELoss()
    np.set_printoptions(precision=5,suppress=True)

    if use_old:
        model = NHNN([1, 8, 1], init='uni')
        params = model.params
    else:
        model = HebbianNetworkClassifier ([1, 8, 1], init='uni')
        params = model.parameters()

    # torch.autograd.set_detect_anomaly(True)
    model = model.to(torch.float)
    model.reset_weights()

    model = model.to(torch.float)


    optimizer = torch.optim.AdamW(params, lr=0.01, weight_decay=0.001)
    
    s = 100
    inputs = torch.rand((s, 1))
    for i in range(100):
        tr(model, loss_fn, optimizer, i, inputs)
