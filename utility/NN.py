from torch import nn
import torch

import torch
import torch.nn as nn

class NN(nn.Module):
    """
    Neural Network class that extends the nn.Module class from PyTorch.
    """

    def __init__(self, nodes: list, grad=False, init=None, device="cpu", wopt=True):
        """
        Initializes the Neural Network.

        Args:
            nodes (list): List of integers representing the number of neurons in each layer.
            grad (bool): Flag indicating whether to compute gradients during training. Default is False.
            init (str): Initialization method for the weights. Default is None.
            device (str): Device to run the network on. Default is "cpu".
            wopt (bool): Flag indicating whether to optimize the network. Default is True.
        """
        super().__init__()

        self.device = torch.device(device)
        # layers of the network
        self.nodes = torch.tensor(nodes).to(self.device)
        # total number of synapses
        self.nweights = sum([self.nodes[i] * self.nodes[i + 1] for i in range(len(self.nodes) - 1)])

        self.network = []
        self.activations = []
        self.grad = grad

        # add linear layers to the network
        for i in range(len(nodes) - 1):
            self.network.append(nn.Linear(nodes[i], nodes[i + 1], bias=False))

        # Initialize weights
        self.reset_weights(init)


    def reset_weights(self, init="maintain"):
        """
        Resets the weights of the network.

        :param init: Initialization method for the weights. Default is "maintain".
        """
        for l in self.network:
            if init == 'xa_uni':
                torch.nn.init.xavier_uniform(l.weight.data, 0.3)
            elif init == 'sparse':
                torch.nn.init.sparse_(l.weight.data, 0.8)
            elif init == 'uni':
                torch.nn.init.uniform_(l.weight.data, -0.1, 0.1)
            elif init == 'normal':
                torch.nn.init.normal_(l.weight.data, 0, 0.024)
            elif init == 'ka_uni':
                torch.nn.init.kaiming_uniform_(l.weight.data, 3)
            elif init == 'uni_big':
                torch.nn.init.uniform_(l.weight.data, -1, 1)
            elif init == 'xa_uni_big':
                torch.nn.init.xavier_uniform(l.weight.data)
            elif init == 'zero':
                torch.nn.init.zeros_(l.weight.data)
            elif type=='maintain':
                l.weight.data = torch.clone(l.weight.data)
        self.activations = []
        self.float()

    def forward(self, inputs):
        """
        Performs forward pass through the network.

        Args:
            inputs: Input data.

        Returns:
            Output of the network.
        """
        self.activations = []
        x = inputs.to(self.device)

        # add input variables to activations
        self.activations.append(torch.clone(x).to(self.device))

        # perform forward pass through the network
        for l in self.network:
            x = l(x)
            x = torch.tanh(x)
            # adding activations of the neurons of current layer
            self.activations.append(torch.clone(x))

        return x

    def get_weights(self):
        """
        Returns the weights of the network.

        Returns:
            List of weight tensors.
        """
        tmp = []
        for l in self.network:
            tmp.append(l.weight)
        return tmp

    def set_weights(self, weights):
        """
        Sets the weights of the network.

        Args:
            weights: List of weight tensors or flattened weight vector.
        """
        # if weights is a list of tensors
        if type(weights) == list and type(weights[0]) == torch.Tensor:
            # set the weights of the network
            for i in range(len(self.network)):
                self.network[i].weight.data = weights[i]
        # if weights is a flattened vector
        elif len(weights) == self.nweights:
            tmp = self.get_weights()
            start = 0
            for i, l in enumerate(tmp):
                size = l.numel() + start
                params = torch.tensor(weights[start:size], requires_grad=self.grad)
                start = size
                self.network[i].weight.data = params.view(l.size())

