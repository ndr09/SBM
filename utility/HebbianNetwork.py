import torch
import torch.nn as nn
from utility.HebbianLinearLayer import HebbianLinearLayer
import queue
from torch.nn import functional as F
import numpy as np

class HebbianNetwork(nn.Module):

    def __init__(
            self, 
            layers: list, 
            init,
            device='cpu',
            dropout=0.0,
            bias=False,
            activation=torch.tanh,
            rank=1
    ) -> None:
        """
        Initializes the HebbianNetwork.
        
        :param layers: List of integers representing the number of neurons in each layer.
        :param init: Initialization method for the weights.
        :param device: Device to use for the network.
        :param dropout: Dropout rate to use.
        :param bias: Whether to use a bias.
        :param activation: Activation function to use.
        :param rank: Rank of the C parameter of Hebbian learning rule. Default is 1.
        """

        super(HebbianNetwork, self).__init__()
        self.layers = nn.modules.ModuleList()
        self.device = device
        self.layer_list = layers

        # Create the layers
        for i in range(len(layers) - 1):
            last_layer = i == len(layers) - 2
            if last_layer:
                self.num_output = layers[i + 1]

            self.layers.append(HebbianLinearLayer(
                layers[i], layers[i + 1], 
                device=device, 
                last_layer=last_layer, 
                bias=bias, 
                activation=activation, 
                dtype=torch.float32,
                rank=rank
            ))
            
            if i > 0:
                # append the current layer to the previous layer's list of attached layers
                self.layers[i - 1].attach_hebbian_layer(self.layers[i])


        self.reset_weights(init)
        self.dropout = nn.Dropout(dropout)

    def learn(self, input):
        """
        Forward pass through the network, learning the weights with Hebbian learning.
        """
        for layer in self.layers:
            input = self.dropout(input)
            input = layer.learn(input)
        return input

    def forward(self, input):
        """
        Forward pass through the network, without learning the weights.
        """
        for layer in self.layers:
            input = layer(input)

        return input
    
    def reset_weights(self, init='uni'):
        """
        Resets the weights of the network.
        If 'mantain' is passed, only the grad graph of the weights is resetted.
        """
        for layer in self.layers:
            layer.reset_weights(init)
    
    def get_weights(self):
        """
        Returns the weights of the network.
        """
        tmp = {}
        for i, layer in enumerate(self.layers):
            tmp[i] = layer.weight
        return tmp
    
    def set_weights(self, weights):
        """
        Sets the weights of the network.
        """
        for i, layer in enumerate(self.layers):
            layer.weight = weights[i].clone().detach().to(self.device)