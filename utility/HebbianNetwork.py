import torch
import torch.nn as nn
from utility.HebbianLinearLayer import HebbianLinearLayer
from utility.HebbianTraceLinearLayer import HebbianTraceLinearLayer
import queue
from torch.nn import functional as F
import numpy as np
import torchvision.transforms.functional as TF
import torchvision

class HebbianNetwork(nn.Module):

    def __init__(
            self, 
            layers: list, 
            device='cpu',
            dropout=0.0,
            bias=False,
            activation=torch.tanh,
            neuron_centric=True,
            init='linear',
            use_d=False,
            train_weights=False,
            use_targets=False
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
        self.activation = activation
        self.use_targets = use_targets

        # Create the layers
        for i in range(len(layers) - 1):
            last_layer = i == len(layers) - 2
            if last_layer:
                self.num_output = layers[i + 1]

            self.layers.append(HebbianLinearLayer(
                layers[i], layers[i + 1], 
                device=device, 
                last_layer=last_layer, 
                first_layer=(i == 0),
                bias=bias, 
                activation=activation, 
                dtype=torch.float32,
                neuron_centric=neuron_centric,
                init=init,
                use_d=use_d,
                train_weights=train_weights
            ))

        self.reset_weights(init)
        self.dropout = nn.Dropout(dropout)

    def reshape_input(self, input):
        """
        Reshapes the input to the correct shape. If the input is not in batch form, it is reshaped to be in batch form.
        If the layer has a bias, the input is padded with ones.
        """
        if len(input.shape) == 1:
            input = input.unsqueeze(0)

        gaussian_filter = torchvision.transforms.GaussianBlur(5, sigma=(1.0, 1.0))
        input = gaussian_filter(input)

        if len(input.shape) == 3 or len(input.shape) == 4:
            input = input.reshape(input.shape[0], -1)

        input = (input - input.mean(dim=-1, keepdim=True)) / input.std(dim=-1, keepdim=True)
        return input


    def learn(self, input, targets=None):
        """
        Forward pass through the network, learning the weights with Hebbian learning.
        """
        input = self.reshape_input(input)
        hebb_losses = []
    
        for layer in self.layers:
            input = self.dropout(input)
            if layer.last_layer and targets is not None and self.use_targets:
                out, loss = layer.learn(input, targets)
            else:
                out, loss = layer.learn(input)
            input = out
            hebb_losses.append(loss)
        return input, sum(hebb_losses)

    def forward(self, input):
        """
        Forward pass through the network, without learning the weights.
        """
        input = self.reshape_input(input)
        hebb_losses = []

        for layer in self.layers:
            out, loss = layer(input)
            hebb_losses.append(loss)
            input = out

        return input, sum(hebb_losses)
        
    def reset_weights(self, init='uni'):
        """
        Resets the weights of the network.
        If 'mantain' is passed, only the grad graph of the weights is resetted.
        """
        for layer in self.layers:
            layer.reset_weights(init)
