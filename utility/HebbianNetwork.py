import torch
import torch.nn as nn
from utility.HebbianLinearLayer import HebbianLinearLayer
from utility.HebbianTraceLinearLayer import HebbianTraceLinearLayer
import queue
from torch.nn import functional as F
import numpy as np

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

        if len(input.shape) == 3 or len(input.shape) == 4:
            input = input.reshape(input.shape[0], -1)

        return input


    def learn(self, input, targets=None):
        """
        Forward pass through the network, learning the weights with Hebbian learning.
        """
        input = self.reshape_input(input)
        input = self.activation(input)

        if self.use_targets:            
            old_input = input
            for i in range(len(self.layers)):
                input = self.dropout(input)
                out = self.layers[i].forward(input)
                if i > 0:
                    # calculate the inverse
                    inv_out = self.layers[i].backward(out)
                    self.layers[i - 1].update_weights(old_input, inv_out)
                    old_input = input
                if i == len(self.layers) - 1:
                    self.layers[i].update_weights(input, out, targets)
                input = out
        else:
            for layer in self.layers:
                input = self.dropout(input)
                input = layer.learn(input)
        return input

    def forward(self, input):
        """
        Forward pass through the network, without learning the weights.
        """
        input = self.reshape_input(input)
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
            tmp[i] = layer.weight.clone().detach().to(self.device)
        return tmp
    
    def set_weights(self, weights):
        """
        Sets the weights of the network.
        """
        for i, layer in enumerate(self.layers):
            layer.weight = weights[i].clone().detach().to(self.device)

    def get_flatten_params(self):
        """
        Returns the parameters of the network as a single tensor.
        """
        return torch.cat([p.view(-1).clone().detach() for p in self.parameters()])

    def set_params_flatten(self, flat_params):
        """
        Sets the parameters of the network from a single tensor.
        """
        idx = 0
        for p in self.parameters():
            p.data = flat_params[idx:idx + p.numel()].view(p.size()).clone().detach().to(self.device)
            idx += p.numel()