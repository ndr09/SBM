import torch
import torch.nn as nn
from utility.HebbianLinearLayer import HebbianLinearLayer
import queue
from torch.nn import functional as F
class HebbianNetwork(nn.Module):

    def __init__(
            self, 
            layers: list, 
            init,
            device='cpu',
            dropout=0.0,
            bias=False,
            activation=torch.tanh,
    ) -> None:
        super(HebbianNetwork, self).__init__()
        self.layers = nn.modules.ModuleList()
        self.device = device

        self.layer_list = layers

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
            ))

            nn.Linear
            
            if i > 0:
                self.layers[i - 1].attach_hebbian_layer(self.layers[i])

        self.reset_weights(init)
        
        self.dropout = nn.Dropout(dropout)

        self.float()

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
            tmp[i] = layer.get_weights()
        return tmp
    
    def set_weights(self, weights):
        """
        Sets the weights of the network.
        """
        for i, layer in enumerate(self.layers):
            layer.set_weights(weights[i])
