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
    ) -> None:
        super(HebbianNetwork, self).__init__()
        self.layers = nn.modules.ModuleList()
        self.device = device

        self.layer_list = layers

        for i in range(len(layers) - 1):
            last_layer = i == len(layers) - 2
            if last_layer:
                self.num_output = layers[i + 1]
            self.layers.append(HebbianLinearLayer(layers[i], layers[i + 1], device=device, last_layer=last_layer))
            if i > 0:
                self.layers[i - 1].attach_hebbian_layer(self.layers[i])

        self.reset_weights(init)
        
        self.dropout = nn.Dropout(dropout)

        self.float()

    def learn(self, input):
        for layer in self.layers:
            input = self.dropout(input)
            input = layer.learn(input)
        return input

    def forward(self, input):
        for layer in self.layers:
            input = self.dropout(input)
            input = layer(input)
        return input
    
    def reset_weights(self, init='uni'):
        for layer in self.layers:
            layer.reset_weights(init)

    def create_dropout_mask(self, size, p):
        """
        Creates a dropout mask for a given size and dropout probability.
        Used for the dropout layer in order to have the same mask in the
        learn and forward pass.
        """
        if p > 0 and self.training:
            mask = torch.empty(*size, device=self.device)
            # with bernoulli we get binary values (0 or 1) for each element of the mask
            mask = mask.bernoulli_(1 - p)
            # scale the mask by the inverse of the dropout probability
            # this ensures that the expected sum of subsequent activation is unchanged
            # leading to a more stable learning process
            return mask / (1 - p)
        return torch.ones(*size, device=self.device)
