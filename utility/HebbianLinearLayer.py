import torch
from torch import nn
import torch.nn.functional as F
import math
from torch.nn import init


class HebbianLinearLayer(nn.Module):

    def __init__(
            self, 
            in_features: int, 
            out_features: int, 
            bias: bool = False, 
            device="cpu", 
            activation=F.tanh,
            dtype=torch.float32,
            last_layer=False,
    ) -> None:
        super(HebbianLinearLayer, self).__init__()
        
        factory_kwargs = {'device': device, 'dtype': dtype}

        self.last_layer = last_layer
        
        self.in_features = in_features
        self.out_features = out_features

        self.weight = torch.empty((out_features, in_features), **factory_kwargs, requires_grad=False)
        if bias:
            self.bias = torch.empty(out_features, **factory_kwargs, requires_grad=False)
        else:
            self.bias = None
            
        if self.bias is not None:
            in_features += 1

        self.Ai = nn.Parameter(torch.empty(in_features, requires_grad=True, **factory_kwargs))
        self.Bj = nn.Parameter(torch.empty(out_features, requires_grad=True, **factory_kwargs))
        self.C = nn.Parameter(torch.empty(in_features, requires_grad=True, **factory_kwargs)) 
        self.D = nn.Parameter(torch.empty(in_features, requires_grad=True, **factory_kwargs))
        self.eta = nn.Parameter(torch.empty(in_features, requires_grad=True, **factory_kwargs))

        self.reset(self.Ai, 'normal')
        self.reset(self.Bj, 'normal')
        self.reset(self.C, 'normal')
        self.reset(self.D, 'normal')
        self.reset(self.eta, 'normal')

        self.device = device
        self.dtype = dtype
        self.activation = activation
        
        # this is the following hebbina layer
        self.hebbian_layer = None

        if self.last_layer:
            self.C_last = nn.Parameter(torch.randn(out_features, requires_grad=True, **factory_kwargs))
            self.D_last = nn.Parameter(torch.randn(out_features, requires_grad=True, **factory_kwargs))
            self.eta_last = nn.Parameter(torch.randn(out_features, requires_grad=True, **factory_kwargs))
            self.reset(self.C_last, 'normal')
            self.reset(self.D_last, 'normal')
            self.reset(self.eta_last, 'normal')

    def reshape_input(self, input):
        if len(input.shape) == 1:
            input = input.unsqueeze(0)

        if len(input.shape) == 3 or len(input.shape) == 4:
            input = input.reshape(input.shape[0], -1)
        return input

    def learn(self, input):
        input = self.reshape_input(input)

        out = F.linear(input, self.weight, self.bias)

        self.update_weights(input, self.activation(out))

        if not self.last_layer:
            out = self.activation(out)
        
        return out

    def forward(self, input):
        input = self.reshape_input(input)
        out = F.linear(input, self.weight, self.bias)

        if not self.last_layer:
            out = self.activation(out)

        return out
    
    def attach_hebbian_layer(self, layer):
        self.next_hlayer = layer


    def update_weights(self, presynaptic, postsynaptic):
        if self.bias is not None:
            presynaptic = F.pad(presynaptic, (0, 1), "constant", 1)

        # update weights
        A = self.Ai * presynaptic
        B = self.Bj * postsynaptic

        # mean over the batch
        A = torch.sum(A, dim=0).unsqueeze(0)
        B = torch.sum(B, dim=0).unsqueeze(0)

        if self.last_layer:
            Cj = self.C_last * postsynaptic
            D = torch.matmul(self.D.unsqueeze(0).T, self.D_last.unsqueeze(0)) # [2, 4]
            eta1 = self.eta_last.unsqueeze(0)

        else:
            if self.bias is not None:
                Cj = self.next_hlayer.C[:-1] * postsynaptic
                D = torch.matmul(self.D.unsqueeze(0).T, self.next_hlayer.D[:-1].unsqueeze(0))
                eta1 = self.next_hlayer.eta[:-1].unsqueeze(0)
            else:
                Cj = self.next_hlayer.C * postsynaptic
                D = torch.matmul(self.D.unsqueeze(0).T, self.next_hlayer.D.unsqueeze(0))
                eta1 = self.next_hlayer.eta.unsqueeze(0)

        Ci = self.C * presynaptic 
        C = torch.matmul(Ci.T, Cj) # [2, 4]

        eta0 = self.eta.unsqueeze(0).T
        eta = (eta0 + eta1) / 2 # [2, 4]

        # update weights, we do not need to repeat tensors in order to have equal
        dw = eta * (A.T + B + C + D) / presynaptic.shape[0]
        if self.bias is not None:
            bias = dw[-1, :]
            dw = dw[:-1, :]
            self.bias = self.bias + bias
            
        self.weight = self.weight + dw.T
        # self.weight = self.weight + dw.T

        # l2 weights normalization
        # self.weight = self.weight / torch.max(torch.abs(self.weight))
        self.weight = F.normalize(self.weight, p=2, dim=1)

    def reset_weights(self, init="maintain"):
        # if weight is a parameter
        self.weight = self.weight.clone().detach()
        self.reset(self.weight, init)

        if self.bias is not None:
            self.bias = self.bias.clone().detach()
            self.reset(self.bias, init)

    

    def reset(self, parameter, init="maintain"):
        """
        Resets the weights of the network.

        :param init: Initialization method for the weights. Default is "maintain".
        """    
        if init == 'xa_uni':
            torch.nn.init.xavier_uniform(parameter, 0.3)
        elif init == 'sparse':
            torch.nn.init.sparse_(parameter, 0.8)
        elif init == 'uni':
            torch.nn.init.uniform_(parameter, -0.1, 0.1)
        elif init == 'normal':
            torch.nn.init.normal_(parameter, 0, 0.1)
        elif init == 'normal_small':
            torch.nn.init.normal_(parameter, 0, 0.01)
        elif init == 'ka_uni':
            torch.nn.init.kaiming_uniform_(parameter, 3)
        elif init == 'uni_big':
            torch.nn.init.uniform_(parameter, -1, 1)
        elif init == 'xa_uni_big':
            torch.nn.init.xavier_uniform(parameter)
        elif init == 'zero':
            torch.nn.init.zeros_(parameter)