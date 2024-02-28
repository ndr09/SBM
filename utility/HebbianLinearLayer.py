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

        self.bias = None
        self.weight = torch.empty((out_features, in_features), **factory_kwargs, requires_grad=False)                          
        if bias: self.bias = torch.empty(out_features, **factory_kwargs, requires_grad=False)
  
            
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
        A = presynaptic * self.Ai
        B = postsynaptic * self.Bj

        if self.last_layer:
            Cj = postsynaptic * self.C_last
            D = torch.matmul(
                self.D.unsqueeze(0).T, 
                self.D_last.unsqueeze(0)
            ) 
            eta1 = self.eta_last.unsqueeze(0)

        else:
            if self.bias is not None:
                Cj = postsynaptic * self.next_hlayer.C[:-1]
                D = torch.matmul(
                    self.D.unsqueeze(0).T, 
                    self.next_hlayer.D[:-1].unsqueeze(0)
                )
                eta1 = self.next_hlayer.eta[:-1].unsqueeze(0)
            else:
                Cj = postsynaptic * self.next_hlayer.C
                D = torch.matmul(
                    self.D.unsqueeze(0).T, 
                    self.next_hlayer.D.unsqueeze(0)
                )
                eta1 = self.next_hlayer.eta.unsqueeze(0)

        Ci = presynaptic * self.C 
        C = torch.bmm(Ci.unsqueeze(-1), Cj.unsqueeze(1)).permute(0, 2, 1)

        eta0 = self.eta.unsqueeze(0).T
        eta = (eta0 + eta1) / 2 

        # update weights, we do not need to repeat tensors in order to have equal
        dw = eta.T * (A.unsqueeze(1) + B.unsqueeze(2) + C + D.T.unsqueeze(0))
        dw = dw.sum(dim=0) # mean over the batches
        if self.bias is not None:
            bias = dw[:, -1]
            dw = dw[: , :-1]
            self.bias = self.bias + bias
            self.bias = self.normalize(self.bias)
            
        # apply the hebbian change in weights
        self.weight = self.weight + dw
        self.weight = self.normalize(self.weight)

    def reset_weights(self, init="maintain"):
        self.weight = self.weight.clone().detach()
        if self.bias is not None: self.bias = self.bias.clone().detach()

        if init == 'linear':
            torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            if self.bias is not None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                torch.nn.init.uniform_(self.bias, -bound, bound)
        else: 
            # if weight is a parameter
            self.reset(self.weight, init)
            if self.bias is not None: 
                self.reset(self.bias, init)
        
        # with mantain I should keep the weights as they are, thus do not normalize
        if init != "mantain":
            self.weight = self.normalize(self.weight)
            if self.bias is not None: self.bias = self.normalize(self.bias)


    def normalize(self, tensor):
        # return tensor / torch.max(torch.abs(tensor))
        return F.normalize(tensor, p=2, dim=-1)

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
        elif init == 'normal_big':
            torch.nn.init.normal_(parameter, 0, 1)
        elif init == 'ka_uni':
            torch.nn.init.kaiming_uniform_(parameter, a=math.sqrt(5))
        elif init == 'uni_big':
            torch.nn.init.uniform_(parameter, -1, 1)
        elif init == 'xa_uni_big':
            torch.nn.init.xavier_uniform(parameter)
        elif init == 'zero':
            torch.nn.init.zeros_(parameter)

    def get_weights(self):
        return (self.weight, self.bias)
    
    def set_weights(self, weights):
        self.weight, self.bias = weights