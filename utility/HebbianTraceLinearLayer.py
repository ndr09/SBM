import torch
from torch import nn
import torch.nn.functional as F
import math
from torch.nn import init


class HebbianTraceLinearLayer(nn.Module):

    def __init__(
            self, 
            in_features: int, 
            out_features: int, 
            bias: bool = False, 
            device="cpu", 
            activation=F.tanh,
            dtype=torch.float32,
            last_layer=False,
            neuron_centric=False,
            init='linear',
            use_d=None
    ) -> None:
        """
        Initializes the HebbianLinearLayer.

        :param in_features: Number of input features.
        :param out_features: Number of output features.
        :param bias: Whether to use a bias.
        :param device: Device to use for the layer.
        :param activation: Activation function to use.
        :param dtype: Data type to use.
        :param last_layer: Whether this is the last layer of the network.
        :param neuron_centric: Whether to use neuron-centric or synapse-centric Hebbian learning.
        """
        super(HebbianTraceLinearLayer, self).__init__()
        
        factory_kwargs = {'device': device, 'dtype': dtype}

        self.last_layer = last_layer
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        self.activation = activation
        self.neuron_centric = neuron_centric
        self.init = init

        # if bias is True, add one to the input features
        self.bias = bias
        if bias: in_features += 1

        # weight is not a parameter, thus it will not appear in the list of parameters of the model
        self.weight = torch.zeros((out_features, in_features), **factory_kwargs, requires_grad=False) 
        self.reset(self.weight, init)
        self.trace = torch.zeros((in_features, out_features), **factory_kwargs, requires_grad=False)   
        self.yin = None

        if self.neuron_centric:               
            # parameters of the Hebbian learning rule, using nn.Parameter to make them appear in the list of parameters of the model
            self.Ai = nn.Parameter(torch.empty((in_features), requires_grad=True, **factory_kwargs))
            self.Aj = nn.Parameter(torch.empty((out_features), requires_grad=True, **factory_kwargs))
            self.eta = nn.Parameter(torch.ones(1, requires_grad=True, **factory_kwargs) * 0.01)

            # initialize the parameters with the given distribution
            self.reset(self.Ai, 'normal')
            self.reset(self.Aj, 'normal')
            self.reset(self.eta, 'normal_small')

        else: 
            # parameters of the Hebbian learning rule, using nn.Parameter to make them appear in the list of parameters of the model
            self.A = nn.Parameter(torch.empty((in_features, out_features), requires_grad=True, **factory_kwargs))
            self.eta = nn.Parameter(torch.ones(1, requires_grad=True, **factory_kwargs) * 0.01)

            # initialize the parameters with the given distribution
            self.reset(self.A, 'normal')
            self.reset(self.eta, 'normal_small')
            
        # this is the following hebbian layer, it will be set at netwwork level with the attach_hebbian_layer method
        self.next_hlayer = None


    def reshape_input(self, input):
        """
        Reshapes the input to the correct shape. If the input is not in batch form, it is reshaped to be in batch form.
        If the layer has a bias, the input is padded with ones.
        """
        if len(input.shape) == 1:
            input = input.unsqueeze(0)

        if len(input.shape) == 3 or len(input.shape) == 4:
            input = input.reshape(input.shape[0], -1)

        if self.bias:
            # add ones for the bias
            presynaptic = F.pad(presynaptic, (0, 1), "constant", 1)

        return input

    def forward(self, input):
        """
        Performs a forward pass through the layer.
        """
        input = self.reshape_input(input)
        return F.tanh(input @ (self.weight.T + self.trace))

    def learn(self, input, targets):
        """
        Performs a forward pass through the layer, learning the weights with Hebbian learning.
        """
        input = self.reshape_input(input) # [batch, in_features]
        # weight: [out_features, in_features]

        linear_out = input @ self.weight.T # [batch, out_features]
        trc = torch.mul(self.A, self.trace) # [in_features, out_features]
        trace_out = input @ trc # [batch, out_features]

        output = F.tanh(linear_out + trace_out)

        self.trace = (1 - self.eta) * self.trace + self.eta * torch.bmm(input.unsqueeze(2), linear_out.unsqueeze(1)).sum(0) # [in_features, out_features]

        return output
    
    def attach_hebbian_layer(self, layer):
        """
        Attaches the next Hebbian layer to this layer. In this way, the next layer can be accessed from this layer
        and the parameters C, D and eta of the next layer can be used in the update_weights method.
        """
        self.next_hlayer = layer

   

    def reset_weights(self, init="maintain"):
        """
        Resets the weights of the layer.
        """
        # need to detach to clean the computational graph of the gradients
        self.weight = self.weight.clone().detach()
        self.trace = self.trace.clone().detach()
        self.reset(self.weight, init)


    def normalize(self, tensor):
        """
        Normalizes the tensor with the L2 norm.
        """
        # return tensor / torch.max(torch.abs(tensor))
        return F.normalize(tensor, p=2, dim=1)


    def reset(self, parameter, init="maintain"):
        """
        Resets the weights of the tensor.

        :param init: Initialization method for the weights. Default is "maintain".
        :return: The initialized parameter.
        """    
        if init == 'xa_uni':
            return torch.nn.init.xavier_uniform(parameter, 0.3)
        elif init == 'sparse':
            return torch.nn.init.sparse_(parameter, 0.8)
        elif init == 'uni':
            return torch.nn.init.uniform_(parameter, -0.1, 0.1)
        elif init == 'normal':
            return torch.nn.init.normal_(parameter, 0, 0.1)
        elif init == 'normal_small':
            return torch.nn.init.normal_(parameter, 0, 0.01)
        elif init == 'normal_big':
            return torch.nn.init.normal_(parameter, 0, 1)
        elif init == 'ka_uni' or init == 'linear':
            return torch.nn.init.kaiming_uniform_(parameter, a=math.sqrt(5))
        elif init == 'uni_big':
            return torch.nn.init.uniform_(parameter, -1, 1)
        elif init == 'xa_uni_big':
            return torch.nn.init.xavier_uniform(parameter)
        elif init == 'zero':
            return torch.nn.init.zeros_(parameter)
        return parameter