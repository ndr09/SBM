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
            init="linear",
            neuron_centric=True,
            use_d=False,
            rank=1
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
        super(HebbianLinearLayer, self).__init__()
        
        factory_kwargs = {'device': device, 'dtype': dtype}

        self.last_layer = last_layer
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        self.activation = activation
        self.neuron_centric = neuron_centric
        self.use_d = use_d
        self.init = init
  
        # if bias is True, add one to the input features
        self.bias = bias
        if bias: in_features += 1

        # weight is not a parameter, thus it will not appear in the list of parameters of the model
        self.weight = torch.empty((out_features, in_features), **factory_kwargs, requires_grad=False)     
        self.reset(self.weight, init) # initialize the weights with a normal distribution
        # normalize the weight so the norm is 1 as it will be in the update_weights method
        self.weight = self.normalize(self.weight)   

        if self.neuron_centric:               
            # parameters of the Hebbian learning rule, using nn.Parameter to make them appear in the list of parameters of the model
            self.Ai = nn.Parameter(torch.empty(in_features, requires_grad=True, **factory_kwargs))
            self.Bj = nn.Parameter(torch.empty(out_features, requires_grad=True, **factory_kwargs))
            self.Ci = nn.ParameterList([
                nn.Parameter(self.reset(torch.empty(in_features, requires_grad=True, **factory_kwargs), 'normal'))
                for _ in range(rank)
            ])
            self.Cj = nn.ParameterList([
                nn.Parameter(self.reset(torch.empty(out_features, requires_grad=True, **factory_kwargs), 'normal'))
                for _ in range(rank)
            ])
            self.etai = nn.Parameter(torch.empty(in_features, requires_grad=True, **factory_kwargs))
            self.etaj = nn.Parameter(torch.empty(out_features, requires_grad=True, **factory_kwargs))
            # since B is used on the output, but used only for the previous layer, I can set it directly here
            # self.eta = nn.Parameter(torch.ones(1, requires_grad=True, **factory_kwargs) * 0.1)

            # initialize the parameters with the given distribution
            self.reset(self.Ai, 'normal')
            self.reset(self.Bj, 'normal')
            self.reset(self.etai, 'normal_small')
            self.reset(self.etaj, 'normal_small')

            if self.use_d:
                self.Di = nn.Parameter(torch.empty(in_features, requires_grad=True, **factory_kwargs))
                self.Dj = nn.Parameter(torch.empty(out_features, requires_grad=True, **factory_kwargs))
                self.reset(self.Di, 'normal_small')
                self.reset(self.Dj, 'normal_small')
        else: 
            # parameters of the Hebbian learning rule, using nn.Parameter to make them appear in the list of parameters of the model
            self.Ai = nn.Parameter(torch.empty((in_features, out_features), requires_grad=True, **factory_kwargs))
            self.Bj = nn.Parameter(torch.empty((in_features, out_features), requires_grad=True, **factory_kwargs))
            self.C = nn.Parameter(torch.empty((in_features, out_features), requires_grad=True, **factory_kwargs))
            self.eta = nn.Parameter(torch.empty((in_features, out_features), requires_grad=True, **factory_kwargs))

            # initialize the parameters with the given distribution
            self.reset(self.Ai, 'normal')
            self.reset(self.Bj, 'normal')
            self.reset(self.eta, 'normal_small')
            if self.use_d:
                self.D = nn.Parameter(torch.empty((in_features, out_features), requires_grad=True, **factory_kwargs))
                self.reset(self.D, 'normal')

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

    def learn(self, input, targets=None):
        """
        Performs a forward pass through the layer, learning the weights with Hebbian learning.
        """
        # ensure the shape of the input is correct
        input = self.reshape_input(input)
        
        # calculate the output of the layer
        out = F.linear(input, self.weight)

        # update the weights with the Hebbian learning rule
        if self.neuron_centric:
            if self.last_layer and targets is not None:
                self.update_weights_neuro_centric(input, targets)
            else:
                self.update_weights_neuro_centric(input, self.activation(out))
            # self.update_oja(self.activation(input), self.activation(out))
        else:
            if self.last_layer and targets is not None:
                self.update_weights_synaptic_centric(input, targets)  
            else: 
                self.update_weights_synaptic_centric(input, self.activation(out))

        # apply the activation function only if this is not the last layer
        # for optimal convergence properties given the Cross-Entropy loss
        # out = F.linear(input, self.weight)
        if not self.last_layer:
            out = self.activation(out)
        
        return out

    def forward(self, input):
        """
        Performs a forward pass through the layer, without learning the weights.
        """
        # ensure the shape of the input is correct
        input = self.reshape_input(input)

        # calculate the output of the layer
        out = F.linear(input, self.weight)

        # apply the activation function only if this is not the last layer
        # for optimal convergence properties given the Cross-Entropy loss
        if not self.last_layer:
            out = self.activation(out)

        return out
    
    def attach_hebbian_layer(self, layer):
        """
        Attaches the next Hebbian layer to this layer. In this way, the next layer can be accessed from this layer
        and the parameters C, D and eta of the next layer can be used in the update_weights method.
        """
        self.next_hlayer = layer

    def update_weights_neuro_centric(self, presynaptic, postsynaptic):
        """
        Updates the weights of the layer with the Neuro-centric Hebbian learning rule.
        
        :param presynaptic: The presynaptic activations (input of the layer).
        :param postsynaptic: The postsynaptic activations (output of the layer).
        """

        # b -> batch, i -> in_features, o -> out_features
        A = torch.einsum('bi, i -> bi', presynaptic, self.Ai) # [batch, in_features] -> batched outer product
        B = torch.einsum('bo, o -> bo', postsynaptic, self.Bj) # [batch, out_features] -> batched outer product

        prepost = torch.einsum('bi, bo -> bio', presynaptic, postsynaptic) # [batch, in_features, out_features] -> batched outer product
        CiCj = torch.zeros((self.in_features, self.out_features), device=self.device, dtype=self.dtype)
        for i in range(len(self.Ci)): CiCj = CiCj + torch.einsum('i, o -> io', self.Ci[i], self.Cj[i])
        C = torch.einsum('io, bio -> bio', CiCj, prepost) # [batch, in_features, out_features], batched outer product
        
        eta = (self.etai.unsqueeze(-1) + self.etaj.unsqueeze(0)) / 2 # [in_features, out_features] -> "outer sum"
        abcd = (A.unsqueeze(2) + B.unsqueeze(1) + C) # sum of the marices (with automatic broadcasting)
        
        if self.use_d: 
            DiDj = torch.einsum('i, o -> io', self.Di, self.Dj) # [in_features, out_features] -> outer product
            abcd = abcd + DiDj.unsqueeze(0)

        dw = eta * abcd # [batch, in_features, out_features]
        dw = dw.sum(dim=0) # sum over the batches
            
        # apply the hebbian change in weights
        self.weight = self.weight + dw.T # [out_features, in_features]

        # normalize the weights with the given rule
        self.weight = self.normalize(self.weight)

    def update_weights_synaptic_centric(self, presynaptic, postsynaptic):
        """
        Updates the weights of the layer with the Synaptic-centric Hebbian learning rule.
        
        :param presynaptic: The presynaptic activations (input of the layer).
        :param postsynaptic: The postsynaptic activations (output of the layer).
        """
        # b -> batch, i -> in_features, o -> out_features
        A = torch.einsum('bi, io -> bio', presynaptic, self.Ai)
        B = torch.einsum('bo, io -> bio', postsynaptic, self.Bj)
        C = torch.einsum('io, bi, bo -> bio', self.C, presynaptic, postsynaptic)
        
        abcd = (A + B + C) # sum of the marices (with automatic broadcasting)
        if self.use_d: abcd = abcd + self.D.unsqueeze(0)

        dw = self.eta.unsqueeze(0) * abcd  # [out_features, in_features]
        dw = dw.sum(dim=0) # sum over the batches

        # apply the hebbian change in weights
        self.weight = self.weight + dw.T # [in_features, out_features]

        # normalize the weights with the given rule
        self.weight = self.normalize(self.weight)


    def reset_weights(self, init="maintain"):
        """
        Resets the weights of the layer.
        """
        # need to detach to clean the computational graph of the gradients
        self.weight = self.weight.clone().detach()
        self.reset(self.weight, init)


    def normalize(self, tensor):
        """
        Normalizes the tensor with the L2 norm.
        """
        # return tensor / torch.max(torch.abs(tensor))
        return F.normalize(tensor, p=2, dim=-1)

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