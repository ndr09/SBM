import torch
from torch import nn
import torch.nn.functional as F
import math
from torch.nn import init
import torchvision
import numpy as np


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
            first_layer=False,
            init="linear",
            neuron_centric=True,
            use_d=False,
            rank=1,
            train_weights=True,
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
        self.first_layer = first_layer
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        self.activation = activation
        self.neuron_centric = neuron_centric
        self.use_d = use_d
        self.init = init
        self.train_weights = train_weights
        self.rank = rank

        # if bias is True, add one to the input features
        self.bias = bias
        if bias: in_features += 1

        self.init_weights(factory_kwargs) 

        if self.neuron_centric:               
            self.init_neurocentric_params(factory_kwargs=factory_kwargs)
        else: 
            self.init_synapticcentric_params(factory_kwargs=factory_kwargs)


    def init_weights(self, factory_kwargs):
        """
        Initialize weights
        """
        if self.train_weights:
            # weight is a parameter, thus it will appear in the list of parameters of the model
            self.weight = nn.Parameter(torch.empty(
                (self.out_features, self.in_features),
                **factory_kwargs,
                requires_grad=True
            ))
            self.reset(self.weight, init)
            # self.weight.data = self.normalize(self.weight)
        else:
            # weight is not a parameter, thus it will not appear in the list of parameters of the model
            self.weight = torch.empty(
                (self.out_features, self.in_features), 
                **factory_kwargs, 
                requires_grad=False,
            )     
            self.reset(self.weight, init) # initialize the weights with a normal distribution
            # normalize the weight so the norm is 1 as it will be in the update_weights method
            # self.weight = self.normalize(self.weight)

    def init_neurocentric_params(self, factory_kwargs):
            # parameters of the Hebbian learning rule, using nn.Parameter to make them appear in the list of parameters of the model
            self.Ai = nn.Parameter(torch.empty(self.in_features, requires_grad=True, **factory_kwargs))
            self.Bj = nn.Parameter(torch.empty(self.out_features, requires_grad=True, **factory_kwargs))
            self.Ci = nn.ParameterList([
                nn.Parameter(self.reset(torch.empty(self.in_features, requires_grad=True, **factory_kwargs), 'normal_big'))
                for _ in range(self.rank)
            ])
            self.Cj = nn.ParameterList([
                nn.Parameter(self.reset(torch.empty(self.out_features, requires_grad=True, **factory_kwargs), 'normal_big'))
                for _ in range(self.rank)
            ])
            self.Ei = nn.Parameter(torch.empty(self.in_features, requires_grad=True, **factory_kwargs))
            self.Ej = nn.Parameter(torch.empty(self.out_features, requires_grad=True, **factory_kwargs))
            # since B is used on the output, but used only for the previous layer, I can set it directly here
            # self.eta = nn.Parameter(torch.ones(1, requires_grad=True, **factory_kwargs) * 0.01)

            # initialize the parameters with the given distribution
            self.reset(self.Ai, 'normal_small')
            self.reset(self.Bj, 'normal_small')
            self.reset(self.Ei, 'normal')
            self.reset(self.Ej, 'normal')

            if self.use_d:
                self.Di = nn.Parameter(torch.empty(self.in_features, requires_grad=True, **factory_kwargs))
                self.Dj = nn.Parameter(torch.empty(self.out_features, requires_grad=True, **factory_kwargs))
                self.reset(self.Di, 'normal_small')
                self.reset(self.Dj, 'normal_small')


    def init_synapticcentric_params(self, factory_kwargs):
        # parameters of the Hebbian learning rule, using nn.Parameter to make them appear in the list of parameters of the model
        self.Ai = nn.Parameter(torch.empty((self.in_features, self.out_features), requires_grad=True, **factory_kwargs))
        self.Bj = nn.Parameter(torch.empty((self.in_features, self.out_features), requires_grad=True, **factory_kwargs))
        self.C = nn.Parameter(torch.empty((self.in_features, self.out_features), requires_grad=True, **factory_kwargs))
        self.E = nn.Parameter(torch.empty((self.in_features, self.out_features), requires_grad=True, **factory_kwargs))   
        # self.eta = nn.Parameter(torch.ones(1, requires_grad=True, **factory_kwargs) * 0.01)

        # initialize the parameters with the given distribution
        self.reset(self.Ai, 'normal')
        self.reset(self.Bj, 'normal')
        self.reset(self.E, 'normal')
        if self.use_d:
            self.D = nn.Parameter(torch.empty((self.in_features, self.out_features), requires_grad=True, **factory_kwargs))
            self.reset(self.D, 'normal')


    def learn(self, input, targets=None):
        """
        Performs a forward pass through the layer, learning the weights with Hebbian learning.
        """
        # if there are zeros in self.initialized_weights, it means the weights have not been initialized
        # self.init_if_needed(input)

        # calculate the output of the layer
        out = F.linear(input, self.weight)
        
        if targets is not None and self.last_layer:
            # update the weights
            self.update_weights_hebbian(input, targets)
        else:
            self.update_weights_hebbian(input, self.activation(out))

        out = F.linear(input, self.weight)

        # apply the activation function only if this is not the last layer
        # for optimal convergence properties given the Cross-Entropy loss
        if not self.last_layer:
            out = self.activation(out)
        
        return out, self.loss(input, out)
    
    def forward(self, input):
        """
        Performs a forward pass through the layer, without learning the weights.
        """
        # calculate the output of the layer
        out = F.linear(input, self.weight)

        # apply the activation function only if this is not the last layer
        # for optimal convergence properties given the Cross-Entropy loss
        if not self.last_layer:
            out = self.activation(out)

        return out, self.loss(input, out)
                
    def calculate_A(self, presynaptic):
        if self.neuron_centric:
            # [batch, in_features] -> batched outer product
            return torch.einsum('bi, i -> bi', presynaptic, self.Ai).unsqueeze(2)
        else: 
            return torch.einsum('bi, io -> bio', presynaptic, self.Ai)
        
    def calculate_B(self, postsynaptic):
        if self.neuron_centric:
            return torch.einsum('bo, o -> bo', postsynaptic, self.Bj).unsqueeze(1)
        else:
            return torch.einsum('bo, io -> bio', postsynaptic, self.Bj)

    def calculate_C(self, presynaptic, postsynaptic):
        if self.neuron_centric:
            prepost = torch.einsum('bi, bo -> bio', presynaptic, postsynaptic) # [batch, in_features, out_features] -> batched outer product
            CiCj = torch.zeros((self.in_features, self.out_features), device=self.device, dtype=self.dtype)
            for i in range(len(self.Ci)): CiCj = CiCj + torch.einsum('i, o -> io', self.Ci[i], self.Cj[i])
            C = torch.einsum('io, bio -> bio', CiCj, prepost) # [batch, in_features, out_features], batched outer product
            return C
        else:
            return torch.einsum('io, bi, bo -> bio', self.C, presynaptic, postsynaptic)
        
    def calculate_D(self):
        if self.neuron_centric:
            return torch.einsum('i, o -> io', self.Di, self.Dj).unsqueeze(0)
        else:
            return self.D.unsqueeze(0)
        
    def calculate_eta(self):
        if self.neuron_centric:
            return (self.Ei.unsqueeze(-1) + self.Ej.unsqueeze(0)).unsqueeze(0) / 2
        else:
            return self.E.unsqueeze(0)

    def apply_dw(self, dw):
        # apply the hebbian change in weights
        if self.train_weights:
            self.weight.data = self.weight.data + dw / dw.norm() # [in_features, out_features]
            self.weight.data = self.normalize(self.weight)
        else:
            self.weight = self.weight + dw / dw.norm()# [in_features, out_features]
            self.weight = self.normalize(self.weight)

    def update_weights_hebbian(self, presynaptic, postsynaptic):
        """
        Updates the weights of the layer with the Neuro-centric Hebbian learning rule.
        
        :param presynaptic: The presynaptic activations (input of the layer). [batch, in_features]
        :param postsynaptic: The postsynaptic activations (output of the layer). [batch, out_features]
        """
        # topk = max(self.out_features // 15, 1)
        # prob = torch.functional.F.softmax(postsynaptic, dim=1)
        # winner_neuron = torch.multinomial(prob, topk)
        # mask = torch.zeros_like(postsynaptic)
        # mask = mask.scatter(1, winner_neuron, 1)
        # postsynaptic = mask * postsynaptic


        A = self.calculate_A(presynaptic)
        B = self.calculate_B(postsynaptic)
        C = self.calculate_C(presynaptic, postsynaptic)
        
        eta = self.calculate_eta()

        abcd = A + B + C
        if self.use_d: abcd = abcd + self.calculate_D()

        dw = (eta*abcd).sum(dim=0).T
        self.apply_dw(dw)

    def normalize(self, tensor):
        """
        Normalizes the tensor with the L2 norm.
        """
        # return tensor / tensor.std(dim=-1, keepdim=True)
        # return tensor / torch.max(torch.abs(tensor))
        return F.normalize(tensor, p=2, dim=-1)
        # return tensor / ( torch.max(torch.abs(tensor), dim=-1, keepdim=True).values + 1e-8)
        # max_value = torch.max(torch.abs(tensor))
        # l2_penalty = (tensor) ** 2 * torch.sign(tensor)
        # l2_penalty = (l2_penalty / torch.max(torch.abs(l2_penalty))) * max_value
        # return tensor - l2_penalty * 0.1

    def reset_weights(self, init="maintain"):
        """
        Resets the weights of the layer.
        """
        # need to detach to clean the computational graph of the gradients
        if self.train_weights:
            self.weight.data = self.weight.data.clone().detach()
            self.weight.data = self.reset(self.weight, init)
        else:
            self.weight = self.weight.clone().detach()
            self.weight = self.reset(self.weight, init)

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
        elif init == 'ones':
            return torch.nn.init.ones_(parameter) * .01
        else:
            return parameter
        

    def loss(self, input, output):
        """
        Calculates the loss of the layer.

        :param input: The input of the layer.
        :param targets: The targets of the layer.
        :return: The loss of the layer.
        """
        mse_loss = 0
        if self.first_layer:
            # reconstruction loss
            mse = nn.MSELoss()
            topk = 5
            _, topk_idx = torch.topk(output, topk, dim=-1)
            neurons = self.weight[topk_idx]
            mse_loss = mse(neurons, input.unsqueeze(1).repeat(1, topk, 1))

        return mse_loss 

            

            
        