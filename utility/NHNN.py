from utility.NN import NN
from torch import nn
import torch

class HP(nn.Module):
    def __init__(self):
        super().__init__()
        self.l = nn.Parameter(torch.rand(1))

    def forward(self, inputs):
        return self.l * inputs


class NP(nn.Module):
    def __init__(self):
        super().__init__()
        self.l = torch.zeros(1)

    def forward(self, inputs):
        return self.l * inputs


class NHNN(NN):
    """
    Neural Network class that implements the Neural Hebbian Learning Rule.
    """

    def __init__(self, nodes: list, device="cpu", init=None):
        """
        Initializes the Neural Network.

        Args:
            nodes (list): List of integers representing the number of neurons in each layer.
            device (str): Device to run the network on. Default is "cpu".
            init (str): Initialization method for the weights. Default is None.
        """

        super(NHNN, self).__init__(nodes, grad=False, device=device, init=init, wopt=False)

        self.hrules = []
        self.nparams = sum(self.nodes) * 5 - self.nodes[0] - self.nodes[-1]

        self.a = [] # presynaptic weights
        self.b = [] # postsynaptic weights
        self.c = [] # hebbian weights
        self.d = [] # bias weights
        self.e = [] # eta (learning rate)

        self.set_hrules()

        self.dws=[] # delta weights
        for i in range(len(self.network)):
            ih = self.nodes[i]
            oh = self.nodes[i + 1]
            self.dws.append(torch.zeros((oh, ih), requires_grad=False))


        self.params = []
        for l in range(len(self.nodes)):
            #print(l,self.nodes,self.a[l])
            for i in range(self.nodes[l]):
                    self.params.extend(list(self.a[l][i].parameters()))
                    self.params.extend(list(self.b[l][i].parameters()))
                    self.params.extend(list(self.c[l][i].parameters()))
                    self.params.extend(list(self.d[l][i].parameters()))
                    self.params.extend(list(self.e[l][i].parameters()))

        self.float()

    def train(self, mode: bool = True):
        """
        Sets the module in training mode.
        """
        for l in range(len(self.nodes)):
            for i in range(self.nodes[l]):
                self.a[l][i].train()
                self.b[l][i].train()
                self.c[l][i].train()
                self.d[l][i].train()
                self.e[l][i].train()


    def forward(self, inputs):

        tmp = []
        x0 = torch.tanh(inputs)
        
        tmp.append(torch.reshape(torch.clone(x0),(x0.size()[0],1)))

        c = 0

        for l in self.network:
            ih = self.nodes[c]
            oh = self.nodes[c + 1]
            dw = torch.zeros((oh, ih))

            if len(self.activations) > c + 1:
                for i in range(ih):
                    for o in range(oh):
                        # presynaptic value
                        ai = self.a[c][i].forward(self.activations[c][i])

                        # postsynaptic value
                        bj = self.b[c + 1][o].forward(self.activations[c + 1][o])

                        # presynaptic * postsynaptic
                        ci = self.c[c][i].forward(self.activations[c][i])
                        cj = self.c[c + 1][o].forward(self.activations[c + 1][o])
                        cij = ci * cj
                        
                        # bias
                        di = self.d[c][i].forward(torch.ones(1, dtype=torch.float))
                        dj = self.d[c + 1][o].forward(torch.ones(1, dtype=torch.float))
                        dij = di * dj

                        # calculate the change in weights
                        ei = self.e[c][i].forward(torch.ones(1, dtype=torch.float))
                        ej = self.e[c + 1][o].forward(torch.ones(1, dtype=torch.float))
                        eta  =  0.5 * (ei + ej)

                        dw[o, i] = eta * (ai + bj + cij + dij)

                self.dws[c] = dw

            # calculate the change output
            dw_out = torch.matmul(dw, x0)

            # add the change in weights output to the weights output
            x1 = l(x0) + dw_out

            if not c == len(self.network)-1:
                x1 = torch.tanh(x1)

            tmp.append(torch.reshape(torch.clone(x1), (x1.size()[0], 1)))

            # set the output as the input for the next layer
            x0 = x1
            c += 1

        # sets the weights of the network
        for i in range(len(self.network)):
            self.set_weights_layer(self.dws[i],i)
        self.activations = tmp[:]

        return x1

    def forward_nu(self,inputs):
        """
        Performs forward pass through the network.
        """
        x = torch.tanh(inputs)
        for l in self.network[:-1]:
            x = torch.tanh(l(x))
        return self.network[-1](x)

    def set_hrules(self):
        """
        Sets the Hebbian Learning Rule for the network.
        """

        start = 0
        self.a.append([HP() for _ in range(self.nodes[0])])
        start += self.nodes[0]

        self.b.append([NP() for _ in range(self.nodes[0])])
        start += self.nodes[0]

        self.c.append([HP() for _ in range(self.nodes[0])])
        start += self.nodes[0]

        self.d.append([HP() for _ in range(self.nodes[0])])
        start += self.nodes[0]

        self.e.append([HP() for _ in range(self.nodes[0])])
        start += self.nodes[0]

        for i, l in enumerate(self.nodes[1:-1]):
            self.a.append([HP() for _ in range(l)])
            start += self.nodes[i]

            self.b.append([HP() for _ in range(l)])
            start += self.nodes[i]

            self.c.append([HP() for _ in range(l)])
            start += self.nodes[i]

            self.d.append([HP() for _ in range(l)])
            start += self.nodes[i]

            self.e.append([HP() for _ in range(l)])
            start += self.nodes[i]

        self.a.append([NP() for _ in range(self.nodes[-1])])

        self.b.append([HP() for _ in range(self.nodes[-1])])
        start += self.nodes[-1]

        self.c.append([HP() for _ in range(self.nodes[-1])])
        start += self.nodes[-1]

        self.d.append([HP() for _ in range(self.nodes[-1])])
        start += self.nodes[-1]

        self.e.append([HP() for _ in range(self.nodes[-1])])
        start += self.nodes[-1]


    def set_weights_layer(self, weights, i):
        """
        Sets the weights of the network and normalizes them.

        Args:
            weights: List of weight tensors or flattened weight vector.
            i: layer index
        """

        tmp = weights.clone().detach()
        tmp += self.network[i].weight.data
        # get the maximum value of the weights
        reg = torch.max(torch.abs(tmp))

        # normalize the weights
        tmp /= reg if not reg==0. else 1.

        self.network[i].weight.data = tmp

    def update_weights_layer(self, i, activations_i, activations_j):
        """
        Updates the weights of the network using the Hebbian Learning Rule.

        The Hebbian Learning Rule is given by dw = pre + post + C + D
        where:
            - pre = a_i * hrule_i
            - post = b_j * hrule_j
            - C = a_i * hrule_i * b_j * hrule_j
            - D = hrule_i * hrule_j (bias)

        Args:
            i: Layer index
            activations_i: Activations of the neurons of the ith layer
            activations_j: Activations of the neurons of the (i+1)th layer
        """
        l = self.get_weights()[i]

        # get the activations of the neurons of the ith layer (presynaptic)
        pre_i = torch.reshape(self.a[i] * activations_i, (1, activations_i.size()[0]))
        pre_i = pre_i.repeat((activations_j.size()[0], 1))

        # get the activations of the neurons of the (i+1)th layer (postsynaptic)
        post_j = self.b[i + 1] * activations_j
        post_j = torch.reshape(post_j, (activations_j.size()[0], 1))
        post_j = post_j.repeat((1, activations_i.size()[0]))

        # get the Hebbian Learning Rule for the ith layer
        c_i = torch.reshape(self.c[i] * activations_i, (1, activations_i.size()[0]))
        c_j = torch.reshape(self.c[i + 1] * activations_j, (activations_j.size()[0], 1))

        # get the bias
        d_i = torch.reshape(self.d[i], (1, activations_i.size()[0]))
        d_j = torch.reshape(self.d[i + 1], (activations_j.size()[0], 1))

        # calculate the change in weights
        dw = pre_i + post_j + c_i * c_j + d_i * d_j

        # get the learning rate
        pre_eta = self.e[i].repeat(activations_j.size()[0], 1)

        # get the learning rate
        post_eta = torch.reshape(self.e[i + 1], (activations_j.size()[0], 1)).repeat((1, activations_i.size()[0]))

        # calculate the new weights
        nl = l + ((pre_eta + post_eta) / 2) * dw

        self.set_weights_layer(nl, i)


