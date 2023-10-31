import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time


# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, 3, 1)
#         self.conv2 = nn.Conv2d(32, 64, 3, 1)
#         self.dropout1 = nn.Dropout(0.25)
#         self.dropout2 = nn.Dropout(0.5)
#         self.fc1 = nn.Linear(9216, 128)
#         self.fc2 = nn.Linear(128, 10)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = F.relu(x)
#         x = self.conv2(x)
#         x = F.relu(x)
#         x = F.max_pool2d(x, 2)
#         x = self.dropout1(x)
#         x = torch.flatten(x, 1)
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.dropout2(x)
#         x = self.fc2(x)
#         output = F.log_softmax(x, dim=1)
#         return output


class NN(nn.Module):
    def __init__(self, nodes: list, grad=False, device="cpu"):
        super(NN, self).__init__()
        self.device = torch.device(device)
        self.nodes = torch.tensor(nodes).to(self.device)
        self.nweights = sum([self.nodes[i] * self.nodes[i + 1] for i in
                             range(len(self.nodes) - 1)])  # nodes[0]*nodes[1]+nodes[1]*nodes[2]+nodes[2]*nodes[3]

        self.networks = []
        self.activations = []
        for i in range(len(nodes) - 1):
            self.networks.append(nn.Linear(nodes[i], nodes[i + 1], bias=False))
        self.set_weights([0. for _ in range(self.nweights)])
        self.double()


    def forward(self, inputs):
        with torch.no_grad():
            self.activations = []
            x = inputs.to(self.device)
            self.activations.append(torch.clone(x).to(self.device))

            for l in self.networks:
                x = l(x)
                self.activations.append(torch.clone(x))
                x = torch.tanh(x)

            return x

    def get_weights(self):
        tmp = []
        for l in self.networks:
            tmp.append(l.weight.data)
        return tmp

    def set_weights(self, weights):

        if type(weights) == list and type(weights[0]) == torch.Tensor:
            for i in range(len(self.networks)):
                self.networks[i].weight = nn.Parameter(weights[i])
        elif len(weights) == self.nweights:
            tmp = self.get_weights()
            start = 0
            i = 0
            for l in tmp:
                size = l.size()[0] * l.size()[1] + start
                params = torch.tensor(weights[start:size])
                start = size
                self.networks[i].weight = nn.Parameter(torch.reshape(params, (l.size()[0], l.size()[1])).to(self.device))
                i += 1


class HNN(NN):
    def __init__(self, nodes: list, eta: float, hrules=None, grad=False):
        super(HNN, self).__init__(nodes, grad=grad)

        self.hrules = []
        self.eta = eta
        start = 0
        if hrules is not None:
            self.set_hrules(hrules)

    def set_hrules(self, hrules: list):
        assert len(hrules) == self.nweights * 4
        start = 0
        for l in self.get_weights():
            size = l.size()[0] * l.size()[1] * 4 + start
            params = torch.tensor(hrules[start:size])
            self.hrules.append(torch.reshape(params, (l.size()[0], l.size()[1], 4)))
            start = size

    def update_weights(self):

        weights = self.get_weights()
        for i in range(len(weights)):
            l = weights[i]
            activations_i = self.activations[i].to(self.device)
            activations_i1 = torch.reshape(self.activations[i + 1].to(self.device), (self.activations[i + 1].size()[0],1))
            hrule_i = self.hrules[i].to(self.device)
            # la size dovra essere l1, l
            pre = hrule_i[:,:,0]*activations_i
            post = hrule_i[:,:,1]*activations_i1
            C_i = activations_i*hrule_i[:,:,2]
            C_j = activations_i1*hrule_i[:,:,2]
            C = C_i*C_j
            D = hrule_i[:,:,3]
            dw = pre+post+C+D
            weights[i] += self.eta*dw

        self.set_weights(weights)


class NHNN(NN):
    def __init__(self, nodes: list, eta: float, hrules=None, grad=False, device="cpu"):
        super(NHNN, self).__init__(nodes, grad=grad, device=device)

        self.hrules = []
        self.nparams = sum(self.nodes) * 4 - self.nodes[0] - self.nodes[-1]
        self.eta = torch.tensor(eta).to(self.device)
        if hrules is not None:
            self.set_hrules(hrules)

    def set_hrules(self, hrules: list):

        assert len(hrules) == sum(self.nodes) * 4 - self.nodes[0] - self.nodes[-1], "needed " + str(
            sum(self.nodes) * 4 - self.nodes[0] - self.nodes[-1]) + " received " + str(len(hrules))
        start = 0
        size = self.nodes[0] * 3 + start
        tmp = np.reshape(hrules[start:size], (self.nodes[0], 3))
        tmp1 = np.zeros((self.nodes[0], 4))
        for i in range(self.nodes[0]):
            tmp1[i] = np.insert(tmp[i], 1, 0.)

        params = torch.tensor(tmp1)
        self.hrules.append(params)

        for l in self.nodes[1:-1]:
            size = l * 4 + start
            params = torch.tensor(hrules[start:size])
            self.hrules.append(torch.reshape(params, (l, 4)))

            start = size

        size = self.nodes[-1] * 3 + start
        params = torch.tensor(hrules[start:size])
        tmp = torch.reshape(params, (self.nodes[-1], 3))
        tmp1 = torch.tensor([[0.] for i in range(self.nodes[-1])])
        self.hrules.append(torch.hstack((tmp1, tmp)).to(self.device))


    def update_weights(self):
        weights = self.get_weights()
        num_layers = len(weights)
        t = time.time()
        dws = []
        for i in range(num_layers):
            l = weights[i]

            l_size = l.shape
            activations_i = self.activations[i].to(self.device)
            activations_i1 = self.activations[i + 1].to(self.device)
            hrule_i = self.hrules[i].to(self.device)
            hrule_i1 = self.hrules[i + 1].to(self.device)

            # Use broadcasting to perform element-wise operations without loops

            pre_i = torch.reshape(hrule_i[:, 0] * activations_i, (1, activations_i.size()[0]))
            pre_i = pre_i.repeat((activations_i1.size()[0], 1))
            post_j = torch.reshape(hrule_i1[:, 1] * activations_i1, (activations_i1.size()[0], 1))
            post_j = post_j.repeat((1, activations_i.size()[0]))
            c_i = torch.reshape(hrule_i[:, 2] * activations_i, (1, activations_i.size()[0]))
            c_j = torch.reshape(hrule_i1[:, 2] * activations_i1, (activations_i1.size()[0],1))
            d_i = torch.reshape(hrule_i[:, 3], (1,activations_i.size()[0]))
            d_j = torch.reshape(hrule_i1[:, 3], ( activations_i1.size()[0],1))

            dw = pre_i + post_j + c_i * c_j + d_i * d_j
            dws.append(dw)
            l += self.eta*dw
        self.set_weights(weights)


if __name__ == "__main__":
    model = NHNN([4, 2, 1], 0.1, [float(i) for i in range(23)])
    # print(model.get_weights())
    # print([t.size() for t in model.hrules])
    # print([t.size() for t in model.get_weights()])
    # w = [torch.tensor([[1., 1., 1., 1.], [1., 1., 1., 1.]]), torch.tensor([[2., 2.]])]
    model.set_weights([float(i) for i in range(model.nweights)])
    #
    # print(model.get_weights())
    model.forward(torch.tensor([1., 1., 1., 1.]))
    model.update_weights1()
    print(model.get_weights())
