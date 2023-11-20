import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from scipy.stats import kstest


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
    def __init__(self, nodes: list, grad=False, init=None, device="cpu"):
        super(NN, self).__init__()
        self.device = torch.device(device)
        self.nodes = torch.tensor(nodes).to(self.device)
        self.nweights = sum([self.nodes[i] * self.nodes[i + 1] for i in
                             range(len(self.nodes) - 1)])  # nodes[0]*nodes[1]+nodes[1]*nodes[2]+nodes[2]*nodes[3]

        self.networks = []
        self.activations = []
        self.grad = grad
        for i in range(len(nodes) - 1):
            self.networks.append(nn.Linear(nodes[i], nodes[i + 1], bias=False))
        if init is None:
            self.set_weights([0. for _ in range(self.nweights)])
        else:
            for l in self.networks:
                if init == 'xa_uni':
                    torch.nn.init.xavier_uniform(l.weight.data, 0.3)
                elif init == 'sparse':
                    torch.nn.init.sparse_(l.weight.data, 0.8)
                elif init == 'uni':
                    torch.nn.init.uniform_(l.weight.data, -0.1, 0.1)
                elif init == 'normal':
                    torch.nn.init.normal_(l.weight.data, 0, 0.024)
                elif init == 'ka_uni':
                    torch.nn.init.kaiming_uniform_(l.weight.data, 3)
                elif init == 'uni_big':
                    torch.nn.init.uniform_(l.weight.data, -1, 1)
                elif init == 'xa_uni_big':
                    torch.nn.init.xavier_uniform(l.weight.data)
        self.double()

    def forward(self, inputs):
        with torch.no_grad():
            self.activations = []
            x = inputs.to(self.device)
            self.activations.append(torch.clone(x).to(self.device))
            c = 0
            for l in self.networks:
                x = l(x)
                x = torch.tanh(x)
                c+=1
                self.activations.append(torch.clone(x))

            return x

    def get_weights(self):
        tmp = []
        for l in self.networks:
            tmp.append(l.weight.data)
        return tmp

    def set_weights(self, weights):
        if type(weights) == list and type(weights[0]) == torch.Tensor:
            for i in range(len(self.networks)):
                self.networks[i].weight = nn.Parameter(weights[i], requires_grad=self.grad)
        elif len(weights) == self.nweights:
            tmp = self.get_weights()
            start = 0
            i = 0
            for l in tmp:
                size = l.size()[0] * l.size()[1] + start
                params = torch.tensor(weights[start:size])
                start = size
                self.networks[i].weight = nn.Parameter(
                    torch.reshape(params, (l.size()[0], l.size()[1])).to(self.device))
                i += 1


class HNN(NN):
    def __init__(self, nodes: list, eta: float, hrules=None, grad=False, init=None):
        super(HNN, self).__init__(nodes, grad=grad, init=init)

        self.hrules = []
        self.eta = eta
        start = 0
        if hrules is not None:
            self.set_hrules(hrules)

    def set_hrules(self, hrules: list):
        assert len(hrules) == self.nweights * 4, "needed " + str(
            self.nweights * 4) + " received " + str(len(hrules))
        start = 0
        for l in self.get_weights():
            size = l.size()[0] * l.size()[1] * 4 + start
            params = torch.tensor(hrules[start:size])
            self.hrules.append(torch.reshape(params, (l.size()[0], l.size()[1], 4)))
            start = size
    def set_etas(self, etas:list):
        start = 0
        self.eta = []
        for l in self.get_weights():
            size = l.size()[0] * l.size()[1] + start
            params = torch.tensor(etas[start:size])
            self.eta.append(torch.reshape(params, (l.size()[0], l.size()[1])))
            start = size

    def update_weights(self):

        weights = self.get_weights()
        for i in range(len(weights)):
            l = weights[i]
            activations_i = self.activations[i].to(self.device)
            activations_i1 = torch.reshape(self.activations[i + 1].to(self.device),
                                           (self.activations[i + 1].size()[0], 1))
            hrule_i = self.hrules[i].to(self.device)
            # la size dovra essere l1, l
            pre = hrule_i[:, :, 0] * activations_i
            post = hrule_i[:, :, 1] * activations_i1
            C_i = activations_i * hrule_i[:, :, 2]
            C_j = activations_i1 * hrule_i[:, :, 2]
            C = C_i * C_j
            D = hrule_i[:, :, 3]
            dw = pre + post + C + D
            weights[i] += self.eta[i] * dw

        self.set_weights(weights)


class NHNN(NN):
    def __init__(self, nodes: list, eta: float, hrules=None, grad=False, device="cpu", init=None):
        super(NHNN, self).__init__(nodes, grad=grad, device=device, init=init)

        self.hrules = []
        self.nparams = sum(self.nodes) * 4 - self.nodes[0] - self.nodes[-1]
        self.eta = []
        self.set_eta([eta for _ in range(sum(self.nodes))])
        self.eta
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

    def set_eta(self, etas:list):
        assert len(etas) == sum(self.nodes), "needed " + str(
            sum(self.nodes)) + " received " + str(len(etas))
        self.eta = []
        start = 0
        for l in self.nodes:
            self.eta.append(torch.tensor(etas[start:start+l]).to(self.device))
            start += l


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

            c_i = torch.reshape(torch.where(hrule_i[:, 2] == 1., 1., hrule_i[:, 2] * activations_i),
                                (1, activations_i.size()[0]))
            c_j = torch.reshape(torch.where(hrule_i1[:, 2] == 1., 1., hrule_i1[:, 2] * activations_i1),
                                (activations_i1.size()[0], 1))
            d_i = torch.reshape(hrule_i[:, 3], (1, activations_i.size()[0]))
            d_j = torch.reshape(hrule_i1[:, 3], (activations_i1.size()[0], 1))

            dw = pre_i + post_j + torch.where((c_i == 1.) & (c_j == 1.), 0, c_i * c_j) + torch.where((d_i == 1.) & (d_j == 1.), 0, d_i * d_j)
            dws.append(dw)
            # print(self.eta[i].repeat(activations_i1.size()[0], 1).size(),
            #       torch.reshape(self.eta[i+1], (activations_i1.size()[0], 1)).repeat((1, activations_i.size()[0])).size(),
            #       dw.size())
            pre_eta = self.eta[i].repeat(activations_i1.size()[0], 1)
            post_eta =  torch.reshape(self.eta[i+1], (activations_i1.size()[0], 1)).repeat((1, activations_i.size()[0]))
            l += (pre_eta+post_eta)/2 * dw

        self.set_weights(weights)


class ANHNN(NHNN):
    def __init__(self, nodes: list, eta: float, history_length: int, stability_window_size: int, hrules=None,
                 grad=False, device="cpu"):
        super(ANHNN, self).__init__(nodes, eta, hrules, grad=grad, device=device)
        self.hl = torch.tensor(history_length).to(self.device)
        self.hi = torch.tensor(0).to(self.device)  # History index
        self.sws = torch.tensor(stability_window_size).to(self.device)
        self.nins = torch.tensor(sum(nodes[1:])).to(self.device)
        self.ah = torch.zeros((self.hl, self.nins)).to(self.device)
        self.hrules_node = None
        if not len(self.hrules) == 0:
            self.hrules_node = self.get_hrules_for_nodes()
        self.stable_nodes = set()

    def _copy_act(self):
        return torch.concat(self.activations[1:])

    def store_activation(self):
        if self.hi < self.hl:
            self.ah[self.hi] = self._copy_act()
            self.hi += 1
        else:
            self.ah = torch.vstack((self.ah, self._copy_act()))[1:]
            self.prune_stable_nodes()

    def get_hrules_for_nodes(self):
        return torch.concat(self.hrules[1:])

    def check_stability(self):
        stability = torch.zeros(self.nins, dtype=torch.int8)
        # print(torch.transpose(self.ah, 0, 1).size())
        x = torch.t(self.ah)[:, :self.hl - self.sws]
        y = torch.t(self.ah)[:, self.hl - self.sws:]
        for i in range(self.nins):
            if i not in self.stable_nodes:
                stability[i] = 1 if kstest(y[i], x[i], method="asymp")[1] > 0.05 else 0
                if stability[i] == 1:
                    self.stable_nodes.add(i)
            else:
                stability[i] = 1

        return stability

    def prune(self, hrules, stability):
        stability = torch.reshape(stability, (stability.size()[0], 1))
        pruned = torch.where(stability == 0, hrules, torch.tensor([[0., 0., 1., 1., ]]))
        # print("pruning", hrules.size(), pruned.size())
        return pruned

    def set_hrules_ft(self, new_rules):
        start = 0
        tmp = []
        tmp.append(self.hrules[0].clone())
        for i in range(1, len(self.hrules)):
            end = start + self.hrules[i].size()[0]
            # print(i,start, end, new_rules[start:end].clone(), new_rules.size())
            tmp.append(new_rules[start:end].clone())
            start += end
        self.hrules = tmp

    def prune_stable_nodes(self):
        stability = self.check_stability()
        if self.hrules_node is None:
            self.hrules_node = self.get_hrules_for_nodes()
        hrules = self.prune(self.hrules_node, stability)
        self.set_hrules_ft(hrules)


if __name__ == "__main__":
    model = HNN([4, 2, 2,5], 0.1, hrules=[float(i) for i in range(88)], init="rand")
    model.set_etas([0.5 for i in range(22)])
    print(model.get_weights())
    #model.set_weights([float(i) for i in range(model.nweights)])
    print("f", model.forward(torch.tensor([1., 1., 1., 1.])))
    model.update_weights()
    #
    # print(model.get_weights())
    print(model.hrules)
    print(model.hrules[1].size())
