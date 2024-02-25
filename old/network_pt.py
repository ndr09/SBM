import numpy as np
import torch
import torch.nn as nn
import time
from scipy.stats import kstest

class NN(nn.Module):
    def __init__(self, nodes: list, grad=False, init=None, device="cpu", wopt=True):
        super().__init__()
        self.device = torch.device(device)
        self.nodes = torch.tensor(nodes).to(self.device)
        self.nweights = sum([self.nodes[i] * self.nodes[i + 1] for i in
                             range(len(self.nodes) - 1)])  # nodes[0]*nodes[1]+nodes[1]*nodes[2]+nodes[2]*nodes[3]

        self.networks = []
        self.activations = []
        self.grad = grad
        for i in range(len(nodes) - 1):
            self.networks.append(nn.Linear(nodes[i], nodes[i + 1], bias=False))

        if wopt:
            self.networks = nn.ParameterList(self.networks)

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
            elif init == 'zero':
                torch.nn.init.zeros_(l.weight.data)
        self.float()

    def forward(self, inputs):

        self.activations = []
        x = inputs.to(self.device)
        self.activations.append(torch.clone(x).to(self.device))
        # print(x)
        c = 0
        for l in self.networks:
            x = l(x)
            # print(x, l.weight.data)
            x = torch.tanh(x)

            c += 1
            self.activations.append(torch.clone(x))

        return x

    def get_weights(self):
        tmp = []
        for l in self.networks:
            tmp.append(l.weight)
        return tmp

    def set_weights(self, weights):
        if type(weights) == list and type(weights[0]) == torch.Tensor:
            for i in range(len(self.networks)):
                self.networks[i].weight.data = weights[i]
        elif len(weights) == self.nweights:
            tmp = self.get_weights()
            start = 0
            i = 0
            for l in tmp:
                size = l.size()[0] * l.size()[1] + start
                params = torch.tensor(weights[start:size], requires_grad=self.grad)
                start = size
                rsh = torch.reshape(params, (l.size()[0], l.size()[1]))
                self.networks[i].weight.data = rsh
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

    def set_etas(self, etas: list):
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
    def __init__(self, nodes: list, hrules=None, grad=False, device="cpu", init=None):
        super(NHNN, self).__init__(nodes, grad=grad, device=device, init=init,wopt=False)

        self.hrules = []
        self.nparams = sum(self.nodes) * 5 - self.nodes[0] - self.nodes[-1]
        if hrules is not None:
            self.set_hrules(hrules)
        # self.hrules = torch.nn.ParameterList(self.hrules)
        self.float()

    def forward(self, inputs):

        self.activations = []
        x0 = inputs
        self.activations.append(torch.clone(x0))
        # print(x)
        c = 0

        for l in self.networks:
            x1 = l(x0)
            x1 = torch.tanh(x1)
            self.activations.append(torch.clone(x1))
            self.update_weights_layer(c, x0, x1)
            x0 = x1
            c += 1

        return x1


    def reset_weights(self):
        self.set_weights(torch.zeros(self.nweights).detach().numpy().tolist())

    def set_hrules2(self, hrules):
        a = []
        b = []
        c = []
        d = []
        e = []

        start = 0
        a.append(torch.tensor(hrules[start:start+self.nodes[0]]))
        start += self.nodes[0]

        b.append(torch.zeros(self.nodes[0]))

        c.append(torch.tensor(hrules[start:start + self.nodes[0]]))
        start += self.nodes[0]

        d.append(torch.tensor(hrules[start:start + self.nodes[0]]))
        start += self.nodes[0]

        e.append(torch.tensor(hrules[start:start + self.nodes[0]]))
        start += self.nodes[0]

        # print(self.nodes.tolist()[1,-1])
        for l in self.nodes[1:-1]:
            a.append(torch.tensor(hrules[start:start + self.nodes[l]]))
            start += self.nodes[l]

            b.append(torch.tensor(hrules[start:start + self.nodes[l]]))
            start += self.nodes[l]

            c.append(torch.tensor(hrules[start:start + self.nodes[l]]))
            start += self.nodes[l]

            d.append(torch.tensor(hrules[start:start + self.nodes[l]]))
            start += self.nodes[l]

            e.append(torch.tensor(hrules[start:start + self.nodes[l]]))
            start += self.nodes[l]


        a.append(torch.zeros(self.nodes[-1]))

        b.append(torch.tensor(hrules[start:start + self.nodes[-1]]))
        start += self.nodes[-1]

        c.append(torch.tensor(hrules[start:start + self.nodes[-1]]))
        start += self.nodes[-1]

        d.append(torch.tensor(hrules[start:start + self.nodes[-1]]))
        start += self.nodes[-1]

        e.append(torch.tensor(hrules[start:start + self.nodes[-1]]))
        start += self.nodes[-1]

        self.a = torch.nn.ParameterList(a)
        self.b = torch.nn.ParameterList(b)
        self.c = torch.nn.ParameterList(c)
        self.d = torch.nn.ParameterList(d)
        self.e = torch.nn.ParameterList(e)



    def set_hrules(self, hrules: list):
        assert len(hrules) == sum(self.nodes) * 5 - self.nodes[0] - self.nodes[-1], "needed " + str(
            sum(self.nodes) * 5 - self.nodes[0] - self.nodes[-1]) + " received " + str(len(hrules))
        start = 0

        size = self.nodes[0] * 4 + start
        tmp = np.reshape(hrules[start:size], (self.nodes[0], 4))
        tmp1 = np.zeros((self.nodes[0], 5))
        for i in range(self.nodes[0]):
            tmp1[i] = np.insert(tmp[i], 1, 0.)

        params = torch.tensor(tmp1)
        self.hrules.append(params)

        for l in self.nodes[1:-1]:
            size = l * 5 + start
            params = torch.tensor(hrules[start:size])
            self.hrules.append(torch.reshape(params, (l, 5)))

            start = size

        size = self.nodes[-1] * 4 + start
        params = torch.tensor(hrules[start:size])
        tmp = torch.reshape(params, (self.nodes[-1], 4))
        tmp1 = torch.tensor([[0.] for i in range(self.nodes[-1])])
        self.hrules.append(torch.hstack((tmp1, tmp)))

        self.hrules = torch.nn.ParameterList(self.hrules)

    def set_weights_layer(self, weights, i):

        self.networks[i].weight.data = weights#torch.nn.Parameter(weights)


    def update_weights_layer(self, i, activations_i, activations_i1):
        weights = self.get_weights()
        l = weights[i]

        # hrule_i = self.hrules[i]
        # hrule_i1 = self.hrules[i + 1]
        # print(self.a[i] * activations_i)
        pre_i = torch.reshape(self.a[i] * activations_i, (1, activations_i.size()[0]))
        print("a_i", activations_i)
        print("A*a_i",pre_i)
        print("A",self.a[i])
        print("aa ", self.a[i]*activations_i)
        # exit(0)
        # print("ABCD",hrule_i1)


        pre_i = pre_i.repeat((activations_i1.size()[0], 1))

        post_j = torch.reshape(self.b[i+1] * activations_i1, (activations_i1.size()[0], 1))
        post_j = post_j.repeat((1, activations_i.size()[0]))

        c_i = torch.reshape(self.c[i] * activations_i, (1, activations_i.size()[0]))
        c_j = torch.reshape(self.c[i+1] * activations_i1, (activations_i1.size()[0], 1))
        # print(self.d)
        d_i = torch.reshape(self.d[i], (1, activations_i.size()[0]))
        d_j = torch.reshape(self.d[i+1], (activations_i1.size()[0], 1))

        dw = pre_i + post_j + c_i * c_j + d_i * d_j

        pre_eta = self.e[i].repeat(activations_i1.size()[0], 1)
        post_eta = torch.reshape(self.e[i+1], (activations_i1.size()[0], 1)).repeat((1, activations_i.size()[0]))
        nl = l+ (pre_eta + post_eta) / 2 * dw
        print("\\\\\\\\",nl)
        print("=========", self.a[1])
        # exit(0)

        self.set_weights_layer(nl, i)
        exit(1)

    def update_weights_layer2(self, i, activations_i, activations_i1):
        weights = self.get_weights()
        l = weights[i]

        hrule_i = self.hrules[i]
        hrule_i1 = self.hrules[i + 1]
        print(hrule_i[:, 0] * activations_i)
        pre_i = torch.reshape(hrule_i[:, 0] * activations_i, (1, activations_i.size()[0]))
        print(1,pre_i)
        print(2,hrule_i[:, 0])
        print(3,hrule_i)
        exit(0)

        pre_i = pre_i.repeat((activations_i1.size()[0], 1))

        post_j = torch.reshape(hrule_i1[:, 1] * activations_i1, (activations_i1.size()[0], 1))
        post_j = post_j.repeat((1, activations_i.size()[0]))

        c_i = torch.reshape(torch.where(hrule_i[:, 2] == 1., 1., hrule_i[:, 2] * activations_i),
                            (1, activations_i.size()[0]))
        c_j = torch.reshape(torch.where(hrule_i1[:, 2] == 1., 1., hrule_i1[:, 2] * activations_i1),
                            (activations_i1.size()[0], 1))
        d_i = torch.reshape(hrule_i[:, 3], (1, activations_i.size()[0]))
        d_j = torch.reshape(hrule_i1[:, 3], (activations_i1.size()[0], 1))

        dw = pre_i + post_j + c_i * c_j + d_i * d_j

        pre_eta = hrule_i[:, 4].repeat(activations_i1.size()[0], 1)
        post_eta = torch.reshape(hrule_i1[:, 4], (activations_i1.size()[0], 1)).repeat((1, activations_i.size()[0]))
        nl = l+ (pre_eta + post_eta) / 2 * dw
        print(nl)
        print("=========", hrule_i1)
        self.set_weights_layer(nl, i)


class NHNN2(NN):
    def __init__(self, nodes: list, eta: float, window: float, hrules=None, grad=False, device="cpu", init=None):
        super(NHNN2, self).__init__(nodes, grad=grad, device=device, init=init)

        self.hrules = []
        self.nparams = sum(self.nodes) * 5 - self.nodes[0] - self.nodes[-1]
        self.eta = []
        self.activations = [[] for _ in range(window)]
        self.window = window
        self.t = 0
        self.set_eta([eta for _ in range(sum(self.nodes))])
        self.eta = None
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

    def set_eta(self, etas: list):
        assert len(etas) == sum(self.nodes), "needed " + str(
            sum(self.nodes)) + " received " + str(len(etas))
        self.eta = []
        start = 0
        for l in self.nodes:
            self.eta.append(torch.tensor(etas[start:start + l]).to(self.device))
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

            dw = pre_i + post_j + torch.where((c_i == 1.) & (c_j == 1.), 0, c_i * c_j) + torch.where(
                (d_i == 1.) & (d_j == 1.), 0, d_i * d_j)
            dws.append(dw)
            # print(self.eta[i].repeat(activations_i1.size()[0], 1).size(),
            #       torch.reshape(self.eta[i+1], (activations_i1.size()[0], 1)).repeat((1, activations_i.size()[0])).size(),
            #       dw.size())
            pre_eta = self.eta[i].repeat(activations_i1.size()[0], 1)
            post_eta = torch.reshape(self.eta[i + 1], (activations_i1.size()[0], 1)).repeat(
                (1, activations_i.size()[0]))
            l += (pre_eta + post_eta) / 2 * dw

        self.set_weights(weights)

    def appfor(self, inputs):
        mw = self.t if self.t < self.window else self.window
        ps = self.t % self.window
        appw = [torch.zeros(l.shape) for l in self.get_weights()]
        # print("################",mw)
        for i in range(mw):
            tmp = self.calc_dw(i)

            for l in range(len(tmp)):
                appw[l] += tmp[l]
        for l in range(len(self.networks)):
            self.networks[l].weight.data = nn.Parameter(appw[l], requires_grad=self.grad)
            # print(appw[l])

        with torch.no_grad():
            self.activations[ps] = []
            x = inputs.to(self.device)
            self.activations[ps].append(torch.clone(x).to(self.device))

            c = 0
            for l in self.networks:
                x = l(x)
                x = torch.tanh(x)

                c += 1
                self.activations[ps].append(torch.clone(x))
            self.t += 1

            return x

    def calc_dw(self, time):
        weights = self.get_weights()
        num_layers = len(weights)

        dws = []

        for i in range(num_layers):
            l = torch.zeros(weights[i].shape)

            l_size = l.shape
            activations_i = self.activations[time][i].to(self.device)
            activations_i1 = self.activations[time][i + 1].to(self.device)
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

            dw = pre_i + post_j + torch.where((c_i == 1.) & (c_j == 1.), 0, c_i * c_j) + torch.where(
                (d_i == 1.) & (d_j == 1.), 0, d_i * d_j)
            # print(self.eta[i].repeat(activations_i1.size()[0], 1).size(),
            #       torch.reshape(self.eta[i+1], (activations_i1.size()[0], 1)).repeat((1, activations_i.size()[0])).size(),
            #       dw.size())
            pre_eta = self.eta[i].repeat(activations_i1.size()[0], 1)
            post_eta = torch.reshape(self.eta[i + 1], (activations_i1.size()[0], 1)).repeat(
                (1, activations_i.size()[0]))
            dw = (pre_eta + post_eta) / 2 * dw
            dws.append(dw)

        return dws


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

def tr(model, lsfn, opti):
    # model.reset_weights()
    inputs = torch.tensor([[float(i)] for i in range(10)], dtype=torch.float)
    y = torch.tensor([[float(i)**2] for i in range(10)], dtype=torch.float)

    model.train()
    yh = torch.zeros((10,1))
    for i in range(10):
         yh[i,0] = model.forward(torch.tensor([inputs[i,0]]))
    #yh = model.forward(inputs)

    print(yh.shape)
    print(y.shape)

    loss = lsfn(yh,y)
    print(loss)

    loss.backward()
    for l in model.a:
        print("############# ",l.grad, l)
    opti.step()
    opti.zero_grad()
    for l in model.a:
        print("@@@@@@@@@@@@@@@",l.grad, l)

if __name__ == "__main__":
    loss_fn = torch.nn.MSELoss()
    lr = 0.2
    model =NHNN([1,2,2,1],init='uni')
    model.set_hrules2([1. for i in range(model.nparams.item())])
    # model.reset_weights()
    model = model.to(torch.float)
    print(" PARAMS ")
    print(list(model.parameters()))
    for p in model.parameters():
        print(p)

    print("/////////////////////////////////////////////")
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    for i in range(1):
        # print('ph',model.hrules[-1])
        # print('pw',model.get_weights()[1])
        tr(model,loss_fn, optimizer)
        # print("w",model.get_weights()[1])
        #
        # print('h',model.hrules[-1])
        # print("=============")

    # w
    # Parameter
    # containing:
    # tensor([[37.9631, 37.9631],
    #         [37.9631, 37.9631]], requires_grad=True)

    #
    # print(model.forward(torch.tensor([1., 1., 1., 1.])))
    # for i in range(len(model.hrules)):
    #     print(i, model.hrules[i])
    #
    # print(model.hrules[-1])
    # print(model.networks[0])
