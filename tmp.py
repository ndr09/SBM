import numpy as np
import torch
import torch.nn as nn
import time
from scipy.stats import kstest
import gymnasium as gym



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
            self.networks = self.networks

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


class EP(nn.Module):
    def __init__(self):
        super().__init__()
        self.l = nn.Parameter(torch.rand(1))

    def forward(self, inputs):
        return self.l * inputs


class NHNN(NN):
    def __init__(self, nodes: list, device="cpu", init=None):
        super(NHNN, self).__init__(nodes, grad=False, device=device, init=init, wopt=False)

        self.hrules = []
        self.nparams = sum(self.nodes) * 5 - self.nodes[0] - self.nodes[-1]
        self.a = []
        self.b = []
        self.c = []
        self.d = []
        self.e = []
        self.set_hrules()

        self.dws=[]
        for i in range(len(self.networks)):
            ih = self.nodes[i]
            oh = self.nodes[i + 1]
            self.dws.append(torch.zeros((oh, ih),requires_grad=True))
        # self.hrules = torch.nn.ParameterList(self.hrules)
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
        # print(x)
        c = 0


        for l in self.networks:
            ih = self.nodes[c]
            oh = self.nodes[c + 1]
            dw = torch.zeros((oh, ih))

            if len(self.activations) > c + 1:
                for i in range(ih):
                    for o in range(oh):
                        ai = self.a[c][i].forward(self.activations[c][i])
                        # print("##################################",self.activations[c][i],self.a[c][i].l.weight,ai)

                        bj = self.b[c + 1][o].forward(self.activations[c + 1][o])

                        cij = self.c[c][i].forward(self.activations[c][i]) * self.c[c + 1][o].forward(
                            self.activations[c + 1][o])
                        dij = self.d[c][i].forward(torch.ones(1, dtype=torch.float)) * self.d[c + 1][o].forward(
                            torch.ones(1, dtype=torch.float))

                        eta  =  0.5 * (self.e[c][i].forward(torch.ones(1, dtype=torch.float)) + self.e[c + 1][o].forward(
                                torch.ones(1, dtype=torch.float)))
                        # print(eta, ai,bj,cij,dij)
                        dw[o, i] = eta * (ai + bj + cij + dij)
                        #print(eta, ai,bj,cij,dij, dw[o, i])

                self.dws[c] = dw
            # print(x0.size(), dw.size())

            dw1 = torch.matmul(dw,x0)
            # print(x0.size(), dw.size(), dw1)
            # print("=======")
            a = l(x0)
            x1 = a + dw1
            if not c == len(self.networks)-1:
                x1 = torch.tanh(x1)

            tmp.append(torch.reshape(torch.clone(x1), (x1.size()[0], 1)))
            #     tmp.append(torch.reshape(torch.clone(x1), (x1.size()[0], 1)))
            # else:
            #     tmp.append(torch.reshape(torch.tanh(torch.clone(x1)), (x1.size()[0], 1)))

            # print('aa',x1.size(),l(x0).size(),dw1.size())
            x0 = x1
            c += 1
        for i in range(len(self.networks)):
            self.set_weights_layer(self.dws[i],i)
        self.activations = tmp[:]
        return x1

    def forward_nu(self,inputs):
        x = torch.tanh(inputs)
        for l in self.networks[:-1]:
            x = torch.tanh(l(x))
        return self.networks[-1](x)

    def reset_weights(self, init="maintain"):
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
            elif type=='maintain':
                    l.weight.data = torch.clone(l.weight.data)
        self.activations = []

    def set_hrules(self):


        start = 0
        self.a.append([HP() for _ in range(self.nodes[0])])
        start += self.nodes[0]

        self.b.append([NP() for _ in range(self.nodes[0])])

        self.c.append([HP() for _ in range(self.nodes[0])])
        start += self.nodes[0]

        self.d.append([HP() for _ in range(self.nodes[0])])
        start += self.nodes[0]

        self.e.append([EP() for _ in range(self.nodes[0])])
        start += self.nodes[0]

        for l in self.nodes[1:-1]:
            self.a.append([HP() for _ in range(l)])
            start += self.nodes[l]

            self.b.append([HP() for _ in range(l)])
            start += self.nodes[l]

            self.c.append([HP() for _ in range(l)])
            start += self.nodes[l]

            self.d.append([HP() for _ in range(l)])
            start += self.nodes[l]

            self.e.append([EP() for _ in range(l)])
            start += self.nodes[l]

        self.a.append([NP() for _ in range(self.nodes[-1])])

        self.b.append([HP() for _ in range(self.nodes[-1])])
        start += self.nodes[-1]

        self.c.append([HP() for _ in range(self.nodes[-1])])
        start += self.nodes[-1]

        self.d.append([HP() for _ in range(self.nodes[-1])])
        start += self.nodes[-1]

        self.e.append([EP() for _ in range(self.nodes[-1])])
        start += self.nodes[-1]

    def set_weights_layer(self, weights, i):
        tmp = weights.clone().detach()
        #print("========",i,"========")
        wold = self.networks[i].weight.data
        #print("wold ",wold)
        #print("tmp ", weights)
        tmp += wold
        reg = torch.max(torch.abs(tmp))
        #print("\t reg",reg, tmp)
        tmp /= reg if not reg==0.  else 1.
        #print(tmp)
        #print("================")

        self.networks[i].weight.data = tmp

    def update_weights_layer(self, i, activations_i, activations_i1):
        weights = self.get_weights()
        l = weights[i]

        pre_i = torch.reshape(self.a[i] * activations_i, (1, activations_i.size()[0]))

        pre_i = pre_i.repeat((activations_i1.size()[0], 1))

        post_j = self.b[i + 1] * activations_i1

        post_j = torch.reshape(post_j, (activations_i1.size()[0], 1))

        post_j = post_j.repeat((1, activations_i.size()[0]))
        c_i = torch.reshape(self.c[i] * activations_i, (1, activations_i.size()[0]))
        c_j = torch.reshape(self.c[i + 1] * activations_i1, (activations_i1.size()[0], 1))
        # print(self.d)
        d_i = torch.reshape(self.d[i], (1, activations_i.size()[0]))
        d_j = torch.reshape(self.d[i + 1], (activations_i1.size()[0], 1))

        dw = pre_i + post_j + c_i * c_j + d_i * d_j

        pre_eta = self.e[i].repeat(activations_i1.size()[0], 1)
        post_eta = torch.reshape(self.e[i + 1], (activations_i1.size()[0], 1)).repeat((1, activations_i.size()[0]))
        nl = l + (pre_eta + post_eta) / 2 * dw

        self.set_weights_layer(nl, i)



def tr(model, lsfn, opti, it, inp):
    model.reset_weights('uni')
    s =10
    idx = torch.randperm(inp.nelement())
    inputs = inp.view(-1)[idx].view(inp.size())
    #torch.tensor([[float(i)] for i in range(1,s+1)], dtype=torch.float)
    y = inputs*2#torch.tensor([[float(i) ** 2] for i in range(10)], dtype=torch.float)
    torch.autograd.set_detect_anomaly(True)
    model.train()
    yh = torch.zeros((s, 1))
    yt = torch.zeros((s, 1))
    yt1 = torch.zeros((s, 1))

    for i in range(s):
        a = model.forward(torch.tensor([inputs[i, 0]]))
        # print(a)
        yh[i, 0] = a
        # yh = model.forward(inputs)

    with torch.no_grad():
        model.reset_weights('maintain')

        for i in range(s):
            a = model.forward_nu(torch.tensor([inputs[i, 0]]))
            yt1[i, 0] = a

    with torch.no_grad():
        model.reset_weights('uni')

        for i in range(s):
            a = model.forward(torch.tensor([inputs[i, 0]]))
            yt[i, 0] = a


    # print(yh.shape)
    # print(y.shape)
    print("---------------------")
    # for l in model.networks:
    #     print(l.weight.data)
    loss = lsfn(yh, y)

    print(it," ",loss)
    print("out",torch.clone(yh.flatten()).detach().numpy())
    print("nup",torch.clone(yt1.flatten()).detach().numpy())
    print("res",torch.clone(yt.flatten()).detach().numpy())

    # print(yt.flatten())
    # print(yt1.flatten())

    print(y.flatten())

    loss.backward(retain_graph=True )
    # for r in model.a:
    #     print('pre',r[0].l,r[0].l.grad)
    opti.step()
    opti.zero_grad()
    # for r in model.a:
    #     print('post',r[0].l,r[0].l.grad)
    # for l in model.networks:
    #     print(l.weight.data)

if __name__ == "__main__":
    loss_fn = torch.nn.MSELoss()
    np.set_printoptions(precision=5,suppress=True)
    lr = 0.1
    model = NHNN([1, 2, 1], init='uni')
    # model.set_hrules2([1. for i in range(model.nparams.item())])
    # model.reset_weights()
    model = model.to(torch.float)
    model.reset_weights()

    # print(" PARAMS ")
    # print(list(model.params))
    # for p in model.params:
    #     print(p)

    # print("/////////////////////////////////////////////")
    optimizer = torch.optim.SGD(model.params, lr=lr)
    s = 10
    inputs = torch.rand((s, 1))
    for i in range(100):
        tr(model, loss_fn, optimizer, i, inputs)
