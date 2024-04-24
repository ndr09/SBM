import os

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pickle


class NN():
    def __init__(self, nodes: list, Ntype: str = "host", eta: float = 0.1):
        self.nodes = nodes
        self.activations = [[0 for i in range(node)] for node in nodes]
        self.nweights = sum([self.nodes[i] * self.nodes[i + 1] for i in
                             range(len(self.nodes) - 1)])  # nodes[0]*nodes[1] + nodes[1]*nodes[2] + nodes[2]*nodes[3]

        self.weights = [[] for _ in range(len(self.nodes) - 1)]
        self.Ntype = Ntype
        self.eta = eta

    def activate(self, inputs):
        self.activations[0] = [np.tanh(x) for x in inputs]
        #print(f"input: {inputs}, actvs: {self.activations[0]}")
        for i in range(1, len(self.nodes)):
            self.activations[i] = [0. for _ in range(self.nodes[i])]
            for j in range(self.nodes[i]):
                sum = 0  # self.weights[i - 1][j][0]
                for k in range(self.nodes[i - 1]):
                    #print(f"k: {k}, i-1: {i-1}: {self.activations[i - 1][k - 1]}")
                    sum += self.activations[i - 1][k] * self.weights[i - 1][j][k]

                # use eta in the
                if self.Ntype == "guest" and (i == len(self.nodes) - 1):
                  self.activations[i][j] = sum * self.eta
                else:
                  self.activations[i][j] = np.tanh(sum)

        #print("-----activations------\n" + str(self.activations) + "\n-----------\n")
        return np.array(self.activations[-1])

    def set_weights(self, weights):
        # self.weights = [[] for _ in range(len(self.nodes) - 1)]
        c = 0
        for i in range(1, len(self.nodes)):
            self.weights[i - 1] = [[0 for _ in range(self.nodes[i - 1])] for __ in range(self.nodes[i])]
            for j in range(self.nodes[i]):
                for k in range(self.nodes[i - 1]):
                    self.weights[i - 1][j][k] = weights[c]
                    c += 1
        # print(c)

    def get_list_weights(self):
        wghts = []
        for i in range(1, len(self.nodes)):
            for j in range(self.nodes[i]):
                for k in range(self.nodes[i - 1]):
                    wghts.append(np.abs(self.weights[i - 1][j][k]))
        return wghts

    def nx_pos_because_it_was_too_hard_to_add_a_multipartite_from_a_list_as_it_works_for_the_bipartite(self, G):
        pos = {}
        nodes_G = list(G)
        input_space = 1.75 / self.nodes[0]
        output_space = 1.75 / self.nodes[-1]

        for i in range(self.nodes[0]):
            pos[i] = np.array([-1., i * input_space])

        c = 0
        for i in range(self.nodes[0] + self.nodes[1], sum(self.nodes)):
            pos[i] = np.array([1, c * output_space])
            c += 1

        center_node = []
        for n in nodes_G:
            if not n in pos:
                center_node.append(n)

        center_space = 1.75 / len(center_node)
        for i in range(len(center_node)):
            pos[center_node[i]] = np.array([0, i * center_space])
        return pos

    def nxpbiwthtaamfalaiwftb(self, G):
        return self.nx_pos_because_it_was_too_hard_to_add_a_multipartite_from_a_list_as_it_works_for_the_bipartite(G)

    def nn_prune_weights(self, prune_ratio, fold=None):
        wghts_abs = self.get_list_weights()
        thr = np.percentile(wghts_abs, prune_ratio)
        for i in range(1, len(self.nodes)):
            for j in range(self.nodes[i]):
                for k in range(self.nodes[i - 1]):
                    if np.abs(self.weights[i - 1][j][k]) <= thr:
                        self.weights[i - 1][j][k] = 0.
        if fold is not None:
            mat = self.from_list_to_matrix()
            graph = nx.from_numpy_matrix(mat, create_using=nx.DiGraph)
            plt.clf()
            pos = self.nxpbiwthtaamfalaiwftb(graph)
            nx.draw(graph, pos=pos, with_labels=True, font_weight='bold')
            # print("saving")
            plt.savefig(fold + "_init.png")

    def from_list_to_matrix(self):
        matrix = np.zeros((sum(self.nodes), sum(self.nodes)))
        # set inputs
        for i in range(1, len(self.nodes)):
            for j in range(self.nodes[i]):
                for k in range(self.nodes[i - 1]):
                    ix = k + sum(self.nodes[:i - 1])
                    ox = j + (sum(self.nodes[:i]))
                    matrix[ix][ox] = self.weights[i - 1][j][k]
        return matrix
    
class HNN(NN):
    """
    A neural network (HNN) with Hebbian learning rules for weight updates.

    Attributes
    ----------
    nodes : list
        The structure of the neural network, specified by the number of nodes in each layer.
    eta : float
        Learning rate (default is 0.1).
    hrules : list
        Hebbian learning rules for weight updates.

    Methods
    -------
    set_hrules(hrules)
        Set Hebbian learning rules for weight updates.
    update_weights()
        Update weights using Hebbian learning rules.
    """

    def __init__(self, nodes, eta=0.1):
        super().__init__(nodes)
        self.hrules = [[[0, 0, 0, 0] for i in range(node)] for node in nodes]
        self.eta = eta
        self.set_weights([0 for _ in range(self.nweights)])

    def set_hrules(self, hrules):
        self.hrules = [[] for _ in range(len(self.nodes) - 1)]
        c = 0

        # foreach layer, except input one
        for i in range(1, len(self.nodes)):
            # initialize a rule for each weight
            self.hrules[i - 1] = [[0 for a in range(self.nodes[i - 1])] for b in range(self.nodes[i])]

            # foreach node in layer i
            for j in range(self.nodes[i]):
                # foreach node in layer i - 1
                for k in range(self.nodes[i - 1]):
                    self.hrules[i - 1][j][k] = [hrules[c + i] for i in range(4)]
                    c += 4

    def update_weights(self):
        # foreach layer, except input one
        for l in range(1, len(self.nodes)):
            # foreach node in layer l
            for o in range(self.nodes[l]):
                dw = self.eta * (self.hrules[l - 1][o][0][0] * self.weights[l - 1][o][0] * self.activations[l][o] +
                                 self.hrules[l - 1][o][0][1] * self.activations[l][o] +
                                 self.hrules[l - 1][o][0][2] * self.weights[l - 1][o][0] +
                                 self.hrules[l - 1][o][0][3])
                self.weights[l - 1][o][0] += dw
                # foreach node in layer l - 1
                for i in range(1, self.nodes[l - 1]):
                    # print(self.hrules[l - 1][o][i])
                    dw = self.eta * (
                            self.hrules[l - 1][o][i][0] * self.activations[l - 1][i - 1] * self.activations[l][o] +
                            self.hrules[l - 1][o][i][1] * self.activations[l][o] +
                            self.hrules[l - 1][o][i][2] * self.activations[l - 1][i - 1] +
                            self.hrules[l - 1][o][i][3])
                    self.weights[l - 1][o][i] += dw