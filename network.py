import os

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pickle


class NN():
    def __init__(self, nodes):
        self.nodes = nodes
        self.activations = [[0 for i in range(node)] for node in nodes]
        self.nweights = nodes[0]*nodes[1]+nodes[1]*nodes[2]

        self.weights = [[] for _ in range(len(self.nodes) - 1)]

    def activate(self, inputs):
        self.activations[0] = [np.tanh(x) for x in inputs]
        for i in range(1, len(self.nodes)):
            self.activations[i] = [0. for _ in range(self.nodes[i])]
            for j in range(self.nodes[i]):
                sum = self.weights[i - 1][j][0]
                for k in range(1, self.nodes[i - 1] ):
                    sum += self.activations[i - 1][k - 1] * self.weights[i - 1][j][k]
                self.activations[i][j] = np.tanh(sum)
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
        #print(c)


class HNN(NN):
    def __init__(self, nodes, eta=0.1):
        super().__init__(nodes)
        self.hrules = [[[0, 0, 0, 0] for i in range(node)] for node in nodes]
        self.eta = eta
        self.set_weights([0 for _ in range(self.nweights)])

    def set_hrules(self, hrules):
        self.hrules = [[] for _ in range(len(self.nodes) - 1)]
        c = 0
        for i in range(1, len(self.nodes)):
            self.hrules[i - 1] = [[0 for a in range(self.nodes[i - 1] )] for b in range(self.nodes[i])]
            for j in range(self.nodes[i]):
                for k in range(self.nodes[i - 1] ):
                    self.hrules[i - 1][j][k] = [hrules[c + i] for i in range(4)]
                    c += 4

    def update_weights(self):
        for l in range(1, len(self.nodes)):
            for o in range(self.nodes[l]):
                # print(self.hrules[l - 1][o][0])
                dw = self.eta * (self.hrules[l - 1][o][0][0] * self.weights[l - 1][o][0] * self.activations[l][o] +
                                 self.hrules[l - 1][o][0][1] * self.activations[l][o] +
                                 self.hrules[l - 1][o][0][2] * self.weights[l - 1][o][0] +
                                 self.hrules[l - 1][o][0][3])
                self.weights[l - 1][o][0] += dw
                for i in range(1, self.nodes[l - 1] ):
                    # print(self.hrules[l - 1][o][i])
                    dw = self.eta * (
                            self.hrules[l - 1][o][i][0] * self.activations[l - 1][i - 1] * self.activations[l][o] +
                            self.hrules[l - 1][o][i][1] * self.activations[l][o] +
                            self.hrules[l - 1][o][i][2] * self.activations[l - 1][i - 1] +
                            self.hrules[l - 1][o][i][3])
                    self.weights[l - 1][o][i] += dw


class RNN(HNN):
    def __init__(self, nodes, prune_ratio, eta, seed, random):
        super().__init__(nodes, eta)
        self.nodes = nodes[:]
        self.tns = sum(self.nodes)
        self.prune_ratio = prune_ratio
        self.eta = eta
        self.nweights = nodes[0] * nodes[-1] + nodes[0] *nodes[1]+ (nodes[1]**2 - nodes[1])+ nodes[1]*nodes[2]

        self.weights = np.zeros((self.tns, self.tns), dtype=float)
        self.hrules = np.array((self.tns, self.tns, 4), dtype=float)
        self.activations = np.zeros(self.tns)
        self.prune_flag = False
        self.pruned_synapses = set()
        self.top_sort = []
        self.cycle_history = []
        self.random = random
        self.rng = np.random.default_rng(seed)

    def set_hrules(self, hrules):
        assert len(hrules) == self.nweights * 4
        self.hrules = np.zeros((self.tns, self.tns, 4))
        # print("############# "+str(len(hrules)))
        c = 0
        #set input to other nodes rules
        for i in range(self.nodes[0]):
            for o in range(self.nodes[0], self.tns):
                self.hrules[i, o, 0] = hrules[c]
                self.hrules[i, o, 1] = hrules[c + 1]
                self.hrules[i, o, 2] = hrules[c + 2]
                self.hrules[i, o, 3] = hrules[c + 3]
                c += 4
        #set Hidden to Hidden nodes rules
        for i in range(self.nodes[0], self.nodes[0]+self.nodes[1]):
            for o in range(self.nodes[0], self.nodes[0] + self.nodes[1]):
                if not i == o:
                    self.hrules[i, o, 0] = hrules[c]
                    self.hrules[i, o, 1] = hrules[c + 1]
                    self.hrules[i, o, 2] = hrules[c + 2]
                    self.hrules[i, o, 3] = hrules[c + 3]
                    c += 4
        #set H to output nodes rules
        for i in range(self.nodes[0], self.nodes[0] + self.nodes[1]):
            for o in range(self.nodes[0] + self.nodes[1], self.tns):
                if not i == o:
                    self.hrules[i, o, 0] = hrules[c]
                    self.hrules[i, o, 1] = hrules[c + 1]
                    self.hrules[i, o, 2] = hrules[c + 2]
                    self.hrules[i, o, 3] = hrules[c + 3]

    def sanitize_weights(self):
        # clean impossible connections
        #   outputs to other nodes
        for o in range(self.nodes[0] + self.nodes[1], self.tns):
            for n in range(self.tns):
                self.weights[o, n] = 0
        #   incoming edges to input
        for n in range(self.tns):
            for i in range(self.nodes[0]):
                self.weights[n, i] = 0

    def simmetric_weights(self):
        self.weights = np.triu(self.weights)
        self.weights = self.weights + self.weights.T - np.diag(np.diag(self.weights))

    def activate(self, inputs):
        if not self.prune_flag:
            for i in range(len(inputs)):
                self.activations[i] = np.tanh(inputs[i])
            actv = list(range(self.nodes[0], self.tns - self.nodes[-1]))
            if self.random:
                self.rng.shuffle(actv)
            for i in actv:
                self.activations[i] = np.tanh(np.dot(self.activations, self.weights[:,i]))
            for i in range(self.nodes[0] + self.nodes[1], self.tns):  # fires output
                self.activations[i] = np.tanh(np.dot(self.activations, self.weights[:,i]))
        else:
            # neurons can appear more than one time in the cycle history, as a node can be in more than one cycle.
            # However, if it fired it don't have to fire again
            fired_neurons = set()
            for n in self.top_sort:  # top sort contains both input and outputs
                if n < self.tns and not n in fired_neurons:  # it is a true node
                    if n >= self.nodes[0]:  # is not an input
                        self.activations[n] = np.tanh(np.dot(self.activations, self.weights[:, n]))
                        fired_neurons.add(n)
                    else:  # n is an input
                        self.activations[n] = np.tanh(inputs[n])
                        fired_neurons.add(n)
                else:  # n is a fake node, it contains a cycle, check the history
                    fns = [n]
                    while len(fns) > 0:
                        fn = fns.pop(0)
                        for l in range(len(self.cycle_history)):
                            if fn in self.cycle_history[l].keys():
                                cns = self.cycle_history[l][fn]
                                self.rng.shuffle(cns)
                                for cn in cns:
                                    if cn >= self.tns:  # cn is a fake node
                                        fns.append(cn)
                                    else:  # cn is a true node, already shuffled, so it fires
                                        if not n in fired_neurons:  # it fires only if it has not yet fired
                                            self.activations[cn] = np.tanh(np.dot(self.activations, self.weights[:, cn]))
        return self.activations[self.nodes[0] + self.nodes[1]:self.tns]

    def get_weightsToPrune(self):
        ws = list()
        for i in range(self.tns):
            st = i if i > self.nodes[0] else self.nodes[0]
            for o in range(st + 1, self.tns):
                ws.append(np.abs(self.weights[i, o]))
        return ws

    def prune_weights(self):
        preprunew = self.weights.copy()
        wsToThr = self.get_weightsToPrune()
        # upw = np.abs(np.triu(wsToThr).flat)
        thr = np.percentile(wsToThr, self.prune_ratio)
        # print(thr)
        self.prune_flag = True

        for i in range(self.tns):
            for j in range(i, self.tns):
                if np.abs(self.weights[i, j]) <= thr:
                    self.weights[i, j] = 0.
                    self.pruned_synapses.add((i, j))

        self.sanitize_weights()
        dag, hc = self.dag()
        top_sort = nx.topological_sort(dag)
        self.top_sort = list(top_sort)
        self.cycle_history = hc

    def update_weights(self):
        for i in range(self.nodes[0]):
            for o in range(self.nodes[0], self.tns):
                if i == o or (i, o) in self.pruned_synapses:
                    self.weights[i, o] = 0.
                else:
                    self.weights[i, o] = self.weights[i, o] + self.eta * (
                            self.hrules[i, o, 0] * self.activations[i] +
                            self.hrules[i, o, 1] * self.activations[o] +
                            self.hrules[i, o, 2] * self.activations[i] * self.activations[o] +
                            self.hrules[i, o, 3])

        for i in range(self.nodes[0], self.nodes[0]+self.nodes[1]):
            for o in range(self.nodes[0], self.nodes[0] + self.nodes[1]):
                if i == o or (i, o) in self.pruned_synapses:
                    self.weights[i, o] = 0.
                else:
                    self.weights[i, o] = self.weights[i, o] + self.eta * (
                            self.hrules[i, o, 0] * self.activations[i] +
                            self.hrules[i, o, 1] * self.activations[o] +
                            self.hrules[i, o, 2] * self.activations[i] * self.activations[o] +
                            self.hrules[i, o, 3])
        for i in range(self.nodes[0], self.nodes[0] + self.nodes[1]):
            for o in range(self.nodes[0] + self.nodes[1], self.tns):
                if i == o or (i, o) in self.pruned_synapses:
                    self.weights[i, o] = 0.
                else:
                    self.weights[i, o] = self.weights[i, o] + self.eta * (
                            self.hrules[i, o, 0] * self.activations[i] +
                            self.hrules[i, o, 1] * self.activations[o] +
                            self.hrules[i, o, 2] * self.activations[i] * self.activations[o] +
                            self.hrules[i, o, 3])

        self.sanitize_weights()

    def cycles(self):
        adj_matrix = np.array(self.weights)
        graph = nx.from_numpy_matrix(adj_matrix, create_using=nx.DiGraph)
        return nx.simple_cycles(graph)

    def _add_nodes(self, dag, old_dag, cycles, history_of_cycles, offset):
        max_fn_id = 0
        cycling_nodes = set()
        old_nodes = list(old_dag)
        for cycle in cycles:
            for node in cycle:
                cycling_nodes.add(node)
        # inputs have no cycles
        for i in range(self.nodes[0]):
            dag.add_node(i)
        # outputs neither
        for o in range(self.nodes[0] + self.nodes[1], self.tns):
            dag.add_node(o)

        # inner nodes can have cycle
        for n in range(self.nodes[0], self.nodes[0] + self.nodes[1]):
            if n not in cycling_nodes:
                if n in old_nodes:
                    dag.add_node(n)
            else:
                fns = [i for i in range(len(cycles)) if n in cycles[i]]
                for fn in fns:
                    if not dag.has_node(self.tns + fn + offset):
                        dag.add_node(self.tns + fn + offset)
                        history_of_cycles[-1][self.tns + fn + offset] = cycles[fn][:]
                        max_fn_id += 1

        # and also fake nodes that hide cycle can have cycle
        for n in old_nodes:
            if not n in cycling_nodes:
                dag.add_node(n)
            else:
                fns = [i for i in range(len(cycles)) if n in cycles[i]]
                for fn in fns:
                    if not dag.has_node(self.tns + fn + offset):
                        dag.add_node(self.tns + fn + offset)
                        history_of_cycles[-1][self.tns + fn + offset] = cycles[fn][:]
                        max_fn_id += 1

        return dag, cycles, history_of_cycles, cycling_nodes, max_fn_id

    def _get_in_edges(self, adj_m, node):
        return [i for i in adj_m.keys() if node in adj_m[i]]

    def _get_out_edges(self, adj_m, node):
        return adj_m[node]

    def _add_edges(self, dag, adj_m, history_of_cycles, not_cycling_nodes):
        for n in not_cycling_nodes:
            inc = self._get_in_edges(adj_m, n)
            outc = self._get_out_edges(adj_m, n)
            for i in inc:
                if i in not_cycling_nodes:
                    if not i == n:
                        dag.add_edge(i, n)
                else:
                    fnid = None
                    for id in history_of_cycles[-1].keys():
                        fnid = id if i in history_of_cycles[-1][id] else fnid

                    if not fnid == n:
                        dag.add_edge(fnid, n)

            for o in outc:
                if o in not_cycling_nodes:
                    if o in not_cycling_nodes:
                        if not n == o:
                            dag.add_edge(n, o)
                    else:
                        fnid = None
                        for id in history_of_cycles[-1].keys():
                            fnid = id if o in history_of_cycles[-1][id] else fnid
                        if not n == fnid:
                            dag.add_edge(n, fnid)

        for fake_node in history_of_cycles[-1].keys():
            for n in history_of_cycles[-1][fake_node]:
                inc = self._get_in_edges(adj_m, n)
                outc = self._get_out_edges(adj_m, n)
                for i in inc:
                    if i in not_cycling_nodes:
                        if not i == fake_node:
                            dag.add_edge(i, fake_node)
                    else:
                        fnid = None
                        for id in history_of_cycles[-1].keys():
                            fnid = id if i in history_of_cycles[-1][id] else fnid
                        if not fnid == fake_node:
                            dag.add_edge(fnid, fake_node)
                for o in outc:
                    if o in not_cycling_nodes:
                        if not o == fake_node:
                            dag.add_edge(fake_node, o)
                    else:
                        fnid = None
                        for id in history_of_cycles[-1].keys():
                            fnid = id if o in history_of_cycles[-1][id] else fnid
                        if not fnid == fake_node:
                            dag.add_edge(fake_node, fnid)
        return dag

    def _merge_equivalent_cycle(self, cycles):
        merged_cycles = list()
        if len(cycles) > 0:
            merged_cycles.append(cycles[0])
            for i in range(1, len(cycles)):
                switching = []
                for j in range(len(merged_cycles)):
                    tmp = set(*[cycles[i] + merged_cycles[j]])
                    if len(tmp) != len(merged_cycles[j]):
                        if len(tmp) == len(cycles[i]):
                            switching.append(j)
                        else:
                            switching.append(-1)
                    else:
                        switching.append(-2)
                if not -2 in switching:
                    if not -1 in switching:
                        merged_cycles[max(switching)] = cycles[i]
                    else:
                        merged_cycles.append(cycles[i])

        return merged_cycles

    def nx_pos_because_it_was_too_hard_to_add_a_multipartite_from_a_list_as_it_works_for_the_bipartite(self, G):
        pos = {}
        nodes_G = list(G)
        input_space = 1.75 / self.nodes[0]
        output_space = 1.75 / self.nodes[-1]

        for i in range(self.nodes[0]):
            pos[i] = np.array([-1., i * input_space])

        c = 0
        for i in range(self.nodes[0] + self.nodes[1], self.tns):
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

    def dag(self, fold=None):
        graph = nx.from_numpy_matrix(np.array(self.weights), create_using=nx.DiGraph)
        adj_matrix = nx.to_dict_of_lists(graph)
        # plt.clf()
        cycles = self._merge_equivalent_cycle(list(nx.simple_cycles(graph)))
        history_of_cycles = list()
        dag = None
        offset = 0
        cc = 1
        # print(adj_matrix)
        # print("-------")
        while len(cycles) != 0:
            history_of_cycles.append(dict())
            dag = nx.DiGraph()

            dag, cycles, history_of_cycles, cycling_nodes, max_fn_id = self._add_nodes(dag, graph, cycles,
                                                                                       history_of_cycles,
                                                                                       offset)
            offset += max_fn_id
            not_cycling_nodes = [n for n in list(graph) if n not in cycling_nodes]
            dag = self._add_edges(dag, adj_matrix, history_of_cycles, not_cycling_nodes)
            graph = dag.copy()
            if not fold is None:
                plt.clf()
                pos = self.nxpbiwthtaamfalaiwftb(dag)
                nx.draw(dag, pos=pos, with_labels=True, font_weight='bold')
                plt.savefig(fold + "/" + str(cc) + ".png")
            cc += 1
            cycles = self._merge_equivalent_cycle(list(nx.simple_cycles(graph)))
            adj_matrix = nx.to_dict_of_lists(graph)  # nx.to_numpy_matrix(graph)
            # print(adj_matrix)
            # print("-------")
        if dag is None:
            dag = graph

        return dag, history_of_cycles


if __name__ == "__main__":
    nn = NN([8,5,4])
    nn.set_weights([0 for i in range(10000)])
    for i in range(0):
        fold = "./gtest/" + str(i)
        os.makedirs(fold, exist_ok=True)
        rnn = RNN([3, 5, 2], 60, 0.001, 0, True)
        rng = np.random.default_rng(i)
        weights = np.zeros((10, 10))
        # input
        for ii in range(8):
            for jj in range(3, 10):
                if ii != jj:
                    weights[ii, jj] = rng.random()

        plt.clf()
        rnn.weights = weights
        rnn.prune_weights()
        pregraph = nx.from_numpy_matrix(rnn.weights, create_using=nx.DiGraph)
        pos = rnn.nx_pos_because_it_was_too_hard_to_add_a_multipartite_from_a_list_as_it_works_for_the_bipartite(
            pregraph)

        nx.draw(pregraph, pos=pos, with_labels=True, font_weight='bold')
        plt.savefig(fold + "/initial.png")

        # print(rnn.weights)
        dag, hc = rnn.dag(fold)
        top_sort = nx.topological_sort(dag)
        rnn.top_sort = list(top_sort)
        rnn.cycle_history = hc

        plt.clf()
        pos = rnn.nxpbiwthtaamfalaiwftb(dag)
        nx.draw(dag, pos=pos, with_labels=True, font_weight='bold')
        plt.savefig(fold + "/final_dag.png")
        with open(fold + "/hc.txt", "wb") as f:
            pickle.dump(hc, f)
        print(rnn.top_sort)
        print(hc)
        rnn.activate([1., 1., 1.])
        print(rnn.activations)
        print("end of test " + str(i))
