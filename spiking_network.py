import numpy as np

class LifNeuron():
    def __init__(self, threshold, beta):
        self.threshold = threshold
        self.beta = beta
        self.membrane_potential = 0
        self.spike = 0

    def activate(self, input):
        self.membrane_potential = self.beta*self.membrane_potential + input - self.spike
        if self.membrane_potential >= 1:
            self.spike = 1
        else:
            self.spike = 0

class SNN():
    def __init__(self, nodes, prune_ratio, threshold, beta):
        self.threshold = threshold
        self.beta = beta
        self.neurons = [[LifNeuron(self.threshold, self.beta) for i in range(node)] for node in nodes]
        self.nweights = 0
        for i in range(len(self.neurons) - 1):
            self.nweights += self.neurons[i]*self.neurons[i+1]
        self.prune_ratio = prune_ratio
        self.weights = [[] for _ in range(len(self.neurons) - 1)]

    def activate(self, inputs):
        for i in range(self.neurons[0]):
            self.neurons[0][i].activate(inputs[i])
        for i in range(1, len(self.neurons)):
            for j in range(self.neurons[i]):
                sum = self.weights[i - 1][j][0]
                for k in range(1, self.neurons[i - 1]):
                    sum += self.neurons[i - 1][k - 1].spike * self.weights[i - 1][j][k]
                self.neurons[i][j].activate(sum)
        return np.array([self.neurons[-1][i].spike for i in range(self.neurons[-1])])

    def set_weights(self, weights):
        c = 0
        for i in range(1, len(self.nodes)):
            self.weights[i - 1] = [[0 for _ in range(self.nodes[i - 1])] for __ in range(self.nodes[i])]
            for j in range(self.nodes[i]):
                for k in range(self.nodes[i - 1]):
                    self.weights[i - 1][j][k] = weights[c]
                    c += 1