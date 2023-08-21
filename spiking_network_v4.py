import numpy as np
import torch
from snntorch import spikegen
import math

def threshold_norm(threshold):
    if threshold >= 0:
        return (min(threshold, 6) + 6)
    else:
        return (max(threshold, -6) + 6)
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class LifNeuron():
    def __init__(self, threshold, beta):
        self.threshold = threshold
        self.beta = beta
        self.membrane_potential = 0
        self.spike = 0

    def activate(self, input):
        self.membrane_potential = self.beta*self.membrane_potential + input
        if self.threshold >= 0 and self.membrane_potential >= self.threshold:
            self.spike = 1
            self.membrane_potential -= self.threshold
        elif self.threshold < 0 and self.membrane_potential <= self.threshold:
            self.spike = -1
            self.membrane_potential -= self.threshold
        else:
            self.spike = 0

class SNN():
    def __init__(self, nodes, prune_ratio):
        self.neurons = [[LifNeuron(1, 0.5) for i in range(node)] for node in nodes]
        self.nthresholds = 0
        for i in range(len(nodes)):
            for j in range(nodes[i]):
                self.nthresholds += 1
        self.nbetas = self.nthresholds
        self.nweights = 0
        for i in range(len(nodes) - 1):
            self.nweights += nodes[i]*nodes[i+1] # weights
            self.nweights += nodes[1+i] # biases
        self.prune_ratio = prune_ratio
        self.weights = [[] for _ in range(len(self.neurons) - 1)]

    def activate(self, inputs):
        output = [0 for i in range(len(self.neurons[-1]))]
        for k in range(len(inputs)):
            for i in range(len(inputs[k])):
                self.neurons[0][i].activate(inputs[k][i])
            for i in range(1, len(self.neurons)): 
                for j in range(len(self.neurons[i])):
                    sum = self.weights[i - 1][j][0]
                    for k in range(0, len(self.neurons[i - 1])):
                        sum += self.neurons[i - 1][k].spike * self.weights[i - 1][j][k]
                    self.neurons[i][j].activate(sum)
            output = np.add(output, np.array([self.neurons[-1][i].spike for i in range(len(self.neurons[-1]))]))
        return output

    def set_params(self, weights):
        c = 0
        d = 0
        for i in range(1, len(self.neurons)):
            self.weights[i - 1] = [[0 for _ in range(len(self.neurons[i - 1]) + 1)] for __ in range(len(self.neurons[i]))]
            for j in range(len(self.neurons[i])):
                for k in range(len(self.neurons[i - 1]) + 1):
                    self.weights[i - 1][j][k] = weights[c]
                    c += 1
        thresholds = [weights[self.nweights+i] for i in range(0, self.nthresholds)]
        betas = [weights[self.nweights+self.nthresholds+i] for i in range(0, self.nbetas)]
        for i in range(len(self.neurons)):
            for j in range(len(self.neurons[i])):
                self.neurons[i][j].threshold = thresholds[d]
                d += 1
        d = 0
        for i in range(len(self.neurons)):
            for j in range(len(self.neurons[i])):
                self.neurons[i][j].beta = sigmoid(betas[d])
                d += 1