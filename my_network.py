import os

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pickle
from cma import CMAEvolutionStrategy as cmaes
from network import NN, HNN
from random import Random

# the guests are the NNs deciding the weight of the host
# the weights of the guests are setted by an external cmaes, with nWeights * guestnWeights vars
# each individual of the population representes an HostNN
class HostMultipleNN(NN):
    def __init__(self, nodes: list, guestNodes: list):
        super().__init__(nodes)
        self.guestNodes = guestNodes

        self.guestnWeights = sum([self.guestNodes[i] * self.guestNodes[i + 1] for i in
                             range(len(self.guestNodes) - 1)])

        # set an empty list for each connection between layers of the hostNN
        self.guests = [[] for _ in range(len(self.nodes) - 1)]

        # init a guest NN for each weight
        for i in range(1, len(self.nodes)):
            # initialize the guest NNs from layer i-1 to layer i
            self.guests[i - 1] = [[NN(self.guestNodes) for _ in range(self.nodes[i - 1])] for __ in range(self.nodes[i])]

        self.init_weights()

    # initialize all the weights to zero
    def init_weights(self):
        # foreach layer, input excluded
        for i in range(1, len(self.nodes)):
            # initialize the weights from layer i-1 to layer i to 0
            self.weights[i - 1] = [[0 for _ in range(self.nodes[i - 1])] for __ in range(self.nodes[i])]
    
    # set the weights of the guest and activate them to get the host's weights
    def set_weights(self, weights=None):  

        self.set_guests_weights(weights)
        self.init_weights()

        # foreach layer, input excluded
        for i in range(1, len(self.nodes)):        
            # for each node in layer i
            for j in range(self.nodes[i]):
                 # for each node in layer i - 1
                for k in range(self.nodes[i - 1]):
                    # set the weight as the output of the guest
                    # TODO: check activate input
                    self.weights[i - 1][j][k] = self.guests[i - 1][j][k].activate([i-1, j])[0]
    
    def set_guests_weights(self, weights):
        g = 1 # guest number
        # foreach layer, input excluded
        for i in range(1, len(self.nodes)):        
            # for each node in layer i
            for j in range(self.nodes[i]):
                 # for each node in layer i - 1
                for k in range(self.nodes[i - 1]):
                    start = self.guestnWeights * (g-1)
                    end = self.guestnWeights * (g)
                    # set the weights of the guests
                    self.guests[i - 1][j][k].set_weights(weights[start:end])
                    g += 1

class HostSingleNN(NN): 
    def __init__(self, nodes: list, guestNodes: list):
        super().__init__(nodes)
        self.guestNodes = guestNodes

        self.guestnWeights = sum([self.guestNodes[i] * self.guestNodes[i + 1] for i in
                             range(len(self.guestNodes) - 1)])

        # init a guestNN
        self.guest = NN(self.guestNodes)

        self.init_weights()

    # initialize all the weights to zero
    def init_weights(self):
        # foreach layer, input excluded
        for i in range(1, len(self.nodes)):
            # initialize the weights from layer i-1 to layer i to 0
            self.weights[i - 1] = [[0 for _ in range(self.nodes[i - 1])] for __ in range(self.nodes[i])]
    
    # set the weights of the guest and activate them to get the host's weights
    def set_weights(self, weights=None):  
        # set the weights of the NN
        self.guest.set_weights(weights)

        # set all the weights to zero, as we are not computing the delta
        self.init_weights()

        # foreach layer, input excluded
        for i in range(1, len(self.nodes)):        
            # for each node in layer i
            for j in range(self.nodes[i]):
                 # for each node in layer i - 1
                for k in range(self.nodes[i - 1]):
                    # set the weight as the output of the guest
                    # TODO: check activate input
     
                    self.weights[i - 1][j][k] = self.guest.activate([i-1, j])[0]


# an NN with multiple HNN which determine its weights
class HostMultipleHNN(NN):
    def __init__(self, nodes: list, guestNodes: list, eta: float):
        super().__init__(nodes)
        self.guestNodes = guestNodes

        self.guestnWeights = sum([self.guestNodes[i] * self.guestNodes[i + 1] for i in
                             range(len(self.guestNodes) - 1)])

        # set an empty list for each connection between layers of the hostNN
        self.guests = [[] for _ in range(len(self.nodes) - 1)]

        self.eta = eta
        # init a guest NN for each weight
        for i in range(1, len(self.nodes)):
            # initialize the guest NNs from layer i-1 to layer i
            self.guests[i - 1] = [[HNN(self.guestNodes, self.eta) for _ in range(self.nodes[i - 1])] for __ in range(self.nodes[i])]

        self.init_weights()

    # initialize all the weights to zero
    def init_weights(self):
        # foreach layer, input excluded
        for i in range(1, len(self.nodes)):
            # initialize the weights from layer i-1 to layer i to 0
            self.weights[i - 1] = [[0 for _ in range(self.nodes[i - 1])] for __ in range(self.nodes[i])]
    
    # set the weights of the guest and activate them to get the host's weights
    def set_weights(self, weights=None):  

        self.set_guests_hrules(weights)
        # reset the weights of the host to zero
        self.init_weights()

        # foreach layer, input excluded
        for i in range(1, len(self.nodes)):        
            # for each node in layer i
            for j in range(self.nodes[i]):
                 # for each node in layer i - 1
                for k in range(self.nodes[i - 1]):
                    # set the weight as the output of the guest
                    # TODO: check activate output
                    self.weights[i - 1][j][k] = self.guests[i - 1][j][k].activate([i-1, j])[0]
    
    def set_guests_hrules(self, weights):
        g = 1 # guest number
        # foreach layer, input excluded
        for i in range(1, len(self.nodes)):        
            # for each node in layer i
            for j in range(self.nodes[i]):
                 # for each node in layer i - 1
                for k in range(self.nodes[i - 1]):
                    start = self.guestnWeights * (g-1) * 4
                    end = self.guestnWeights * (g) * 4
                    # set the weights of the guests
                    self.guests[i - 1][j][k].set_hrules(weights[start:end])
                    g += 1

    def update_guests_weights(self):
        for i in range(1, len(self.nodes)):        
            # for each node in layer i
            for j in range(self.nodes[i]):
                 # for each node in layer i - 1
                for k in range(self.nodes[i - 1]):
                    # set the weights of the guests
                    self.guests[i - 1][j][k].update_weights()


class HostSingleHNN(NN): 
    def __init__(self, nodes: list, guestNodes: list):
        super().__init__(nodes)
        self.guestNodes = guestNodes

        self.guestnWeights = sum([self.guestNodes[i] * self.guestNodes[i + 1] for i in
                             range(len(self.guestNodes) - 1)])

        # init a guestNN
        self.guest = HNN(self.guestNodes, 0.01)

        self.init_weights()

    # initialize all the weights to zero
    def init_weights(self):
        # foreach layer, input excluded
        for i in range(1, len(self.nodes)):
            # initialize the weights from layer i-1 to layer i to 0
            self.weights[i - 1] = [[0 for _ in range(self.nodes[i - 1])] for __ in range(self.nodes[i])]
    
    # set the weights of the guest and activate them to get the host's weights
    def set_weights(self, weights=None):  

        self.guest.set_hrules(weights)
        # reset the weights of the host to zero
        self.init_weights()

        # foreach layer, input excluded
        for i in range(1, len(self.nodes)):        
            # for each node in layer i
            for j in range(self.nodes[i]):
                 # for each node in layer i - 1
                for k in range(self.nodes[i - 1]):
                    # set the weight as the output of the guest
                    # TODO: check activate output
                    self.weights[i - 1][j][k] = self.guest.activate([i-1, j])[0]
    
 