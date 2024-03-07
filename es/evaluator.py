import torch
import numpy as np
import random
from deap import tools
from bisect import bisect_right
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

class Evaluator:
    """ Class to evaluate the individuals, this is the loss of the froward forward algorithm """

    def __init__(self, loss, device, output_size, model):
        """
        Initialize the evaluator
        :param x_pos: positive examples
        :param x_neg: negative examples
        :param output_size: output size of the layer
        """
        self.loss = loss
        self.device = device
        self.output_size = output_size
        self.model = model
        self.backup_weights = self.model.get_weights()

    def evaluate(self, individual, need_grad=False):
        """
        Evaluate the individual, the fitness is the mean of the squared distance of the positive and negative examples
        :param individual: individual to evaluate
        :return: the fitness of the individual
        """
        if need_grad:
            individual.requires_grad = True
        else:
            individual.requires_grad = False

        # backup weights in oreder to restore them after the evaluation
        self.model.set_params_flatten(individual.detach().clone().to(self.device))
        self.model.set_weights(self.backup_weights)

        inputs, targets = self.batch
        # forward pass
        _ = self.model.learn(inputs.to(self.device))
        output = self.model.forward(inputs.to(self.device))
        
        loss = self.loss(output, targets.to(self.device))

        if not need_grad:
            # clean all the gradients and graph
            self.model.zero_grad()
            loss = loss.clone().detach().requires_grad_(False)
        return loss, self.model.get_weights()

    def set_batch(self, batch):
        """
        Set the batch for the evaluation, used in batched training
        """
        self.batch = batch
        self.backup_weights = self.model.get_weights()

    def set_best_weights(self, best_weights):
        """
        Set the best weights found so far
        """
        self.model.set_weights(best_weights)
