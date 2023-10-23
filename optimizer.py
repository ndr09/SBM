from deap import base, creator, tools, algorithms
import math
import torch
import numpy as np
import random
from scipy.special import gamma


# Vanilla LMMAES algorithm
class LMMAES(object):
    """
    LMMAES algorithm: this algorithm implements the LMMAES algorithm
    Low Memory Matrix Adaptation Evolution Strategy proposed in the paper:
    "Large Scale Black-Box Optimization by Limited-Memory Matrix Adaptation"
    Here is the link: https://ieeexplore.ieee.org/document/8410043
    """

    def __init__(
            self,
            n,
            lambda_=None,
            mu=None,
            m=None,
            sigma=None,
            device='cpu',
    ):
        """
        Initialize the LMMAES algorithm
        :param n: number of dimensions of the problem
        :param lambda_: number of generated offsprings
        :param mu: number of selected individuals
        :param m: number of vectors that will approximate the covariance matrix
        :param sigma: learning rate
        :param device: device to use for torch
        """
        # device to use
        self.device = device

        # number of parameters
        self.n = n

        # number of generated offsprings
        # default: 4 + 3*ln(n)
        self.lambda_ = lambda_ if lambda_ is not None else 4 + int(math.floor(3 * math.log(n)))

        # number of selected individuals
        # default: lambda/2
        self.mu = mu if mu is not None else int(math.floor(self.lambda_ / 2))

        # weight vector initialization assigned to each selected individual
        # default: log(mu+1/2)-log(i) for i=1,...,mu
        denominator = sum([math.log(self.mu + 1 / 2) - math.log(j + 1) for j in range(self.mu)])
        self.w = torch.tensor([(math.log(self.mu + 1 / 2) - math.log(i + 1)) / denominator for i in range(self.mu)]).to(
            self.device)

        # mu_w vector initialization
        # weight assigned to all selected individual
        self.mu_w = 1 / torch.sum(self.w ** 2)

        # m parameter initialization -> default 4 + 3*ln(n)
        # number of vectors that will approximate the covariance matrix
        self.m = 4 + int(math.floor(3 * math.log(n))) if m is None else m

        # c_sigma initialization -> default 2*lambda/n
        # parameter for the Cumulative Step Size Adaptation
        # controls the learning rate of the step size adaptation
        self.c_sigma = (self.lambda_ * 2) / self.n

        # c_d initialization
        # it is a weight vector exponentially decaying
        # to appy on every vector approximating the matrix M
        self.c_d = torch.tensor([1 / ((1.5 ** i) * self.n) for i in range(self.m)]).to(self.device)

        # c_c initialization
        self.c_c = torch.tensor([self.lambda_ / ((4 ** i) * self.n) for i in range(self.m)]).to(self.device)

        # init centroid vector
        # init to a zero vector
        self.y = torch.randn(self.n).float().to(device)

        # init sigma, this is my global learning rate
        self.sigma = sigma if sigma is not None else 0.1

        # init the evolution path vector p_sigma
        # it is an exponentially fading record of recent most successful steps
        self.p_sigma = torch.zeros(self.n).float().to(self.device)

        # init the vector esimating the covariance matrix
        self.M = torch.zeros((self.m, self.n)).float().to(self.device)

        # init vectors containing the offspring's direction vectors
        self.d = torch.zeros((self.lambda_, self.n)).float().to(self.device)

        # init vectors containing the offspring's randomness vectors
        self.z = torch.zeros((self.lambda_, self.n)).float().to(self.device)

        # init the number of iterations
        self.t = 0
        self.population = []

    def ask(self):
        """
        Generate a new population of individuals
        :return: the new population
        """
        # z are lambda samples from a normal distribution
        self.population = []
        for i in range(self.lambda_):
            self.z[i] = self.create_z()
            # direction vector -> initialized as random
            self.d[i] = self.z[i].clone()
            # direction vector is updated with the previous m directions
            for j in range(min(self.t, self.m)):
                # if d and M has the same direction, similarity is high,
                # it means that the direction is good, and we can use
                # it to update the direction vector by that factor
                similarity = (self.M[j] @ self.d[i])
                self.d[i] = (1 - self.c_d[j]) * self.d[i] + self.c_d[j] * self.M[j] * similarity

            # creating the individual
            # d[i] is now the mutation given by N(0, C) where C is the covariance matrix
            ind = (self.y + self.sigma * self.d[i]).detach()

            self.population.append(ind.to(self.device))
        return self.population[:]

    def get_sorted_idx(self, fitness):
        """
        Get the ordered list of the indexes of the individuals, ordered by fitness
        :param population: population of individuals
        :return: the ordered list of the indexes of the individuals, ordered by fitness
        """
        # get the ordered list of the indexes of the mu best individuals
        sorted_idx = [i for _, i in sorted(zip(fitness, range(len(fitness))), reverse=True)][0:self.mu]
        return sorted_idx

    def tell(self,  fitness):
        """
        Update the parameters of the algorithm
        :param fitness: fitness of the individual in the same order
        :param population: generated population, already evaluated
        """

        # get the ordered list of the indexes of the mu best individuals
        sorted_idx = self.get_sorted_idx(fitness)
        # calculate the weighted sum of the mu best individuals
        weighted_d = torch.zeros((self.mu, self.n)).float().to(self.device)
        weighted_z = torch.zeros((self.mu, self.n)).float().to(self.device)

        j = 0
        for i in sorted_idx:
            weighted_d[j] = self.w[j] * self.d[i]
            weighted_z[j] = self.w[j] * self.z[i]
            j += 1

        # update the evolution path of the best solutions
        self.p_sigma = (1 - self.c_sigma) * self.p_sigma \
                       + torch.sqrt(self.mu_w * self.c_sigma * (2 - self.c_sigma)) \
                       * torch.sum(weighted_z, dim=0)

        # update the support vectors for the covariacne matrix
        for i in range(self.m):
            self.M[i] = (1 - self.c_c[i]) * self.M[i] \
                        + torch.sqrt(self.mu_w * self.c_c[i] * (2 - self.c_c[i])) \
                        * torch.sum(weighted_z, dim=0)

        # update sigma
        self.sigma = self.sigma * torch.exp(((torch.norm(self.p_sigma) ** 2 / self.n) - 1) * self.c_sigma / 2)

        # calculate new centroid
        self.y = self.y + self.sigma * torch.sum(weighted_d, dim=0)

        # update the number of iterations
        self.t += 1

    def create_z(self):
        """
        Create a new noise vector
        :return: the noise vector
        """
        return torch.randn(self.n).float().to(self.device)