
from es.LMMAES import LMMAES
import torch
from scipy.special import gamma
import math
import numpy as np

class LMMAEGES(LMMAES):
    """
    LMMA-EG-ES algorithm: this aglorithm is a modification of the LMMA-ES algorithm
    Low Memory Matrix Adaptation with Estimated Gradient Evolution Strategy
    where I will exploit the information of the gradient.
    You can use the real gradient or an estimated one for when the gradient is not available.
    It uses the gradient information to compute an ADAM-like update of the parameters.
    It can use the Lévy flight in the creation of the noise vectors.
    """

    def __init__(
            self,
            n,
            evaluator,
            lambda_=None,
            mu=None,
            m=None,
            sigma=None,
            device='cpu',
            scale=1.0,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            weight_decay=0,
            use_levy_flight=True,
            use_real_gradient=False,
            momentum_strategy='mom',
            starting_point=None,
    ):
        """
        Initialize the LMM-EG-ES algorithm
        :param n: number of dimensions of the problem
        :param lambda_: number of generated offsprings
        :param mu: number of selected individuals
        :param m: number of vectors that will approximate the covariance matrix
        :param sigma: learning rate
        :param device: device to use for torch
        :param beta1: beta1 parameter of ADAM
        :param beta2: beta2 parameter of ADAM
        :param epsilon: epsilon parameter of ADAM
        :param weight_decay: weight decay parameter of ADAM
        :param use_levy_flight: tells if to use Lévy flight or not
        :param use_real_gradient: tells if to use the real gradient or not
        :param momentum_strategy: tells which momentum strategy to use, 'mom' or 'adam'
        :param starting_point: starting point of the algorithm, default is a vector of size n with values from a normal distribution

        """
        super().__init__(n=n, lambda_=lambda_, mu=mu, m=m, sigma=sigma, device=device, starting_point=starting_point)

        # tells if to use Lévy flight or not
        self.loss = None
        self.old_loss = None
        self.use_levy_flight = use_levy_flight
        self.scale = scale
        self.momentum_strategy = momentum_strategy

        # evaluator
        self.evaluator = evaluator

        # tells if to use the real gradient or not
        self.use_real_gradient = use_real_gradient

        # init adam parameters
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay

        # init the momentum and velocity vector
        self.momentum = torch.zeros(self.n).float().to(device)
        self.velocity = torch.zeros(self.n).float().to(device)
        self.gradient = None
        self.v_hat_max = - torch.ones(self.n).float().to(device) * math.inf

        # if we use the real gradient, we need to keep track
        # of the torch graph to compute the gradient
        if self.use_real_gradient:
            self.y = self.y.requires_grad_(True)
            self.y.retain_grad()

        self.step = 0

    def update(self, population):
        """
        Update the parameters of the algorithm
        :param population: generated population, already evaluated
        """
        super().update(population)
        sorted_idx = self.get_sorted_idx(population)

        self.step += 1

        if self.use_real_gradient:
            self.y = self.y.detach().clone()
            self.y = self.y.requires_grad_(True)
            self.y.retain_grad()

        # calculate the gradient using the mu best individuals
        self.calculate_grad([p for i, p in enumerate(population) if i in sorted_idx])

    def create_z(self):
        """
        Create a new noise vector
        :return: the new noise vector
        """
        z = super().create_z()
        if self.gradient is None:
            return z
        else:
            # get a copy of the gradient
            gradient = self.gradient.clone().detach()

            # scale the gradient
            # gradient = gradient / gradient.norm()
            gradient = gradient * self.scale

            # apply levy flight to the gradient
            # levy flight gives the weight to the gradient
            if self.use_levy_flight:
                gradient = gradient * abs(self.levy_flight(1))

            # add the noise to the gradient
            grad_noised = gradient + z

            # normalize the gradient
            grad_noised = grad_noised / grad_noised.std()

            return grad_noised

    def calculate_grad(self, population):
        """
        Calculate the estimation gradient or the real one for the loss function
        :param population: population to use to calculate the gradient
        """
        self.old_loss = self.loss
        self.loss = self.evaluator.evaluate(self.y, need_grad=self.use_real_gradient)[0]

        if self.use_real_gradient:
            # torch automatically computes the gradient
            self.loss.backward()
            gradient = self.y.grad.detach().clone()
        elif population is not None:
            tmp = torch.zeros((self.lambda_, self.n), device=self.device)
            for i, p in enumerate(population):
                # calculate the direction from the centroid to the individual
                direction = self.y - p

                # calculate the loss difference
                loss_diff = self.loss - p.fitness.values[0]

                # calculate the gradient, penalizing for distant individuals
                tmp[i] = direction * loss_diff / (torch.norm(direction) ** 2)

            gradient = torch.mean(tmp, dim=0)

        if self.momentum_strategy == 'adam':
            # apply adam algorithm
            gradient = gradient + self.weight_decay * self.y.detach().clone()

            self.momentum = self.beta1 * self.momentum + (1 - self.beta1) * gradient
            self.velocity = self.beta2 * self.velocity + (1 - self.beta2) * gradient ** 2

            momentum_hat = self.momentum / (1 - self.beta1 ** (self.t + 1))
            velocity_hat = self.velocity / (1 - self.beta2 ** (self.t + 1))
            self.v_hat_max = torch.maximum(self.v_hat_max, velocity_hat)

            self.gradient = - momentum_hat / (torch.sqrt(self.v_hat_max) + self.epsilon)

        else:
            # apply momentum algorithm
            self.momentum = self.beta1 * self.momentum + (1 - self.beta1) * gradient
            self.gradient = -self.momentum



    def levy_flight(self, beta, scale=1.0):
        """
        Generate a random step length following a Levy distribution.
        :param beta: The tail index of the Levy distribution (typically between 1 and 2).
        :param scale: Scaling factor to control the step size.
        :return: A random step length.
        """
        sigma_u = (
            gamma(1 + beta) * np.sin(np.pi * beta / 2) 
            / (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
        ) ** (1 / beta)
        sigma_v = 1

        u = np.random.normal(0, sigma_u)
        v = np.random.normal(0, sigma_v)

        step = u / (abs(v) ** (1 / beta))

        return scale * step


