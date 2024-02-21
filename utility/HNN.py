import torch
from utility.NN import NN

class HNN(NN):
    """
    Neural Network class that implements the Hebbian Learning Rule.
    """

    def __init__(self, nodes: list, eta: float, hrules=None, grad=False, init=None):
        """
        Initializes the Neural Network.

        Args:
            nodes (list): List of integers representing the number of neurons in each layer.
            eta (float): Learning rate for the Hebbian Learning Rule.
            hrules (list): List of Hebbian Learning Rules for the network. Default is None.
            grad (bool): Flag indicating whether to compute gradients during training. Default is False.
            init (str): Initialization method for the weights. Default is None.
        """
        super(HNN, self).__init__(nodes, grad=grad, init=init)

        self.hrules = []
        self.eta = eta

        if hrules is not None:
            self.set_hrules(hrules)

    def set_hrules(self, hrules: list):
        """
        Sets the Hebbian Learning Rules for the network.
        We use the A, B, C and D parameters for the Hebbian Learning Rule.

        Args:
            hrules (list): List of Hebbian Learning Rules for the network.
        """
        # assert that the number of parameters is correct
        assert len(hrules) == self.nweights * 4, "needed " + str(
            self.nweights * 4) + " received " + str(len(hrules))
        self.hrules = []
        start = 0
        for l in self.get_weights():
            # Calculate the size of the current weight layer
            size = l.size()[0] * l.size()[1] * 4 + start
            # Extract the parameters for the current weight layer
            params = torch.tensor(hrules[start:size])
            # Reshape the parameters into the desired shape
            reshaped_params = torch.reshape(params, (l.size()[0], l.size()[1], 4))
            # Append the reshaped parameters to the hrules list
            self.hrules.append(reshaped_params)
            # Update the starting index for the next weight layer
            start = size

    def set_etas(self, etas: list):
        """
        Sets the learning rates for the network.

        Args:
            etas (list): List of learning rates for the network.
        """
        start = 0
        self.eta = []
        for l in self.get_weights():
            size = l.size()[0] * l.size()[1] + start
            params = torch.tensor(etas[start:size])
            self.eta.append(torch.reshape(params, (l.size()[0], l.size()[1])))
            start = size

    def update_weights(self):
        """
        Updates the weights of the network using the Hebbian Learning Rule.

        The Hebbian Learning Rule is given by dw = pre + post + C + D
        where:
          - pre = a_i * hrule_i
          - post = b_j * hrule_j
          - C = a_i * hrule_i * b_j * hrule_j
          - D = hrule_i * hrule_j (bias)
        """

        weights = self.get_weights()

        for i in range(len(weights)):
            # get the activations of the neurons of the ith layer
            presinaptic = self.activations[i].to(self.device)

            # get the activations of the neurons of the (i+1)th layer
            # keep the size of the activations as (l0, 1)
            postsinaptic = torch.reshape(
                self.activations[i + 1].to(self.device),
                self.activations[i + 1].size()[0], 
                1
            )

            # get the Hebbian Learning Rule for the ith layer
            hrule_i = self.hrules[i].to(self.device)
            hrule_j = self.hrules[i + 1].to(self.device)

            # presinaptic value
            pre = hrule_i[:, :, 0] * presinaptic
            # postsinaptic value
            post = hrule_j[:, :, 1] * postsinaptic
            # presinaptic * postsinaptic
            C_i = presinaptic * hrule_i[:, :, 2]
            C_j = postsinaptic * hrule_j[:, :, 2]
            C = C_i * C_j
            D = hrule_i[:, :, 3] * hrule_j[:, :, 3]

            # calculate the change in weights
            dw = pre + post + C + D

            # update the weights using the Hebbian Learning Rule
            weights[i] += self.eta[i] * dw

        self.set_weights(weights)
