from network import NN, HNN

class HNNHostSingleNN(HNN):
    """
    Host-Neural Network (HNN) class that implements a single neural network (NN) as a guest network.
    This class extends the HNN class and adds functionality for integrating a guest neural network.

    Attributes:
        nodes : list
            List of integers representing the number of nodes in each layer of the host network.
        guestNodes : list
            List of integers representing the number of nodes in each layer of the guest network.
        inputType : str
            Specifies which inputs are used by the guest to compute H-rules:
            - 'ID': Uses node IDs for input.
            - 'A': Uses activations from the previous layer.
            - 'IDA': Combines node IDs and activations.
            - 'IDAL': Combines node IDs, activations, and layer index.
            - 'IDL': Combines node IDs and layer index.
    """

    def __init__(self, nodes: list, guestNodes: list, inputType: str):
        super().__init__(nodes)
        self.guestNodes = guestNodes

        self.guestnWeights = sum([self.guestNodes[i] * self.guestNodes[i + 1] for i in
                             range(len(self.guestNodes) - 1)])

        self.inputType = inputType

        # init a guestNN
        self.guest = NN(self.guestNodes, "guest")

        self.init_weights()

    # initialize all the weights to zero
    def init_weights(self):
        # foreach layer, input excluded
        for i in range(1, len(self.nodes)):
            # initialize the weights from layer i-1 to layer i to 0
            self.weights[i - 1] = [[0 for _ in range(self.nodes[i - 1])] for __ in range(self.nodes[i])]

    # set the weights of the guest and activate them to get the host's weights
    def set_guest_weights(self, weights=None):
        # set the weights of the NN
        self.guest.set_weights(weights)
        self.compute_h_rules()

    def compute_h_rules(self):
      self.hrules = [[] for _ in range(len(self.nodes) - 1)]
      # initialize a rule for each weight

      # foreach layer, input excluded
      for i in range(1, len(self.nodes)):
          self.hrules[i - 1] = [[0 for a in range(self.nodes[i - 1])] for b in range(self.nodes[i])]
          # for each node in layer i
          for j in range(self.nodes[i]):
                # for each node in layer i - 1
              for k in range(self.nodes[i - 1]):

                  input = []

                  if(i-1 > 0):
                    idIn = sum(self.nodes[:i-1]) + k
                  else:
                    idIn = k

                  idOut = sum(self.nodes[:i]) + j

                  match self.inputType:
                    case "ID":
                      input = [idIn, idOut]
                    case "A":
                      input = [self.activations[i-1][k]]
                    case "IDA":
                      input = [idIn, idOut, self.activations[i-1][k]]
                    case "IDAL":
                      input = [idIn, idOut, self.activations[i-1][k], i-1]
                    case "IDAW":
                      input = [idIn, idOut, self.activations[i-1][k], self.weights[i - 1][j][k]]

                  # add bias
                  input.append(1)

                  hrules = self.guest.activate(input)

                  self.hrules[i - 1][j][k] = [hrules[i] for i in range(4)]
    

class HNNHostMultipleNN(HNN):
    """
    Host-Neural Network (HNN) class that implements multiple neural networks (NNs) as guest networks.

    This class extends the HNN class and adds functionality for integrating multiple guest neural networks,
    each corresponding to one parameter of the Hebbian rule.

    Attributes:
        nodes : list
            List of integers representing the number of nodes in each layer of the host network.
        guestNodes : list
            List of integers representing the number of nodes in each layer of each guest network.
        inputType : str
            Specifies which inputs are used by the guest to compute H-rules:
            - 'ID': Uses node IDs for input.
            - 'A': Uses activations from the previous layer.
            - 'IDA': Combines node IDs and activations.
            - 'IDAL': Combines node IDs, activations, and layer index.
            - 'IDL': Combines node IDs and layer index.
    """

    def __init__(self, nodes: list, guestNodes: list, inputType: str):
        super().__init__(nodes, eta=0.001)
        self.guestNodes = guestNodes

        self.guestnWeights = sum([self.guestNodes[i] * self.guestNodes[i + 1] for i in
                             range(len(self.guestNodes) - 1)])


        self.inputType = inputType

        self.guests = [NN(self.guestNodes, "guest") for _ in range(4)]

        self.init_weights()

    # initialize all the weights to zero
    def init_weights(self):
        # foreach layer, input excluded
        for i in range(1, len(self.nodes)):
            # initialize the weights from layer i-1 to layer i to 0
            self.weights[i - 1] = [[0 for _ in range(self.nodes[i - 1])] for __ in range(self.nodes[i])]

    # set the weights of the guest and activate them to get the host's weights
    def set_guest_weights(self, weights=None):
        # len(weights) = numGuestWeights * 4
        # set the weights of the NNs
        g = 1
        for i in range(len(self.guests)):
          start = self.guestnWeights * (g-1)
          end = self.guestnWeights * (g)
          self.guests[i].set_weights(weights[start:end])
          g += 1

        self.compute_h_rules()

    def compute_h_rules(self):
      self.hrules = [[] for _ in range(len(self.nodes) - 1)]
      # initialize a rule for each weight

      # foreach layer, input excluded
      for i in range(1, len(self.nodes)):
          self.hrules[i - 1] = [[0 for a in range(self.nodes[i - 1])] for b in range(self.nodes[i])]
          # for each node in layer i
          for j in range(self.nodes[i]):
                # for each node in layer i - 1
              for k in range(self.nodes[i - 1]):

                  input = []

                  if(i-1 > 0):
                    idIn = sum(self.nodes[:i-1]) + k
                  else:
                    idIn = k

                  idOut = sum(self.nodes[:i]) + j

                  match self.inputType:
                    case "ID":
                      input = [idIn, idOut]
                    case "A":
                      input = [self.activations[i-1][k]]
                    case "IDA":
                      input = [idIn, idOut, self.activations[i-1][k]]
                    case "IDAL":
                      input = [idIn, idOut, self.activations[i-1][k], i-1]
                    case "IDAW":
                      input = [idIn, idOut, self.activations[i-1][k], self.weights[i - 1][j][k]]
                    case "IDAWN":
                      input = [idIn, idOut, self.activations[i-1][k], self.weights[i - 1][j][k]]
                    case "IDN":
                      input = [idIn, idOut]

                  # add bias
                  input.append(1)

                  hrules = [0 for _ in range(len(self.guests))]
                  for h in range(len(self.guests)):
                    if(self.inputType == "IDAWN" or self.inputType == "IDN"):
                      input.append(h)
                    hrules[h] = self.guests[h].activate(input)[0]

                  self.hrules[i - 1][j][k] = [hrules[i] for i in range(4)]

