from network import NN

class HostSingleNN(NN):
    """
    A neural network (host) whose weights are determined by another neural network (guest).

    Attributes
    ----------
    nodes : list
        The structure of the host network, specified by the number of nodes in each layer.
    guestNodes : list
        The structure of the guest network, indicating the nodes in each layer.
    inputType : str
        Specifies which inputs are used by the guest to compute weights:
        - 'ID': Uses node IDs for input.
        - 'A': Uses activations from the previous layer.
        - 'IDA': Combines node IDs and activations.
        - 'IDAL': Combines node IDs, activations, and layer index.
        - 'IDL': Combines node IDs and layer index.
    updateType : str
        Determines how host weights are updated:
        - 'W': Replaces existing weights.
        - 'deltaW': Adds computed values to existing weights.
    """
    def __init__(self, nodes: list, guestNodes: list, inputType: str, updateType: str):
        super().__init__(nodes)
        self.guestNodes = guestNodes

        self.guestnWeights = sum([self.guestNodes[i] * self.guestNodes[i + 1] for i in
                             range(len(self.guestNodes) - 1)])

        self.inputType = inputType
        self.updateType = updateType

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

        # set all the weights to zero, as we are not computing the delta
        if self.updateType == "W":
          self.init_weights()

        self.compute_host_weights()

    def compute_host_weights(self):                                     # [2, 3, 1] example
      # foreach layer, input excluded
        for i in range(1, len(self.nodes)):                             #[1, 2] layers
            # for each node in layer i                                  #              1        2       layer i
            for j in range(self.nodes[i]):                              #[3, 1] -> [0, 1, 2], [0]       j
                 # for each node in layer i - 1                         #              0        1       layer i-1
                for k in range(self.nodes[i - 1]):                      #[2, 3] ->   [0, 1],  [0, 1, 2] k

                    input = []

                    # compute nodes id
                    if(i-1 > 0):
                      idIn = sum(self.nodes[:i-1]) + k
                    else:
                      idIn = k

                    idOut = sum(self.nodes[:i]) + j

                    # change input according to the input type
                    match self.inputType:
                      case "ID":
                        input = [idIn, idOut]
                      case "A":
                        input = [self.activations[i-1][k]]
                      case "IDA":
                        input = [idIn, idOut, self.activations[i-1][k]]
                      case "IDAL":
                        input = [idIn, idOut, i, self.activations[i-1][k]]
                      case "IDL":
                        input = [idIn, idOut, i]

                    # add bias
                    input.append(1)

                    # set the weight as the output of the guest
                    # if update type is W, replace the weight with the new value, otherwise add the new value
                    match self.updateType:
                      case "W":
                        self.weights[i - 1][j][k] = self.guest.activate(input)[0] # compute W
                      case "deltaW":
                        self.weights[i - 1][j][k] += self.guest.activate(input)[0] # compute deltaW



# the guests are the NNs deciding the weight of the host
class HostMultipleNN(NN):
    """
    A neural network (host) whose weights are determined by multiple guest neural networks, one per weight.

    Attributes
    ----------
    nodes : list
        The structure of the host network, specified by the number of nodes in each layer.
    guestNodes : list
        The structure of each guest network, indicating the nodes in each layer.
    inputType : str
        Specifies which inputs are used by the guest to compute weights:
        - 'ID': Uses node IDs for input.
        - 'A': Uses activations from the previous layer.
        - 'IDA': Combines node IDs and activations.
        - 'IDL': Combines node IDs and layer index.
    updateType : str
        Determines how host weights are updated:
        - 'W': Replaces existing weights.
        - 'deltaW': Adds computed values to existing weights.

    """
    def __init__(self, nodes: list, guestNodes: list, inputType: str, updateType: str):
        super().__init__(nodes)
        self.guestNodes = guestNodes

        self.guestnWeights = sum([self.guestNodes[i] * self.guestNodes[i + 1] for i in
                             range(len(self.guestNodes) - 1)])

        # set guest input type
        self.inputType = inputType

        # set weights update type
        self.updateType = updateType

        # set an empty list for each connection between layers of the hostNN
        self.guests = [[] for _ in range(len(self.nodes) - 1)]

        # init a guest NN for each weighte
        for i in range(1, len(self.nodes)):
            # initialize the guest NNs from layer i-1 to layer i
            self.guests[i - 1] = [[NN(self.guestNodes, "guest") for _ in range(self.nodes[i - 1])] for __ in range(self.nodes[i])]

        self.init_weights()

    def set_weights(self, weights):
        self.set_guest_weights(weights)
        self.compute_host_weights()

    def init_weights(self):
      for i in range(1, len(self.nodes)):
            # initialize the weights from layer i-1 to layer i to 0
            self.weights[i - 1] = [[0 for _ in range(self.nodes[i - 1])] for __ in range(self.nodes[i])]


    # activate each guest to compute the weights of the host
    def compute_host_weights(self):

      # foreach layer, input excluded
      for i in range(1, len(self.nodes)):
          # for each node in layer i
          for j in range(self.nodes[i]):
                # for each node in layer i - 1
              for k in range(self.nodes[i - 1]):
                  # set the weight as the output of the guest

                  input = []

                  # compute nodes id
                  if(i-1 > 0):
                    idIn = sum(self.nodes[:i-1]) + k
                  else:
                    idIn = k

                  idOut = sum(self.nodes[:i]) + j

                  # change input according to the input type
                  match self.inputType:
                    case "ID":
                      input = [idIn, idOut]
                    case "A":
                      input = [self.activations[i-1][k]]
                    case "IDA":
                      input = [idIn, idOut, self.activations[i-1][k]]
                    case "IDL":
                        input = [idIn, idOut, i]

                  # add bias
                  input.append(1)

                  # set the weight as the output of the guest
                  # if update type is W, replace the weight with the new value, otherwise add the new value
                  match self.updateType:
                    case "W":
                      self.weights[i - 1][j][k] = self.guests[i - 1][j][k].activate(input)[0]
                    case "deltaW":
                      self.weights[i - 1][j][k] += self.guests[i - 1][j][k].activate(input)[0]

      return self.weights

    def set_guest_weights(self, weights):
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
