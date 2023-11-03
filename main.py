import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from network import NN

nodes = [8, 4, 2]
print("nodes: " + str(nodes))

# activations
activations = [[0 for i in range(node)] for node in nodes]
print("activations: " + str(activations))

# num weights
num_weights = sum([nodes[i] * nodes[i + 1] for i in  #3*2 + 2*2 + 2*7 = 6 + 4 + 14 = 24
                             range(len(nodes) - 1)])
print("num_weights: " + str(num_weights)) 

#weights 
weights = [[] for _ in range(len(nodes) - 1)]
print("weights: " + str(weights))

c =0
weightsC = [0 for _ in range(num_weights)]
for i in range(1, len(nodes)):
            weights[i - 1] = [[0 for _ in range(nodes[i - 1])] for __ in range(nodes[i])]
            for j in range(nodes[i]):
                for k in range(nodes[i - 1]):
                    weights[i - 1][j][k] = weightsC[c]
                    c += 1

print("set_weights: " + str(weights))


weights[0][0][0] = 1
weights[1][0][0] = 3
print("weights: " + str(weights))
# from list to matrix
matrix = np.zeros((sum(nodes), sum(nodes)))
# set inputs
for i in range(1, len(nodes)):
    for j in range(nodes[i]):
        for k in range(nodes[i - 1]):
            ix = k + sum(nodes[:i - 1])
            ox = j + (sum(nodes[:i]))
            matrix[ix][ox] = weights[i - 1][j][k]

print("matrix: \n" + str(matrix))

def nx_pos_because_it_was_too_hard_to_add_a_multipartite_from_a_list_as_it_works_for_the_bipartite(G):
    pos = {}
    nodes_G = list(G)
    input_space = 1.75 / nodes[0] # spazio verticale nodi input
    output_space = 1.75 / nodes[-1] # spazio verticale nodi output

    for i in range(nodes[0]):
        pos[i] = np.array([-1., i * input_space])

    c = 0
    for i in range(nodes[0] + nodes[1], sum(nodes)):
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

graph = nx.from_numpy_array(matrix, create_using=nx.DiGraph)

plt.clf()
pos = nx_pos_because_it_was_too_hard_to_add_a_multipartite_from_a_list_as_it_works_for_the_bipartite(graph)
nx.draw(graph, pos=pos, with_labels=True, font_weight='bold')
# print("saving")
plt.show()


    