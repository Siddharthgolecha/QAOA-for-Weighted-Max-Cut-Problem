#! /usr/bin/python3
import json
import sys
import networkx as nx
import numpy as np
import pennylane as qml
from pennylane import qaoa

NODES = 6
N_LAYERS = 10


def find_max_independent_set(graph, params):
    """Find the maximum independent set of an input graph given some optimized QAOA parameters.

    The code you write for this challenge should be completely contained within this function
    between the # QHACK # comment markers. You should create a device, set up the QAOA ansatz circuit
    and measure the probabilities of that circuit using the given optimized parameters. Your next
    step will be to analyze the probabilities and determine the maximum independent set of the
    graph. Return the maximum independent set as an ordered list of nodes.

    Args:
        graph (nx.Graph): A NetworkX graph
        params (np.ndarray): Optimized QAOA parameters of shape (2, 10)

    Returns:
        list[int]: the maximum independent set, specified as a list of nodes in ascending order
    """

    max_ind_set = []

    cost_h, mixer_h = qaoa.max_independent_set(graph, constrained=True)

    wires = NODES
    depth = N_LAYERS

    def qaoa_layer(gamma, alpha):
        qaoa.cost_layer(gamma, cost_h)
        qaoa.mixer_layer(alpha, mixer_h)

    def circuit(params, **kwargs):
        qml.layer(qaoa_layer, depth, params[0], params[1])

    dev = qml.device("default.qubit", wires=wires)
    cost_function = qml.ExpvalCost(circuit, cost_h, dev)

    optimizer = qml.GradientDescentOptimizer()
    steps = 2
    for i in range(steps):
        params = optimizer.step(cost_function, params)

    @qml.qnode(dev)
    def probability_circuit(gamma, alpha):
        circuit([gamma, alpha])
        return qml.probs(wires=[i for i in range(NODES)])


    probs = probability_circuit(params[0], params[1])
    def decimalToBinary(n):
        s = [0 for i in range(wires)]
        for i in range(1,wires+1):
            s[wires-i] = n%2
            n = n//2
        return s

    max_index = np.argmax(probs)
    binary= decimalToBinary(max_index)
    for i in range(len(binary)):
        if binary[i]:
            max_ind_set.append(i)

    return max_ind_set


if __name__ == "__main__":

    # Load and process input
    graph_string = sys.stdin.read()
    graph_data = json.loads(graph_string)

    params = np.array(graph_data.pop("params"))
    graph = nx.json_graph.adjacency_graph(graph_data)

    max_independent_set = find_max_independent_set(graph, params)

    print(max_independent_set)
