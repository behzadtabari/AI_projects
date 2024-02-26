# Message Passing

Message Passing is a fundamental concept in Graph Neural Networks (GNNs) where nodes exchange information with their neighboring nodes in a graph to update their own representations. This process allows the model to incorporate local graph structure and propagate information throughout the entire graph.

## Description

In the provided code, there is an implementation of a simple example of message passing in a Graph Convolutional Network (GCN) using PyTorch. The GCN consists of two main components:

1. **GraphConvolutionLayer**: This class defines a single graph convolutional layer. It performs message passing by multiplying the input node features with the adjacency matrix and applies a linear transformation followed by a ReLU activation function.

2. **GCN**: This class defines a two-layer GCN model by stacking two `GraphConvolutionLayer` instances. The GCN model takes node features and the adjacency matrix of the graph as inputs and performs message passing through multiple layers to update node representations.

### Inputs

- **Node Features (x)**: A tensor representing the feature vectors of nodes in the graph. The shape of this tensor is `(batch_size, num_nodes, num_features)`, where `batch_size` is the number of samples in the batch, `num_nodes` is the number of nodes in the graph, and `num_features` is the dimensionality of the node features.

- **Adjacency Matrix (adjacency_matrix)**: A tensor representing the adjacency matrix of the graph. The shape of this tensor is `(batch_size, num_nodes, num_nodes)`, where `batch_size` is the number of samples in the batch, and `num_nodes` is the number of nodes in the graph. Each element of the adjacency matrix indicates the presence or absence of an edge between two nodes in the graph.

### Output

The output of the GCN model is the updated node representations after message passing. It is a tensor of shape `(batch_size, num_nodes, num_classes)`, where `batch_size` is the number of samples in the batch, `num_nodes` is the number of nodes in the graph, and `num_classes` is the number of classes for node classification.

## Contact

In case of questions, don't hesitate to contact [Behzad Tabari](mailto:behzad.tabari@tum.de).
