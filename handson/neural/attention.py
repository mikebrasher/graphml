from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import degree
import torch

import numpy as np
import matplotlib.pyplot as plt


def print_dataset(dataset, data):

    print('Dataset: {}'.format(dataset))
    print('------------------')
    print('Number of graphs: {}'.format(len(dataset)))
    print('Number of nodes: {}'.format(data.x.shape[0]))
    print('Number of features: {}'.format(dataset.num_features))
    print('Number of classes: {}'.format(dataset.num_classes))

    print('Graph:')
    print('------------------')
    print('Edges are directed: {}'.format(data.is_directed()))
    print('Graph has isolated nodes: {}'.format(data.has_isolated_nodes()))
    print('Graph has loops: {}'.format(data.has_self_loops()))


def plot_accuracy_degree(data, gat):
    out = gat(data.x, data.edge_index)
    d = degree(data.edge_index[0]).numpy()

    accuracies = []
    sizes = []

    max_connection = 6
    for i in range(max_connection):
        mask = np.where(d == i)[0]
        accuracies.append(accuracy(out.argmax(dim=1)[mask], data.y[mask]))
        sizes.append(len(mask))

    mask = np.where(d >= max_connection)
    accuracies.append(accuracy(out.argmax(dim=1)[mask], data.y[mask]))
    sizes.append(len(mask))

    fig, ax = plt.subplots(dpi=300)
    ax.set_xlabel('node degree')
    ax.set_ylabel('accuracy score')
    plt.bar(['0', '1', '2', '3', '4', '5', '6+'], accuracies)
    for i in range(0, max_connection + 1):
        plt.text(i, accuracies[i], '{:.2f}%'.format(accuracies[i] * 100), ha='center', color='black')
        plt.text(i, accuracies[i] / 2, sizes[i], ha='center', color='white')
    plt.show()


def accuracy(pred, true):
    return torch.sum(pred == true) / len(true)


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.heads = heads
        self.gat1 = GATv2Conv(self.in_channels, self.hidden_channels, heads=self.heads)
        self.gat2 = GATv2Conv(self.hidden_channels * self.heads, self.out_channels, heads=1)
        self.dropout = torch.nn.Dropout(0.6)
        self.elu = torch.nn.ELU()

    def forward(self, x, edge_index):
        h = self.dropout(x)
        h = self.gat1(h, edge_index)
        h = self.elu(h)
        h = self.dropout(h)
        result = self.gat2(h, edge_index)
        return result

    def fit(self, data, max_epoch):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=0.01)

        self.train()
        for epoch in range(max_epoch):
            optimizer.zero_grad()
            out = self(data.x, data.edge_index)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            acc = accuracy(out[data.train_mask].argmax(dim=1), data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            if epoch % 20 == 0:
                val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
                val_acc = accuracy(out[data.val_mask].argmax(dim=1), data.y[data.val_mask])
                print('epoch {:>3} | train loss: {:.3f} | train acc: {:>5.2f} | val loss: {:.2f} | val acc: {:.2f}'.format(
                    epoch, loss, acc * 100, val_loss, val_acc * 100
                ))

    def test(self, data):
        self.eval()
        out = self(data.x, data.edge_index)
        acc = accuracy(out.argmax(dim=1)[data.test_mask], data.y[data.test_mask])
        return acc


def simple_attention_layer():
    seed = 12345
    rng = np.random.default_rng(seed)

    # example graph with 4 nodes
    adjacency = np.array([
        [1, 1, 1, 1],
        [1, 1, 0, 0],
        [1, 0, 1, 1],
        [1, 0, 1, 1]
    ])

    num_nodes = 4
    in_channels = 3
    hidden_channels = 2
    out_channels = 1

    features = rng.uniform(-1, 1, (num_nodes, in_channels))
    weight = rng.uniform(-1, 1, (hidden_channels, in_channels))
    attention = rng.uniform(-1, 1, (out_channels, 2 * hidden_channels))

    connections = np.where(adjacency > 0)

    hidden = features @ weight.T

    # hidden vectors for source nodes of edges, W * xi
    h0 = hidden[connections[0]]
    # hidden vectors for target nodes of edges, W * xj
    h1 = hidden[connections[1]]
    # concatenate source/target vectors for linear attention, W * [xi || xj]
    g = np.concatenate([h0, h1], axis=1)

    def leaky_relu(x, alpha=0.2):
        return np.maximum(alpha * x, x)

    # scale by learnable attention weight and pass through activation
    a = attention @ g.T
    e = leaky_relu(a)

    # place un-normalized values back into matrix
    E = np.zeros(adjacency.shape)
    E[connections[0], connections[1]] = e[0]

    def softmax2D(x, axis):
        # max value across specified axis, reshaped to broadcast properly
        max_value = np.expand_dims(np.max(x, axis=axis), axis)

        # input scaled so that max value is 0 to reduce output in exp()
        scaled = x - max_value
        num = np.exp(scaled)

        # sum of exp(scaled) across specified axis
        den = np.expand_dims(np.sum(num, axis=axis), axis)
        return num / den

    # apply softmax over columns to get attention values
    alpha = softmax2D(E, 1)

    # use adjacency to sum over neighbors for each node
    # resulting H matrix is num_nodes x hidden_channels,
    # i.e. one hidden embedding per node generated using attention
    H = adjacency.T @ alpha @ hidden


def main():
    # simple_attention_layer()

    use_cora = False
    if use_cora:
        dataset = Planetoid(root='../data', name='Cora')
        hidden_channels = 32
    else:
        dataset = Planetoid(root='../data', name='CiteSeer')
        hidden_channels = 16
    data = dataset[0]

    gat = GAT(dataset.num_features, hidden_channels, dataset.num_classes)
    print(gat)

    gat.fit(data, max_epoch=100)

    acc = gat.test(data)
    print('gat test accuracy: {:.2f}'.format(acc * 100))

    plot_accuracy_degree(data, gat)


if __name__ == '__main__':
    main()
