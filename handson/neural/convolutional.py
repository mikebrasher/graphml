from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import FacebookPagePage
from torch_geometric.utils import to_dense_adj
from torch_geometric.utils import degree
from torch_geometric.nn import GCNConv
import torch
from collections import Counter
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


def accuracy(pred, true):
    return torch.sum(pred == true) / len(true)


class MultilayerPerceptron(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.network = torch.nn.Sequential(
            torch.nn.Linear(self.in_channels, self.hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_channels, self.out_channels)
        )
        # self.log_softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, x):
        result = self.network(x)
        # result = self.log_softmax(result)
        return result

    def fit(self, data, max_epoch):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=5e-4)

        self.train()
        for epoch in range(max_epoch):
            optimizer.zero_grad()
            out = self(data.x)
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
        out = self(data.x)
        acc = accuracy(out.argmax(dim=1)[data.test_mask], data.y[data.test_mask])
        return acc


class VanillaGNNLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.linear = torch.nn.Linear(self.in_channels, self.out_channels, bias=False)

    def forward(self, x, adjacency):
        result = self.linear(x)
        result = torch.sparse.mm(adjacency, result)
        return result


class VanillaGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.gnn1 = VanillaGNNLayer(self.in_channels, self.hidden_channels)
        self.relu = torch.nn.ReLU()
        self.gnn2 = VanillaGNNLayer(self.hidden_channels, self.out_channels)

    def forward(self, x, adjacency):
        result = self.gnn1(x, adjacency)
        result = self.relu(result)
        result = self.gnn2(result,  adjacency)
        return result

    def fit(self, data, max_epoch):
        adjacency = to_dense_adj(data.edge_index)[0]
        adjacency += torch.eye(len(adjacency))

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=5e-4)

        self.train()
        for epoch in range(max_epoch):
            optimizer.zero_grad()
            out = self(data.x, adjacency)
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
        adjacency = to_dense_adj(data.edge_index)[0]
        adjacency += torch.eye(len(adjacency))

        self.eval()
        out = self(data.x, adjacency)
        acc = accuracy(out.argmax(dim=1)[data.test_mask], data.y[data.test_mask])
        return acc


# implementation seems mostly correct compared to GCNConv, but slower due to adjacency recalculation
# should probably add bias after left multiplication by sqrt(D) * A * sqrt(D)
class MyGCNLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.linear = torch.nn.Linear(self.in_channels, self.out_channels, bias=False)

    def forward(self, x, edge_index):
        adjacency = to_dense_adj(edge_index)[0]
        adjacency += torch.eye(len(adjacency))
        d = adjacency.sum(dim=0)
        isd = torch.diag(1 / d)
        dad = isd @ adjacency @ isd
        result = self.linear(x)
        result = torch.sparse.mm(dad, result)
        return result


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.gcn1 = GCNConv(self.in_channels, self.hidden_channels)
        self.gcn2 = GCNConv(self.hidden_channels, self.out_channels)
        # self.gcn1 = MyGCNLayer(self.in_channels, self.hidden_channels)
        # self.gcn2 = MyGCNLayer(self.hidden_channels, self.out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self, x, edge_index):
        result = self.gcn1(x, edge_index)
        result = self.relu(result)
        result = self.gcn2(result,  edge_index)
        return result

    def fit(self, data, max_epoch):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=5e-4)

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


def main():

    use_cora = True
    if use_cora:
        dataset = Planetoid(root='../data', name='Cora')
        data = dataset[0]
    else:
        dataset = FacebookPagePage(root='../data')
        data = dataset[0]
        data.train_mask = range(18000)
        data.val_mask = range(18001, 20000)
        data.test_mask = range(20001, 22470)

    # print_dataset(dataset, data)

    plot_degree = False
    if plot_degree:
        degrees = degree(data.edge_index[0]).numpy()
        numbers = Counter(degrees)
        fig, ax = plt.subplots()
        ax.set_xlabel('node degree')
        ax.set_ylabel('number of nodes')
        plt.bar(numbers.keys(), numbers.values())
        plt.show()

    adjacency = to_dense_adj(data.edge_index)[0]
    adjacency += torch.eye(len(adjacency))

    hidden_channels = 16

    mlp = MultilayerPerceptron(dataset.num_features, hidden_channels, dataset.num_classes)
    mlp.fit(data, max_epoch=100)
    acc = mlp.test(data)
    print('mlp test accuracy: {:.2f}'.format(acc * 100))

    gnn = VanillaGNN(dataset.num_features, hidden_channels, dataset.num_classes)
    gnn.fit(data, max_epoch=100)
    acc = gnn.test(data)
    print('gnn test accuracy: {:.2f}'.format(acc * 100))

    gcn = GCN(dataset.num_features, hidden_channels, dataset.num_classes)
    gcn.fit(data, max_epoch=100)
    acc = gcn.test(data)
    print('gnn test accuracy: {:.2f}'.format(acc * 100))


if __name__ == '__main__':
    main()
