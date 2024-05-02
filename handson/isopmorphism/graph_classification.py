import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.nn import GCNConv

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

torch.manual_seed(0)


def print_dataset(dataset):

    print('Dataset: {}'.format(dataset))
    print('------------------')
    print('Number of graphs: {}'.format(len(dataset)))
    print('Number of nodes: {}'.format(dataset[0].x.shape[0]))
    print('Number of features: {}'.format(dataset.num_features))
    print('Number of classes: {}'.format(dataset.num_classes))


def print_loader(loader, label):
    print('{} loader'.format(label))
    for i, batch in enumerate(loader):
        print(' - batch {}: {}'.format(i, batch))


def accuracy(pred, true):
    return ((pred == true).sum() / len(true)).item()


def test(model, loader):
    criterion = torch.nn.CrossEntropyLoss()

    model.eval()
    loss, acc = 0, 0
    for data in loader:
        out = model(data.x, data.edge_index, data.batch)
        loss += criterion(out, data.y) / len(loader)
        acc += accuracy(out.argmax(dim=1), data.y) / len(loader)

    return loss, acc


def train(model, train_loader, val_loader, max_epoch):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(max_epoch):
        total_loss, acc, = 0, 0
        model.train()
        for data in train_loader:
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            total_loss += loss / len(train_loader)
            acc += accuracy(out.argmax(dim=1), data.y) / len(train_loader)
            loss.backward()
            optimizer.step()

        # validation
        val_loss, val_acc = test(model, val_loader)
        if epoch % 20 == 0:
            print('epoch {:>3} | train loss: {:.3f} | train acc: {:>5.2f} | val loss: {:.2f} | val acc: {:.2f}'.format(
                epoch, loss, acc * 100, val_loss, val_acc * 100))


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.conv1 = GCNConv(self.in_channels, self.hidden_channels)
        self.conv2 = GCNConv(self.hidden_channels, self.hidden_channels)
        self.conv3 = GCNConv(self.hidden_channels, self.hidden_channels)
        self.classifier = torch.nn.Linear(self.hidden_channels, self.out_channels)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x, edge_index, batch):
        # node embeddings
        h = self.conv1(x, edge_index)
        h = self.relu(h)
        h = self.conv2(h, edge_index)
        h = self.relu(h)

        # graph level readout
        h = global_mean_pool(h, batch)
        h = self.dropout(h)

        # classifier
        h = self.classifier(h)

        return h


class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.conv1 = GINConv(torch.nn.Sequential(
            torch.nn.Linear(self.in_channels, self.hidden_channels),
            torch.nn.BatchNorm1d(self.hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_channels, self.hidden_channels),
            torch.nn.ReLU()
        ))
        self.conv2 = GINConv(torch.nn.Sequential(
            torch.nn.Linear(self.hidden_channels, self.hidden_channels),
            torch.nn.BatchNorm1d(self.hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_channels, self.hidden_channels),
            torch.nn.ReLU()
        ))
        self.conv3 = GINConv(torch.nn.Sequential(
            torch.nn.Linear(self.hidden_channels, self.hidden_channels),
            torch.nn.BatchNorm1d(self.hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_channels, self.hidden_channels),
            torch.nn.ReLU()
        ))
        num_conv = 3
        self.lin_channels = num_conv * self.hidden_channels
        self.lin1 = torch.nn.Linear(self.lin_channels, self.lin_channels)
        self.lin2 = torch.nn.Linear(self.lin_channels, self.out_channels)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x, edge_index, batch):
        # node embeddings
        h1 = self.conv1(x, edge_index)
        h2 = self.conv2(h1, edge_index)
        h3 = self.conv3(h2, edge_index)

        # graph level readout
        h1 = global_add_pool(h1, batch)
        h2 = global_add_pool(h2, batch)
        h3 = global_add_pool(h3, batch)

        # concatenate graph embeddings
        h = torch.cat((h1, h2, h3), dim=1)

        # classifier
        h = self.lin1(h)
        h = self.relu(h)
        h = self.dropout(h)
        h = self.lin2(h)

        return h


def split_dataset(dataset):
    train_idx = int(len(dataset) * 0.8)
    train_dataset = dataset[:train_idx]
    val_idx = int(len(dataset) * 0.9)
    val_dataset = dataset[train_idx:val_idx]
    test_dataset = dataset[val_idx:]

    # print('training set   = {} graphs'.format(len(train_dataset)))
    # print('validation set = {} graphs'.format(len(val_dataset)))
    # print('test set       = {} graphs'.format(len(test_dataset)))

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # print_loader(train_loader, 'train')
    # print_loader(val_loader, 'val')
    # print_loader(test_loader, 'test')

    return train_loader, val_loader, test_loader


def plot_errors(dataset, models):
    num_row = num_col = 4
    num_sub = num_row * num_col
    for m in models:
        fig, ax = plt.subplots(num_row, num_col)
        for i, data in enumerate(dataset[-num_sub:]):
            out = m(data.x, data.edge_index, data.batch)
            color = 'green' if out.argmax(dim=1) == data.y else 'red'
            ix = np.unravel_index(i, ax.shape)
            ax[ix].axis('off')
            graph = to_networkx(dataset[i], to_undirected=True)
            nx.draw_networkx(graph,
                             pos=nx.spring_layout(graph, seed=0),
                             with_labels=False,
                             node_size=10,
                             node_color=color,
                             width=0.8,
                             ax=ax[ix])
    plt.show()


def main():
    dataset = TUDataset(root='../data', name='PROTEINS').shuffle()
    # print_dataset(dataset)

    train_loader, val_loader, test_loader = split_dataset(dataset)

    in_channels = dataset.num_features
    hidden_channels = 32
    out_channels = dataset.num_classes
    max_epoch = 100

    gin = GIN(in_channels, hidden_channels, out_channels)
    train(gin, train_loader, val_loader, max_epoch=max_epoch)
    test_loss, test_acc = test(gin, test_loader)
    print('gin test loss: {:.2f} | test acc: {:.2f}%'.format(test_loss, test_acc * 100))

    gcn = GCN(in_channels, hidden_channels, out_channels)
    train(gcn, train_loader, val_loader, max_epoch=max_epoch)
    test_loss, test_acc = test(gcn, test_loader)
    print('gcn test loss: {:.2f} | test acc: {:.2f}%'.format(test_loss, test_acc * 100))

    # plot_errors(dataset, [gin, gcn])

    # create ensemble of gcn, gin models
    acc_gcn, acc_gin, acc_ens = 0, 0, 0
    for data in test_loader:
        out_gcn = gcn(data.x, data.edge_index, data.batch)
        out_gin = gin(data.x, data.edge_index, data.batch)
        out_ens = 0.5 * (out_gcn + out_gin)

        acc_gcn += accuracy(out_gcn.argmax(dim=1), data.y) / len(test_loader)
        acc_gin += accuracy(out_gin.argmax(dim=1), data.y) / len(test_loader)
        acc_ens += accuracy(out_ens.argmax(dim=1), data.y) / len(test_loader)

    print('gcn accuracy: {:.2f}%'.format(acc_gcn))
    print('gin accuracy: {:.2f}%'.format(acc_gin))
    print('ens accuracy: {:.2f}%'.format(acc_ens))


if __name__ == '__main__':
    main()
