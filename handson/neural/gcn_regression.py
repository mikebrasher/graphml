from torch_geometric.datasets import WikipediaNetwork
from torch_geometric.nn import GCNConv
import torch_geometric.transforms as T
from torch_geometric.utils import degree
import torch

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import mean_squared_error, mean_absolute_error


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


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.gcn1 = GCNConv(self.in_channels, self.hidden_channels * 4)
        self.gcn2 = GCNConv(self.hidden_channels * 4, self.hidden_channels * 2)
        self.gcn3 = GCNConv(self.hidden_channels * 2, self.hidden_channels)
        self.linear = torch.nn.Linear(self.hidden_channels, self.out_channels)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x, edge_index):
        result = self.gcn1(x, edge_index)
        result = self.relu(result)
        result = self.dropout(result)
        result = self.gcn2(result, edge_index)
        result = self.relu(result)
        result = self.dropout(result)
        result = self.gcn3(result, edge_index)
        result = self.relu(result)
        result = self.linear(result)
        return result

    def fit(self, data, max_epoch):
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=5e-4)

        self.train()
        for epoch in range(max_epoch):
            optimizer.zero_grad()
            out = self(data.x, data.edge_index)
            loss = criterion(out.squeeze()[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            if epoch % 20 == 0:
                val_loss = criterion(out.squeeze()[data.val_mask], data.y[data.val_mask])
                print('epoch {:>3} | train loss: {:.5f} | val loss: {:.5f}'.format(
                    epoch, loss, val_loss))

    def test(self, data):
        self.eval()
        out = self(data.x, data.edge_index)
        criterion = torch.nn.MSELoss()
        acc = criterion(out.squeeze()[data.test_mask], data.y[data.test_mask])
        return acc


def main():

    dataset = WikipediaNetwork(root='../data', name='chameleon',
                               transform=T.RandomNodeSplit(num_val=200, num_test=500))
    data = dataset[0]
    print_dataset(dataset, data)

    df = pd.read_csv('../data/wikipedia/chameleon/musae_chameleon_target.csv')
    values = np.log10(df['target'], dtype=np.float32)
    data.y = torch.tensor(values)

    plot_degree = False
    if plot_degree:
        degrees = degree(data.edge_index[0]).numpy()
        numbers = Counter(degrees)
        fig, ax = plt.subplots()
        ax.set_xlabel('node degree')
        ax.set_ylabel('number of nodes')
        plt.bar(numbers.keys(), numbers.values())
        plt.show()

    plot_target = False
    if plot_target:
        df['target'] = values
        sns.displot(df['target'], kde=True)
        plt.show()

    hidden_channels = 128
    out_channels = 1  # regression
    gcn = GCN(dataset.num_features, hidden_channels, out_channels)
    print(gcn)
    gcn.fit(data, max_epoch=200)
    loss = gcn.test(data)
    print('gnn test loss: {:.5f}'.format(loss))

    out = gcn(data.x, data.edge_index)
    pred = out.squeeze()[data.test_mask].detach().numpy()
    mse = mean_squared_error(data.y[data.test_mask], pred)
    mae = mean_absolute_error(data.y[data.test_mask], pred)

    print('=' * 43)
    print('mse = {:.4f}, rmse = {:.4f}, mae = {:.4f}'.format(mse, np.sqrt(mse), mae))
    print('=' * 43)

    plot_regression = True
    if plot_regression:
        sns.regplot(x=data.y[data.test_mask].numpy(), y=pred)
        plt.show()


if __name__ == '__main__':
    main()
