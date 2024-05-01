from torch_geometric.datasets import Planetoid
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import to_networkx
from torch_geometric.nn import SAGEConv
import torch

import networkx as nx
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
    print('Training nodes: {}'.format(sum(data.train_mask).item()))
    print('Evaluation nodes: {}'.format(sum(data.val_mask).item()))
    print('Test nodes: {}'.format(sum(data.test_mask).item()))
    print('Edges are directed: {}'.format(data.is_directed()))
    print('Graph has isolated nodes: {}'.format(data.has_isolated_nodes()))
    print('Graph has loops: {}'.format(data.has_self_loops()))


def plot_subgraph(data):
    train_loader = NeighborLoader(data, num_neighbors=[10, 10], batch_size=16, input_nodes=data.train_mask)

    for i, subgraph in enumerate(train_loader):
        print('subgraph {}: {}'.format(i, subgraph))

    fig = plt.figure()
    pos = range(221, 225)
    for idx, sub_data in enumerate(train_loader):
        graph = to_networkx(sub_data, to_undirected=True)
        ax = fig.add_subplot(pos[idx])
        ax.set_title('subgraph {}'.format(idx), fontsize=24)
        plt.axis('off')
        nx.draw_networkx(graph, pos=nx.spring_layout(graph, seed=0), with_labels=False, node_color=sub_data.y)
    plt.show()


def accuracy(pred, true):
    return ((pred == true).sum() / len(true)).item()


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.sage1 = SAGEConv(self.in_channels, self.hidden_channels)
        self.sage2 = SAGEConv(self.hidden_channels, self.out_channels)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x, edge_index):
        h = self.sage1(x, edge_index)
        h = self.relu(h)
        h = self.dropout(h)
        h = self.sage2(h, edge_index)
        return h

    def fit(self, data, max_epoch):
        train_loader = NeighborLoader(data, num_neighbors=[10, 10], batch_size=16, input_nodes=data.train_mask)
        num_batch = len(train_loader)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)

        self.train()
        for epoch in range(max_epoch):
            total_loss, val_loss, acc, val_acc = 0, 0, 0, 0
            for batch in train_loader:
                optimizer.zero_grad()
                out = self(batch.x, batch.edge_index)
                loss = criterion(out[batch.train_mask], batch.y[batch.train_mask])
                total_loss += loss
                acc += accuracy(out[batch.train_mask].argmax(dim=1), batch.y[batch.train_mask])
                loss.backward()
                optimizer.step()

                # validation
                val_loss += criterion(out[batch.val_mask], batch.y[batch.val_mask])
                val_acc += accuracy(out[batch.val_mask].argmax(dim=1), batch.y[batch.val_mask])
            if epoch % 20 == 0:
                print('epoch {:>3} | train loss: {:.3f} | train acc: {:>5.2f} | val loss: {:.2f} | val acc: {:.2f}'.format(
                    epoch, loss / num_batch, acc / num_batch * 100, val_loss / num_batch, val_acc / num_batch * 100
                ))

    def test(self, data):
        self.eval()
        out = self(data.x, data.edge_index)
        acc = accuracy(out.argmax(dim=1)[data.test_mask], data.y[data.test_mask])
        return acc


def main():
    dataset = Planetoid(root='../data', name='Pubmed')
    data = dataset[0]
    # print_dataset(dataset, data)
    # plot_subgraph(data)

    hidden_channels = 64
    graph_sage = GraphSAGE(dataset.num_features, hidden_channels, dataset.num_classes)
    print(graph_sage)
    graph_sage.fit(data, 200)

    acc = graph_sage.test(data)
    print('graph_sage test accuracy: {:.2f}'.format(acc * 100))


if __name__ == '__main__':
    main()
