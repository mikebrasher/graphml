import torch
import torch_geometric
import scipy
import sklearn

import numpy as np

seed = 0
torch.manual_seed(seed)
rng = np.random.default_rng(seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def test(model, loader):
    model.eval()
    pred, true = [], []
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_indexx, data.batch)
        pred.append(out.view(-1).cpu())
        true.append(data.y.view(-1).cpu().to(torch.float))
    auc = sklearn.metrics.roc_auc_score(torch.cat(true), torch.cat(pred))
    ap = sklearn.metrics.average_precision(torch.cat(true), torch.cat(pred))
    return auc, ap


def train(model, train_loader, val_loader, max_epoch):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(max_epoch):
        model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out.view(-1), data.y.to(torch.float))
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * data.num_graphs
        # total_loss = total_loss / len(train_loader)

        auc, ap = test(model, val_loader)
        print('epoch {:>2} | loss: {:.4f} | val auc: {:.4f} | val ap: {:.4f}'.format(
            epoch, total_loss, auc, ap))


def seal_processing(dataset):
    data_list = []
    for edge_label_index, y in ((dataset.pos_edge_label_index, 1), (dataset.neg_edge_label_index, 0)):
        for src, dst in edge_label_index.t().tolist():
            # src, dst are target nodes labeled per main graph
            sub_nodes, sub_edge_index, mapping, _ = torch_geometric.utils.k_hop_subgraph(
                [src, dst], num_hops=2, edge_index=dataset.edge_index, relabel_nodes=True
            )
            # assign remapped node labels for subgraph
            src, dst = mapping.tolist()

            # calculate double radius node labeling (DRNL)
            # remove the target nodes (src, dst) from the subgraph
            mask1 = (sub_edge_index[0] != src) | (sub_edge_index[1] != dst)  # edge is not src -> dst
            mask2 = (sub_edge_index[0] != dst) | (sub_edge_index[1] != src)  # edge is not dst -> src
            sub_edge_index = sub_edge_index[:, mask1 & mask2]

            # compute the adjacency matrices for source and destination nodes
            src, dst = (dst, src) if src > dst else (src, dst)  # sort target nodes
            adj = torch_geometric.utils.to_scipy_sparse_matrix(sub_edge_index, num_nodes=sub_nodes.size(0)).tocsr()
            idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
            adj_wo_src = adj[idx, :][:, idx]
            idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
            adj_wo_dst = adj[idx, :][:, idx]

            # calculate distance between every node and the target nodes
            d_src = scipy.sparse.csgraph.shortest_path(adj_wo_dst, directed=False, unweighted=True, indices=src)
            d_src = np.insert(d_src, dst, 0, axis=0)
            d_src = torch.from_numpy(d_src)
            d_dst = scipy.sparse.csgraph.shortest_path(adj_wo_src, directed=False, unweighted=True, indices=dst-1)
            d_dst = np.insert(d_dst, src, 0, axis=0)
            d_dst = torch.from_numpy(d_dst)

            # calculate the node labels according to DRNL formula
            dist = d_src + d_dst
            z = 1 + torch.min(d_src, d_dst) + dist // 2 * (dist // 2 + dist % 2 - 1)
            z[src], z[dst], z[torch.isnan(z)] = 1.0, 1.0, 0.0
            z = z.to(torch.long)

            # build node information matrix from features and one-hot labels
            node_emb = dataset.x[sub_nodes]
            max_drnl = 200  # assumed limit
            node_labels = torch.nn.functional.one_hot(z, num_classes=max_drnl).to(torch.float)
            node_x = torch.cat([node_emb, node_labels], dim=1)

            # create a data object and add it to the list
            data = torch_geometric.data.Data(x=node_x, z=z, edge_index=sub_edge_index, y=y)
            data_list.append(data)

    return data_list


class DGCNN(torch.nn.Module):
    def __init__(self, in_channels, num_nodes=30):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = [32, 16, 32, 128]
        self.out_channels = 1
        self.num_nodes = num_nodes
        self.gcn1 = torch_geometric.nn.GCNConv(self.in_channels, self.hidden_channels[0])
        self.gcn2 = torch_geometric.nn.GCNConv(self.hidden_channels[0], self.hidden_channels[0])
        self.gcn3 = torch_geometric.nn.GCNConv(self.hidden_channels[0], self.hidden_channels[0])
        self.gcn4 = torch_geometric.nn.GCNConv(self.hidden_channels[0], self.out_channels)
        self.global_pool = torch_geometric.nn.SortAggregation(k=self.num_nodes)
        self.conv1 = torch.nn.Conv1d(self.out_channels, self.hidden_channels[1], kernel_size=97, stride=97)
        self.conv2 = torch.nn.Conv1d(self.hidden_channels[1], self.hidden_channels[2], kernel_size=5, stride=1)
        self.max_pool = torch.nn.MaxPool1d(2, 2)
        self.linear1 = torch.nn.Linear(352, self.hidden_channels[3])
        self.linear2 = torch.nn.Linear(self.hidden_channels[3], self.out_channels)
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x, edge_index, batch):
        # calculate node embeddings for each GCN and concatenate results
        # TODO: forward pass chokes immediately on this first GCNConv layer
        # x = [batch_length, num_feature]
        # edge_index = [2, num_edge]
        # batch = [batch_length, ]
        # complains that the batch length does not equal num_edge
        # inside add_remaining_self_loops()
        # probably a bug in the generation of seal data
        h1 = self.gcn1(x, edge_index, batch)
        h1 = self.tanh(h1)
        h2 = self.gcn2(h1, edge_index, batch)
        h2 = self.tanh(h2)
        h3 = self.gcn3(h2, edge_index, batch)
        h3 = self.tanh(h3)
        h4 = self.gcn3(h3, edge_index, batch)
        h4 = self.tanh(h4)
        h = torch.cat([h1, h2, h3, h4], dim=-1)

        # apply pooling, convolution and linear layers
        h = self.global_pool(h, batch)
        h = h.view(h.size(0), 1, h.size(-1))
        h = self.conv1(h)
        h = self.relu(h)
        h = self.max_pool(h)
        h = self.conv2(h)
        h = self.relu(h)
        h = h.view(h.size(0), -1)
        h = self.linear1(h)
        h = self.relu(h)
        h = self.dropout(h)
        h = self.linear2(h)
        h = self.sigmoid(h)
        return h


def main():
    transform = torch_geometric.transforms.RandomLinkSplit(num_val=0.05, num_test=0.1,
                                                           is_undirected=True, split_labels=True)
    dataset = torch_geometric.datasets.Planetoid(root='../data', name='Cora', transform=transform)
    train_data, val_data, test_data = dataset[0]
    print(train_data)

    train_dataset = seal_processing(train_data)
    val_dataset = seal_processing(val_data)
    test_dataset = seal_processing(test_data)

    batch_size = 3  # 32
    train_loader = torch_geometric.loader.DataLoader(train_dataset, batch_size=batch_size)
    val_loader = torch_geometric.loader.DataLoader(val_dataset, batch_size=batch_size)
    test_loader = torch_geometric.loader.DataLoader(test_dataset, batch_size=batch_size)

    model = DGCNN(train_dataset[0].num_features).to(device)
    train(model, train_loader,  val_loader, max_epoch=31)

    test_auc, test_ap = test(model, test_loader)
    print('test auc: {:.4f}, test ap: {:.4f}'.format(test_auc, test_ap))


if __name__ == '__main__':
    main()
