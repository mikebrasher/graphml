import torch
import torch_geometric

import numpy as np

seed = 0
torch.manual_seed(seed)
rng = np.random.default_rng(seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def test(model, data):
    model.eval()
    z = model.encode(data.x, data.edge_index)
    auc, ap = model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)
    return auc, ap


def train(model, train_data, val_data, max_epoch):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(max_epoch):
        model.train()
        optimizer.zero_grad()
        z = model.encode(train_data.x, train_data.edge_index)
        recon_loss = model.recon_loss(z, train_data.pos_edge_label_index)
        normalized_kl_loss = model.kl_loss() / train_data.num_nodes
        loss = recon_loss + normalized_kl_loss
        loss.backward()
        optimizer.step()

        # validation
        val_auc, val_ap = test(model, val_data)
        if epoch % 50 == 0:
            print('epoch {:>3} | loss: {:.4f} | val auc: {:.4f} | val ap: {:.4f}'.format(
                epoch, loss, val_auc, val_ap))


class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = 2 * self.out_channels
        self.conv1 = torch_geometric.nn.GCNConv(self.in_channels, self.hidden_channels)
        self.conv_mu = torch_geometric.nn.GCNConv(self.hidden_channels, self.out_channels)
        self.conv_logstd = torch_geometric.nn.GCNConv(self.hidden_channels, self.out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = self.relu(h)
        mu = self.conv_mu(h, edge_index)
        logstd = self.conv_logstd(h, edge_index)
        return mu, logstd


def main():
    transform = torch_geometric.transforms.Compose([
        torch_geometric.transforms.NormalizeFeatures(),
        torch_geometric.transforms.ToDevice(device),
        torch_geometric.transforms.RandomLinkSplit(num_val=0.05, num_test=0.1,
                                                   is_undirected=True, split_labels=True,
                                                   add_negative_train_samples=False)
    ])

    dataset = torch_geometric.datasets.Planetoid(root='../data', name='Cora', transform=transform)
    train_data, val_data, test_data = dataset[0]

    model = torch_geometric.nn.VGAE(Encoder(dataset.num_features, out_channels=16)).to(device)
    train(model, train_data, val_data, max_epoch=300)

    test_auc, test_ap = test(model, test_data)
    print('test auc: {:.4f} | test ap: {:.4f}'.format(test_auc, test_ap))

    z = model.encode(test_data.x, test_data.edge_index)
    adjacency = torch.sigmoid(z @ z.T)
    print(adjacency)


if __name__ == '__main__':
    main()
