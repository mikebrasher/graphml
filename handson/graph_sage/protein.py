import torch
from torch_geometric.datasets import PPI
from torch_geometric.data import Batch
from torch_geometric.loader import NeighborLoader
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GraphSAGE
from sklearn.metrics import f1_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    train_dataset = PPI(root='../data/PPI', split='train')
    val_dataset = PPI(root='../data/PPI', split='val')
    test_dataset = PPI(root='../data/PPI', split='test')

    train_data = Batch.from_data_list(train_dataset)
    loader = NeighborLoader(train_data, batch_size=2048, shuffle=True,
                            num_neighbors=[20, 10], num_workers=2, persistent_workers=True)

    train_loader = DataLoader(train_dataset, batch_size=2)
    val_loader = DataLoader(val_dataset, batch_size=2)
    test_loader = DataLoader(test_dataset, batch_size=2)

    model = GraphSAGE(in_channels=train_dataset.num_features, hidden_channels=512,
                      num_layers=2, out_channels=train_dataset.num_classes).to(device)

    def test(loader):
        model.eval()
        data = next(iter(loader))
        out = model(data.x.to(device), data.edge_index.to(device))
        true, pred = data.y.numpy(), (out > 0).float().cpu().numpy()
        return f1_score(true, pred, average='micro') if pred.sum() > 0 else 0

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    max_epoch = 300
    for epoch in range(max_epoch):
        model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = criterion(out, data.y)
            total_loss += loss.item() * data.num_graphs
            loss.backward()
            optimizer.step()
        mean_loss = total_loss / len(train_loader.dataset)
        if epoch % 50 == 0:
            print('epoch {:>3} | train loss: {:.3f} | val f1 score: {:.4f}'.format(epoch, mean_loss, test(val_loader)))

    print('test f1 score: {:.4f}'.format(test(test_loader)))


if __name__ == '__main__':
    main()
