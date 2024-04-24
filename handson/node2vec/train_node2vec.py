import time

import networkx
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import networkx as nx
from sklearn.manifold import TSNE

seed =  12345
rng = np.random.default_rng(seed)
torch.manual_seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Word2vec(torch.nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_size = 10

        self.embeddings = torch.nn.Embedding(self.vocab_size, self.embedding_size)
        self.context = torch.nn.Linear(self.embedding_size, self.vocab_size)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        ret = self.embeddings(x)
        ret = self.relu(ret)
        ret = self.context(ret)
        return ret

    def similarity(self, target: int) -> list[tuple[int, int]]:
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        w = self.embeddings.weight
        t = w[target, :].unsqueeze(0)
        result = [(idx, sim.item()) for idx, sim in enumerate(cos(w, t)) if sim > 0]
        result.sort(key=lambda a: a[1], reverse=True)
        return result[1:]


def eval_accuracy(model: Word2vec, dataloader: DataLoader):
    model.eval()
    correct = 0
    count = 0
    for center, context in dataloader:
        logit = model(center)
        batch_size, context_size = context.shape
        topk = logit.topk(context_size, dim=1).indices
        for batch_idx in range(batch_size):
            for context_idx in range(context_size):
                if topk[batch_idx, context_idx] in context[batch_idx, :]:
                    correct += 1
                count += 1
    accuracy = correct / count
    print('correct: {}, count: {}, accuracy: {}'.format(correct, count, accuracy))


class RandomWalkDataset(Dataset):
    def __init__(self, graph: networkx.Graph):
        self.graph = graph
        self.walks_per_node = 80
        self.walk_length = 10
        self.window_size = 2
        self.p = 5
        self.q = 10
        self.random_walks = []
        self.skip_grams_per_walk = self.walk_length - 2 * self.window_size + 1
        for node in self.graph.nodes:
            for _ in range(self.walks_per_node):
                self.random_walks.append(self.random_walk(node))

    def random_walk(self, start: int) -> list[int]:
        walk = [start]

        prev = None
        current = start
        for step in range(self.walk_length):
            neighbors = [node for node in self.graph.neighbors(current)]
            pi = []
            weight = 1
            for n in neighbors:
                if n == prev:
                    a = 1 / self.p
                elif self.graph.has_edge(prev, n):
                    a = 1
                else:
                    a = 1 / self.q
                pi.append(a * weight)
            pi = np.array(pi)
            prob = pi / pi.sum()
            prev = current
            current = rng.choice(neighbors, p=prob)
            walk.append(current)

        return walk

    def __len__(self):
        return len(self.random_walks) * self.skip_grams_per_walk

    def __getitem__(self, idx):
        walk_idx = idx // self.skip_grams_per_walk
        walk = self.random_walks[walk_idx]

        skip_idx = idx % self.skip_grams_per_walk + self.window_size
        center = torch.tensor(walk[skip_idx]).to(device)
        pre = walk[skip_idx - self.window_size:skip_idx]
        post = walk[skip_idx + 1:skip_idx + self.window_size + 1]
        context = torch.tensor(pre + post).to(device)

        return center, context


def train_model(graph: networkx.Graph):
    num_nodes = len(graph)
    model = Word2vec(num_nodes).to(device)

    # criterion = torch.nn.BCEWithLogitsLoss().to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    dataset = RandomWalkDataset(graph)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    print("--- optimization parameters ---")
    count = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            # print(name)
            count += param.nelement()
            # print("  nelement: {}".format(param.nelement()))
    print("total count: {}".format(count))
    print("-------------------------------")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_start = time.time()
    max_epoch = 10
    for epoch in range(max_epoch):
        model.train()
        sum_loss = 0
        for center, context in dataloader:
            logit = model(center)
            batch_size = len(center)
            target = torch.zeros(batch_size, num_nodes).to(device)
            for i in range(batch_size):
                target[i, context[i, :]] = 1

            # print('x: {}'.format(x))
            # print('logit: {}'.format(logit))
            # print('target: {}'.format(target))

            loss = criterion(logit, target).to(device)
            sum_loss += loss.detach()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("epoch {}, loss: {}".format(epoch, sum_loss))
        # eval_accuracy(model, dataloader)

    print("Trained in {} seconds".format(time.time() - train_start))
    eval_accuracy(model, dataloader)

    sim = model.similarity(0)
    print(sim)

    return model


def display_graph(graph: networkx.Graph, labels: list[int]):
    labels = []
    for node in graph.nodes:
        label = graph.nodes[node]['club']
        labels.append(1 if label == 'Officer' else 0)

    plt.figure(figsize=(12, 12), dpi=300)
    plt.axis('off')
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=0),
                     node_color=labels, node_size=800,
                     cmap='coolwarm', font_size=14, font_color='white')
    plt.show()


def main():

    # G = nx.erdos_renyi_graph(10, 0.3, seed=1, directed=False)
    G = nx.karate_club_graph()

    labels = []
    for node in G.nodes:
        label = G.nodes[node]['club']
        labels.append(1 if label == 'Officer' else 0)

    # print(random_walk(G, 0, 10))

    model = train_model(G)
    nodes_wv = model.embeddings.weight.numpy(force=True)
    labels = np.array(labels)

    tsne = TSNE(n_components=2, learning_rate='auto', init='pca', random_state=0).fit_transform(nodes_wv)

    plt.figure(figsize=(6, 6), dpi=300)
    plt.scatter(tsne[:, 0], tsne[:, 1], s=100, c=labels, cmap='coolwarm')
    plt.show()

    # display_graph(G, labels)


if __name__ == '__main__':
    main()
