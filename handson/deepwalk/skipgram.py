#!/usr/bin/env python

import time
import numpy as np
import matplotlib.pyplot as plt
import torch

seed =  12345
rng = np.random.default_rng(seed)
torch.manual_seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class word2vec(torch.nn.Module):
    def __init__(self, vocab_size, context_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_size = context_size
        self.embedding_size = 20

        self.embeddings = torch.nn.Embedding(self.vocab_size, self.embedding_size)
        self.context = torch.nn.Linear(self.embedding_size, self.vocab_size)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        batch_size = x.shape[0]
        ret = self.embeddings(x)
        ret = self.relu(ret)
        ret = self.context(ret)
        return ret


def main():
    
    text = """Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nunc eu sem 
    scelerisque, dictum eros aliquam, accumsan quam. Pellentesque tempus, lorem ut 
    semper fermentum, ante turpis accumsan ex, sit amet ultricies tortor erat quis 
    nulla. Nunc consectetur ligula sit amet purus porttitor, vel tempus tortor 
    scelerisque. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices 
    posuere cubilia curae; Quisque suscipit ligula nec faucibus accumsan. Duis 
    vulputate massa sit amet viverra hendrerit. Integer maximus quis sapien id 
    convallis. Donec elementum placerat ex laoreet gravida. Praesent quis enim 
    facilisis, bibendum est nec, pharetra ex. Etiam pharetra congue justo, eget 
    imperdiet diam varius non. Mauris dolor lectus, interdum in laoreet quis, 
    faucibus vitae velit. Donec lacinia dui eget maximus cursus. Class aptent taciti
    sociosqu ad litora torquent per conubia nostra, per inceptos himenaeos. Vivamus
    tincidunt velit eget nisi ornare convallis. Pellentesque habitant morbi 
    tristique senectus et netus et malesuada fames ac turpis egestas. Donec 
    tristique ultrices tortor at accumsan.
    """.split()

    vocab = set(text)
    vocab_size = len(vocab)
    print('len(vocab) = {}'.format(len(vocab)))
    
    word_to_idx = {word: i for i, word in enumerate(vocab)}

    context_size = 2

    skipgrams = []
    for idx in range(context_size, len(text) - context_size):
        context = [text[j] for j in range(idx - context_size, idx + context_size + 1) if j != idx]
        target = text[idx]
        skipgrams.append((target, context))

    print('skipgrams = {}'.format(skipgrams[0:2]))

    def batch_generator():
        x = []
        y = []
        for target, context in skipgrams:
            x.append(word_to_idx[target])
            y.append([word_to_idx[c] for c in context])
        x = torch.tensor(x, dtype=torch.long).to(device)
        y = torch.tensor(y, dtype=torch.long).to(device)

        size = len(y)
        idx = torch.randperm(size)
        batch_size = 3
        lo = 0
        while lo < size:
            hi = min(lo + batch_size, size)
            sample_idx = idx[lo:hi]
            x_sample = x[sample_idx]
            y_sample = y[sample_idx, :]
            lo += batch_size
            yield x_sample, y_sample

    model = word2vec(vocab_size, context_size).to(device)
    
    def eval_accuracy():
        model.eval()
        correct = 0
        count = 0
        for x, y in batch_generator():
            logit = model(x)
            topk = logit.topk(2 * context_size, dim=1).indices
            batch_size, window_size = y.shape
            for batch_idx in range(batch_size):
                for window_idx in range(window_size):
                    if topk[batch_idx, window_idx] in y[batch_idx, :]:
                        correct += 1
                    count += 1
        accuracy = correct / count
        print('correct: {}, count: {}, accuracy: {}'.format(correct, count, accuracy))

    # criterion = torch.nn.BCEWithLogitsLoss().to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)

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
    max_epoch = 200
    for epoch in range(max_epoch):
        model.train()
        sum_loss = 0
        for x, y in batch_generator():
            logit = model(x)
            batch_size = logit.shape[0]
            target = torch.zeros(batch_size, vocab_size).to(device)
            for i in range(batch_size):
                target[i, y[i, :]] = 1

            # print('x: {}'.format(x))
            # print('logit: {}'.format(logit))
            # print('target: {}'.format(target))
            
            loss = criterion(logit, target).to(device)
            sum_loss += loss.detach()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print("epoch {}, loss: {}".format(epoch, sum_loss))
        eval_accuracy()

    print("Trained in {} seconds".format(time.time() - train_start))
    eval_accuracy()

    dolor = model.embeddings.weight[word_to_idx["dolor"]]
    sit = model.embeddings.weight[word_to_idx["sit"]]
    maximus = model.embeddings.weight[word_to_idx["maximus"]]
    similarity = torch.nn.CosineSimilarity(dim=0)
    print('embedding[dolor]: {}'.format(dolor))
    print('embedding[sit]: {}'.format(sit))
    print('embedding[maximus]: {}'.format(maximus))
    print('similarity(dolor, sit): {}'.format(similarity(dolor, sit)))
    print('similarity(dolor, maximus): {}'.format(similarity(dolor, maximus)))
    
    
if __name__ == '__main__':
    main()
