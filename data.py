import os
import random
import torch

from collections import Counter
from torch.nn.utils.rnn import pad_sequence


class Data(object):

    def __init__(self, data_path):
        self.data_path = data_path
        self.PAD = '<pad>'
        self.UNK = '<unk>'

        self.sents = []   # Index sequences
        self.golds = []
        self.w2i = {self.PAD: 0, self.UNK: 1}
        self.i2w = [self.PAD, self.UNK]
        self.c2i = {self.PAD: 0, self.UNK: 1}
        self.i2c = [self.PAD, self.UNK]
        self.word_counter = []
        self.char_counter = []
        self.label_counter = Counter()

        self.get_data()

    def get_data(self):
        wcount = Counter()
        ccount = Counter()
        def add(w):
            wcount[w] += 1
            if w not in self.w2i:
                self.i2w.append(w)
                self.w2i[w] = len(self.i2w) - 1
            for c in w:
                ccount[c] += 1
                if c not in self.c2i:
                    self.i2c.append(c)
                    self.c2i[c] = len(self.i2c) - 1
            return self.w2i[w]

        with open(self.data_path, 'r') as data_file:
            for line in data_file:
                toks = line.split()
                if toks:
                    self.sents.append([add(tok) for tok in toks])

        self.word_counter = [wcount[self.i2w[i]] for i in range(len(self.i2w))]
        self.char_counter = [ccount[self.i2c[i]] for i in range(len(self.i2c))]

        gold_path = self.data_path[:-5] + 'tags'
        assert os.path.isfile(gold_path)
        self.get_golds(gold_path)

    def get_golds(self, gold_path):
        with open(gold_path, 'r') as f:
            index = 0
            for line in f:
                labels = line.split()
                if labels:
                    self.label_counter.update(labels)
                    self.golds.append(labels)
                    assert len(self.golds[index]) == len(self.sents[index])
                    index += 1
        assert len(self.golds) == len(self.sents)

    def get_batches(self, batch_size):
        pairs = []
        for i in range(len(self.sents)):
            pairs.extend([(i, j) for j in range(len(self.sents[i]))])

        random.shuffle(pairs)
        batches = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:min(i + batch_size, len(pairs))]
            sorted_batch = sorted(batch,
                                  key=lambda x:
                                  len(self.i2w[self.sents[x[0]][x[1]]]),
                                  reverse=True)
            batches.append(sorted_batch)

        return batches

    def tensorize_batch(self, batch, device, width):
        def get_context(i, j, width):
            left = [0 for _ in range(width - j)] + \
                   self.sents[i][max(0, j - width):j]
            right = [0 for _ in range((j + width) - len(self.sents[i]) + 1)] + \
                    self.sents[i][j + 1: min(len(self.sents[i]), j + width) + 1]
            return left + right

        contexts = [get_context(i, j, width) for (i, j) in batch]
        targets = [self.sents[i][j] for (i, j) in batch]
        seqs = [torch.LongTensor([self.c2i[c] for c in self.i2w[target]])
                for target in targets]

        X = torch.LongTensor(contexts).to(device)  # B x 2width
        Y1 = torch.LongTensor(targets).to(device)  # B
        Y2 = pad_sequence(seqs, padding_value=0).to(device)  # T x B
        lengths = torch.LongTensor([seq.shape[0] for seq in seqs]).to(device)

        return X, Y1, Y2, lengths
