import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence


class MMIModel(nn.Module):

    def __init__(self, num_word_types, num_char_types, num_labels, word_dim,
                 char_dim, width, num_lstm_layers):
        super(MMIModel, self).__init__()
        self.wemb = nn.Embedding(num_word_types, word_dim, padding_idx=0)
        self.cemb = nn.Embedding(num_char_types, char_dim, padding_idx=0)
        self.num_labels = num_labels
        self.width = width

        self.loss = Loss()

        self.past = PastEncoder(self.wemb, width, num_labels)
        self.future = FutureEncoder(self.wemb, self.cemb, num_lstm_layers,
                                    num_labels)

    def forward(self, past_words, future_words, padded_chars, char_lengths,
                is_training=True):
        past_rep = self.past(past_words)
        future_rep = self.future(future_words, padded_chars, char_lengths)

        if is_training:
            loss = self.loss(past_rep, future_rep)
            return loss

        else:
            future_probs, future_indices = future_rep.max(1)
            return future_probs, future_indices


class Loss(nn.Module):

    def __init__(self):
        super(Loss, self).__init__()
        self.entropy = Entropy()

    def forward(self, past_rep, future_rep):
        pZ_Y = F.softmax(future_rep, dim=1)
        pZ = pZ_Y.mean(0)
        hZ = self.entropy(pZ)

        x = pZ_Y * F.log_softmax(past_rep, dim=1)  # B x m
        hZ_X_ub = -1.0 * x.sum(dim=1).mean()

        loss = hZ_X_ub - hZ
        return loss


class Entropy(nn.Module):

    def __init__(self):
        super(Entropy, self).__init__()

    def forward(self, probs):
        x = probs * torch.log(probs)
        entropy = -1.0 * x.sum()
        return entropy


class PastEncoder(nn.Module):

    def __init__(self, wemb, width, num_labels):
        super(PastEncoder, self).__init__()
        self.wemb = wemb
        self.linear = nn.Linear(2 * width * wemb.embedding_dim, num_labels)

    def forward(self, words):
        wembs = self.wemb(words)  # B x 2width x d_w
        rep = self.linear(wembs.view(words.shape[0], -1))  # B x m
        return rep


class FutureEncoder(nn.Module):

    def __init__(self, wemb, cemb, num_layers, num_labels):
        super(FutureEncoder, self).__init__()
        self.wemb = wemb
        self.cemb = cemb
        self.lstm = nn.LSTM(cemb.embedding_dim, cemb.embedding_dim, num_layers,
                            bidirectional=True)
        self.linear = nn.Linear(wemb.embedding_dim + 2 * cemb.embedding_dim,
                                num_labels)

    def forward(self, words, padded_chars, char_lengths):
        B = len(char_lengths)
        wembs = self.wemb(words)  # B x d_w

        packed = pack_padded_sequence(self.cemb(padded_chars), char_lengths)
        output, (final_h, final_c) = self.lstm(packed)

        final_h = final_h.view(self.lstm.num_layers, 2, B,
                               self.lstm.hidden_size)[-1]         # 2 x B x d_c
        cembs = final_h.transpose(0, 1).contiguous().view(B, -1)  # B x 2d_c

        rep = self.linear(torch.cat([wembs, cembs], 1))  # B x m
        return rep
