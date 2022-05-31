import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from tokenization import (
    embeddings, 
    NUM_LETTRES
)


class CharEmbedding(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3):
        super(CharEmbedding, self).__init__()
        self.emb_layer = nn.Embedding(input_size, hidden_size)
        self.bigru = BiGRU(hidden_size, hidden_size, num_layers)
        

    def forward(self,cx):
        char_emb = self.emb_layer(x)
        char_emb = self.bigru(char_emb)
        return char_emb


class BiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3):
        super(BiGRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, 
                          hidden_size, 
                          num_layers, 
                          batch_first=True, 
                          bidirectional=True)

    def forward(self, x, hidden=None): #B x N x E
        if hidden == None:
            hidden = torch.zero(self.num_layer*2, x.size(0), self.hidden_size)
        out, _ = self.gru(x)

        return out


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, 
                          hidden_size, 
                          num_layers, 
                          batch_first=True)

    def forward(self, x, hidden=None): #B x N x E
        if hidden == None:
            hidden = torch.zero(self.num_layer, x.size(0), self.hidden_size)
        out, _ = self.gru(x)

        return out


class Attention(nn.Module):
    def __init__(self, key_size, query_size, hidden_size, dropout):
        super(Attention, self).__init__()
        self.W_k = nn.Linear(key_size, hidden_size, bias=False)
        self.W_q = nn.Linear(query_size, hidden_size, bias=False)
        self.w_v = nn.Linear(hidden_size, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, keys, queries):
        scores = []
        for k in keys:
            k, queries = self.W_k(k), self.W_q(queries), 
            features = k.unsqueeze(2) + queries.unsqueeze(2)
            s = self.w_v(nn.tanh(features))
            scores.append(s)

        att = torch.exp(torch.stack(scores))/torch.sum(torch.exp(torch.stack(scores)))
        rep = sum([torch.bmm(a, k) for a, k in zip(att, keys)])
        return rep


class MatchScore(nn.Module):
    def __init__(self, hidden_size, output_size, dropout):
        super().__init__()
        self.l1 = nn.Linear(hidden_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=1)

    
    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        scores = self.w_v(nn.tanh(x))
        return self.softmax(scores)


class WC_Embedding(nn.Module):
    def __init__(self, hidden_size, dropout):
        super(WC_Embedding, self).__init__()
        self.word_emb = nn.Embedding.from_pretrained(torch.Tensor(embeddings))
        self.char_emb = CharEmbedding(NUM_LETTRES, hidden_size)

        self.bigru = BiGRU(hidden_size*2, hidden_size)

    def forward(self, words, chars): 
        word_emb = self.word_emb(words)
        char_emb = self.char_emb(chars) 

        emb, _ = self.bigru(word_emb+char_emb)
        return emb
        

class Encoder(nn.Module):
    def __init__(self, hidden_size, dropout):
        super(Encoder, self).__init__()

        self.wc_emb = WC_Embedding(hidden_size, dropout)
        self.bigru = BiGRU(hidden_size*2, hidden_size)
        self.attention = Attention(hidden_size*2, hidden_size*2, hidden_size, dropout)
        self.gru = GRU(hidden_size, hidden_size)
        self.matchscore = MatchScore(hidden_size*2, 2, dropout)
        self.sig = nn.Sigmoid()
        self.linear = nn.Linear(hidden_size*2, hidden_size)



    def forward(self, q_word, q_char, p_word, p_char): 

        q_emb = self.wc_emb(q_word, q_char)
        p_emb = self.wc_emb(p_word, p_char)

        q_att = self.attention(q_emb, p_emb[: -1, :])
        p_sent = self.gru(torch.bmm(self.sig(self.linear(p_emb[: -1, :]+q_att)),p_emb[: -1, :]+q_att))

        p_att = self.attention(p_emb, q_emb[:, -1, :])
        q_sent = self.gru(torch.bmm(self.sig(self.linear(q_emb[:, -1, :]+p_att)), q_emb[:, -1, :]+p_att))

        q_r = self.attention(q_emb, q_sent[:, -1, :])
        p_r = self.attention(p_sent, q_r)

        return self.matchscore(q_r+p_r)







        

