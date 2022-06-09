import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from tokenization import (
    embeddings, 
    embedding_size, 
    NUM_LETTRES
)


class CharEmbedding(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3):
        super(CharEmbedding, self).__init__()
        self.emb_layer = nn.Embedding(input_size, hidden_size)
        self.bigru = BiGRU(hidden_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size*2, hidden_size)

    def forward(self, x):
        ## x.size() = (batch_size*number_words, number_letters)
        char_emb = self.emb_layer(x) 
        ## char_emb.size() = (batch_size*number_words, number_letters, feature_size)
        char_emb = self.bigru(char_emb) 
        
        return char_emb[:,-1,:]



class BiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3):
        super(BiGRU, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, 
                          hidden_size, 
                          num_layers, 
                          batch_first=True, 
                          bidirectional=True)
        self.linear = nn.Linear(hidden_size*2, hidden_size)

    def forward(self, x, hidden=None):
        if hidden == None:
            hidden = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size)
        out, _ = self.gru(x)
        out = self.linear(out)
        return out


class WC_Embedding(nn.Module):
    def __init__(self, hidden_size, dropout):
        super(WC_Embedding, self).__init__()
        self.word_emb = nn.Embedding.from_pretrained(torch.Tensor(embeddings))
        self.linear = nn.Linear(torch.Tensor(embeddings).size(1), hidden_size)
        self.char_emb = CharEmbedding(NUM_LETTRES, hidden_size)

        self.bigru = BiGRU(hidden_size+embedding_size, hidden_size)

    def forward(self, words, chars): 
       
        #word_emb = self.linear(self.word_emb(words))
        word_emb = self.word_emb(words)
        char_emb = self.char_emb(chars)
        char_emb = char_emb.unsqueeze(1).view(word_emb.size(0), word_emb.size(1), -1) 

        emb_concat = torch.cat((word_emb, char_emb), 2)
        emb = self.bigru(emb_concat)
        
        return emb


class match_GRU(nn.Module):
    def __init__(self, key_size, query_size, hidden_size, dropout, gate=True, num_layers=3):
        super(match_GRU, self).__init__()
        self.W_k = nn.Linear(key_size, hidden_size, bias=False)
        self.W_q = nn.Linear(query_size, hidden_size, bias=False)
        self.w_v = nn.Linear(hidden_size, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.W_g = nn.Linear(hidden_size, hidden_size)
        self.gru = nn.GRU(hidden_size, 
                          hidden_size, 
                          num_layers, 
                          batch_first=True)
        self.gated = gate

    def forward(self, keys, queries):
        if self.gated:
            q_att = []
            for i in range(queries.size(1)):
                q = self.W_q(queries[:,i,:])        
                att = self.attention(keys, q)
                rep = (keys * att.unsqueeze(-1)).sum(1)
                q_att.append(rep)
            q_att = torch.stack(q_att).permute(1, 0, 2)
            
            emb_p = self.gate(queries, q_att)
            return emb_p
        else:
            queries = self.W_q(queries)        
            att = self.attention(keys, queries)
            rep = (keys * att.unsqueeze(-1)).sum(1)
            return rep


    def gate(self, u, c):
        
        g = torch.sigmoid(self.W_g(u+c))
        u_c = g * (u+c)
        out, _ = self.gru(u_c)
        return out


    def attention(self, keys, queries):
        scores = []
        for i in range(keys.size(1)):
            k = self.W_k(keys[:,i,:])
            
            features = k + queries
            s = self.w_v(torch.tanh(features))
            scores.append(s.squeeze(1))
        
        scores = torch.exp(torch.stack(scores))/torch.sum(torch.exp(torch.stack(scores)))  
        return scores.permute(1, 0)

    def masked_softmax(x,lens=None):
        #X : B x N
        x = x.view(x.size(0),x.size(1))
        if lens is None:
            lens = torch.zeros(x.size(0),1).fill_(x.size(1))
        mask  = torch.arange(x.size(1),device=x.device).view(1,-1) < lens.view(-1,1)
        x[~mask] = float('-inf')
        return x.softmax(1)


class MatchScore(nn.Module):
    def __init__(self, hidden_size, output_size, dropout):
        super().__init__()
        self.l1 = nn.Linear(hidden_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.w_v = nn.Linear(hidden_size, output_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        scores = self.w_v(torch.tanh(x))
        return self.softmax(scores)


class Encoder(nn.Module):
    def __init__(self, hidden_size, dropout):
        super(Encoder, self).__init__()

        self.wc_emb = WC_Embedding(hidden_size, dropout)
        self.match_gru_p = match_GRU(hidden_size, hidden_size, hidden_size, dropout)
        self.match_gru_q = match_GRU(hidden_size, hidden_size, hidden_size, dropout)
        self.match_gru_rq = match_GRU(hidden_size, hidden_size, hidden_size, dropout, gate=False)
        self.match_gru_rp = match_GRU(hidden_size, hidden_size, hidden_size, dropout, gate=False)
        self.matchscore = MatchScore(hidden_size, 2, dropout)


    def forward(self, q_word, q_char, p_word, p_char): 

        q_emb = self.wc_emb(q_word, q_char) ## batch_size * Q * feature_size
        p_emb = self.wc_emb(p_word, p_char) ## batch_size * P * feature_size

        p_sent_emb = self.match_gru_p(q_emb, p_emb) ## batch_size * P * feature_size
        q_sent_emb = self.match_gru_q(p_emb, q_emb) ## batch_size * Q * feature_size

        q_r = self.match_gru_rq(q_emb, q_sent_emb[:,-1,:]) ## batch_size * feature_size
        p_r = self.match_gru_rp(p_sent_emb, q_r) ## batch_size * feature_size

        return self.matchscore(q_r+p_r)







        

