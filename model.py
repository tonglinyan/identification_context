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


class Attention(nn.Module):
    def __init__(self, key_size, query_size, hidden_size, dropout):
        super(Attention, self).__init__()
        self.W_k = nn.Linear(key_size, hidden_size, bias=False)
        self.W_q = nn.Linear(query_size, hidden_size, bias=False)
        self.w_v = nn.Linear(hidden_size, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.tanh = nn.Tanh()

    def forward(self, keys, queries):

        queries = self.W_q(queries)        
        att = self.attention(keys, queries)
        rep = (keys * att.unsqueeze(-1)).sum(1)#[sum([a*k for a, k in zip(att.permute(1, 0), k_batch)]) for k_batch in keys]
        return rep

    def attention(self, keys, queries):
        scores = []
        for i in range(keys.size(1)):
            k = self.W_k(keys[:,i,:])
            
            features = k + queries
            s = self.w_v(self.tanh(features))
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


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3):
        super(GRU, self).__init__()
        self.linear = nn.Linear(input_size, input_size)
        self.sig = nn.Sigmoid()
        self.gru = nn.GRU(input_size, 
                          hidden_size, 
                          num_layers, 
                          batch_first=True)

    def forward(self, u, c): 
        ## u.size(): (batch_size, feature_size)
        ## c.size(): (batch_size, feature_size)
        concat = torch.concat((u, c), 1)
        g = self.sig(self.linear(concat))
        print(g.size())
        concat1 = g * concat
        out, _ = self.gru(concat1)
        
        return out

class MatchScore(nn.Module):
    def __init__(self, hidden_size, output_size, dropout):
        super().__init__()
        self.l1 = nn.Linear(hidden_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.w_v = nn.Linear(hidden_size, output_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        scores = self.w_v(self.tanh(x))
        return self.softmax(scores)


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
        

class Encoder(nn.Module):
    def __init__(self, hidden_size, dropout):
        super(Encoder, self).__init__()

        self.wc_emb = WC_Embedding(hidden_size, dropout)
        self.attention_q = Attention(hidden_size, hidden_size, hidden_size, dropout)
        self.attention_p = Attention(hidden_size, hidden_size, hidden_size, dropout)
        self.attention_q1 = Attention(hidden_size, hidden_size, hidden_size, dropout)
        self.attention_p1 = Attention(hidden_size, hidden_size, hidden_size, dropout)
        self.gru = GRU(hidden_size, hidden_size)
        self.matchscore = MatchScore(hidden_size*2, 2, dropout)


    def forward(self, q_word, q_char, p_word, p_char): 

        q_emb = self.wc_emb(q_word, q_char)
        p_emb = self.wc_emb(p_word, p_char)

        q_att = self.attention_q(q_emb, p_emb[:,-1,:])
        # q_att.size() = (batch_size, hidden_size)
        p_sent = self.gru(p_emb[:,-1,:], q_att)
    
        p_att = self.attention_p(p_emb, q_emb[:,-1,:])
        q_sent = self.gru(torch.bmm(self.sig(self.linear(q_emb[:,-1,:]+p_att)), q_emb[:,-1,:]+p_att))

        q_r = self.attention_q1(q_emb, q_sent[:,-1,:])
        p_r = self.attention_p1(p_sent, q_r)

        return self.matchscore(q_r+p_r)







        

