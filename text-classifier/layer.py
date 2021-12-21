import argparse
import os
import torch
from torch import nn
import dgl
import dgl.function as fn
import math

#### GCN from TaxoExpan: Self-supervised Taxonomy Expansion with Position-Enhanced Graph Neural Network ####

class GCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim,activation, dropout ):
      super(GCNLayer, self).__init__()
      self.input_dim = input_dim
      self.output_dim = output_dim
      self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
      if dropout:
        self.dropout = nn.Dropout(p = dropout)
      else:
        self.dropout = 0
      if activation:
        self.activation = activation
      else:
        self.activation = 0
      self.reset_parameters()
      
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, g, h):
      if self.dropout:
        h = self.dropout(h)
      h = torch.mm(h, self.weight)
      h = h * g.ndata['norm']
      g.ndata['h'] = h
      g.update_all(fn.copy_src(src='h', out='m'), fn.sum(msg='m', out='h'))
      h = g.ndata.pop('h')
      h = h * g.ndata['norm']

      if self.activation:
        h = self.activation(h)
      return h

class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, activation, in_dropout=0.1, hidden_dropout=0.1, output_dropout=0.0):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GCNLayer(in_dim, hidden_dim, activation, in_dropout))
        # hidden layers
        for l in range(num_layers - 1):
            self.layers.append(GCNLayer(hidden_dim, hidden_dim, activation, hidden_dropout))
        # output layer
        self.layers.append(GCNLayer(hidden_dim, out_dim, None, output_dropout))

    def forward(self, g, features):
        h = features
        degs = g.in_degrees().float()
        norm = torch.pow(degs, -0.5)
        norm[torch.isinf(norm)] = 0
        norm = norm.to(h.device)
        g.ndata['norm'] = norm.unsqueeze(1)
        for layer in self.layers:
            h = layer(g, h)
        return h
      
class No_GCN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(No_GCN, self).__init__()
        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(in_dim, out_dim))

    def forward(self, g, features):
        for layer in self.layers:
            features = layer(features)
        return features

