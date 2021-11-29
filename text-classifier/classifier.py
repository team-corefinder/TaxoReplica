import argparse
import os
import torch
import math
from torch import nn


class TextClassifier(nn.Module):
    def __init__(self, class_encoder, document_encoder, dimension, token_dimension, g, features, activation, rescaling = False):
      
      #dimension = (class_representation dimension ,document representation dimension)
      #output_dim = total class number
      #activation is Sigmoid function by default.
      super(TextClassifier, self).__init__()
      self.class_encoder = class_encoder
      self.document_encoder = document_encoder
      self.weight = nn.Parameter(torch.Tensor(dimension[0], dimension[1]))
      self.token_dimension = token_dimension
      self.activation = activation
      #self.activation = nn.Softmax(dim = 0)
      self.graph = g
      self.features = features
      self.rescaling = rescaling
      self.reset_parameters()


    def reset_parameters(self):
      stdv = 1. / math.sqrt(self.weight.size(1))
      self.weight.data.uniform_(-stdv, stdv)

    def forward(self,input):
      tokens = input[:, :self.token_dimension]
      mask = input[:, self.token_dimension:]

      L = mask.shape[1]
      mask = torch.reshape(mask, (-1,1,L))

      #get CLS token
      d = self.document_encoder(tokens)[:,0,:]
      #rescale
      stdv = 1. / math.sqrt(d.shape[1])
      d = d * stdv

      d = torch.reshape(d, (-1,1,d.shape[1]))
      h = self.class_encoder(self.graph, self.features)
      p = torch.mm(h, self.weight)
      p = torch.transpose(p, 0, 1)

      p = torch.matmul(d, p)

      if self.rescaling:
        p = torch.exp(p)

      p = self.activation(p)
      if self.rescaling:
        p = (p -1/2)*2
        
      p = torch.mul(mask, p)
      p = torch.transpose(p, 1, 2)

      return p