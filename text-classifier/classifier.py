import argparse
import os
import torch
import math
from torch import nn


class TextClassifier(nn.Module):
    def __init__(self, class_encoder, document_encoder, dimension, token_dimension, g, features, activation = nn.Sigmoid()):
      
      #dimension = (class_representation dimension ,document representation dimension)
      #output_dim = total class number
      #activation is Sigmoid function by default.
      super(TextClassifier, self).__init__()
      self.class_encoder = class_encoder
      self.document_encoder = document_encoder
      self.weight = nn.Parameter(torch.Tensor(dimension[0], dimension[1]))
      self.token_dimension = token_dimension
      self.activation = nn.Sigmoid()
      self.graph = g
      self.features = features
      self.reset_parameters()


    def reset_parameters(self):
      stdv = 1. / math.sqrt(self.weight.size(1))
      self.weight.data.uniform_(-stdv, stdv)

    def forward(self,input):
      tokens = input[:self.token_dimension]

      mask = input[self.token_dimension:]


      #get CLS token
      token = torch.reshape(tokens,(1,-1))
      d = self.document_encoder(token)[0][0]
      #target = torch.diag(mask).float()

      d = torch.reshape(d, (-1,1))
      h = self.class_encoder(self.graph, self.features)
      #c = torch.transpose(h, 0, 1)
      p = torch.mm(h, self.weight)
      p = torch.mm(p, d)
      p = torch.exp(p)
      p = self.activation(p)
      #p = torch.mm(target, p)
      #del d
      #torch.cuda.empty_cache()
    
      return p