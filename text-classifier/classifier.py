import argparse
import os
import torch
import math
from torch import nn


class TextClassifier(nn.Module):
    def __init__(self, class_encoder, dimension, output_dim, activation = nn.Sigmoid()):
      #dimension = (class_representation dimension ,document representation dimension)
      #output_dim = total class number
      #activation is Sigmoid function by default.
      super(TextClassifier, self).__init__()
      self.class_encoder = class_encoder
      self.weight = nn.Parameter(torch.Tensor(dimension[0], dimension[1]))
      self.activation = nn.Sigmoid()
      self.reset_parameters()
      self.output_dim = output_dim

    def reset_parameters(self):
      stdv = 1. / math.sqrt(self.weight.size(1))
      self.weight.data.uniform_(-stdv, stdv)

    def forward(self, g, d, features):
      h = class_encoder(g, features)
      
      #c = torch.transpose(h, 0, 1)
      p = torch.mm(h, self.weight)
      p = torch.mm(p, d)
      p = torch.exp(p)
      p = self.activation(p)
      return p