import argparse
import os
import torch
import math
from torch import nn


class TextClassifier(nn.Module):
    def __init__(self, class_encoder, dimension, activation = nn.Sigmoid()):
      #dimension = (class_representation dimension ,document representation dimension)
      #output_dim = total class number
      #activation is Sigmoid function by default.
      super(TextClassifier, self).__init__()
      self.class_encoder = class_encoder
      self.weight = nn.Parameter(torch.Tensor(dimension[0], dimension[1]))
      self.activation = nn.Sigmoid()
      self.reset_parameters()

    def reset_parameters(self):
      stdv = 1. / math.sqrt(self.weight.size(1))
      self.weight.data.uniform_(-stdv, stdv)

    def forward(self, g, d, features):
      h = self.class_encoder(g, features)
      #c = torch.transpose(h, 0, 1)
      p = torch.mm(h, self.weight)
      #print('sum of h: %f', torch.sum(h))
      p = torch.mm(p, d)
      #print('sum of d: %f', torch.sum(d))
      p = torch.exp(p)
      #print(torch.sum(p))
      p = self.activation(p)
      return p