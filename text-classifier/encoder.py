import argparse
import os
import torch
import dgl
import dgl.function as fn
import math
from torch import nn
from transformers import BertTokenizer, BertModel, BertConfig


class DocuEncoder():
  def __init__(self, config = BertConfig()):
    try:
      model = BertModel.from_pretrained("./pretrained/BERT_model.pt")
      model.eval()

    except:
      model = BertModel.from_pretrained("bert-base-uncased")
      model.save_pretrained("./pretrained/BERT_model.pt")

    self.model = model
    self.config = config
    try:
      self.tokenizer = BertTokenizer.from_pretrained("./pretrained/BERT_tokenizer.pt")
      self.tokenizer.eval()
    
    except:
      self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
      self.tokenizer.save_pretrained("./pretrained/BERT_tokenizer.pt")

  def PrintModelConfig(self):
    print(self.model.config)
  
  def Tokenize(self, document):
    #input: string -> output: tokenized string tensor
    tokenized_input = self.tokenizer(document, return_tensors='pt', truncation = True)
    return tokenized_input

  def Encode(self, tokens):

    #input: token  -> output : hidden_size tensor
    output = self.model(**tokens)
    CLS_token = output.last_hidden_state[0][0]
    return CLS_token


class ClassEncoder(nn.Module):
    def __init__(self, model, feature_model):
      super(ClassEncoder, self).__init__()
      self.model = model
      self.feature_model = feature_model
    
    def CalculateFeatures(self, g):
      features = self.feature_model(g)
      return features

    def forward(self, g, features):
      h = self.model.forward(g,features)
      return h
