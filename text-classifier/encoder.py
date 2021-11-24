import argparse
import os
import torch
import dgl
import dgl.function as fn
import math
from torch import nn
from transformers import BertTokenizer, BertModel, BertConfig


class DocuEncoder(nn.Module):
  def __init__(self, config = BertConfig()):
    super(DocuEncoder, self).__init__()
    try:
      model = BertModel.from_pretrained("/root/data/pretrained/BERT_model.pt")
      model.eval()

    except:
      model = BertModel.from_pretrained("bert-base-uncased")
      model.save_pretrained("/root/data/pretrained/BERT_model.pt")

    self.model = model
    self.config = config

  def PrintModelConfig(self):
    print(self.model.config)


  def forward(self, tokens):

    #input: token  -> output : hidden_size tensor
    output = self.model(tokens)
    return output.last_hidden_state

class DocumentTokenizer():
  def __init__(self, max_length):
    try:
      self.tokenizer = BertTokenizer.from_pretrained("./pretrained/BERT_tokenizer.pt")
      self.tokenizer.eval()
      
    except:
      self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
      self.tokenizer.save_pretrained("./pretrained/BERT_tokenizer.pt")
    self.max_length = max_length

  def Tokenize(self, document):
    #input: string -> output: tokenized string tensor
    tokenized_input = self.tokenizer(document, return_tensors='pt', padding='max_length', truncation = True, max_length = self.max_length)
    return tokenized_input
  
  def DecodeToken(self, tokens):
    raw = self.tokenizer.convert_ids_to_tokens(tokens)
    return raw

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
