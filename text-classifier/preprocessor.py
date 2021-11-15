import argparse
import os
import torch
import dgl
import dgl.function as fn
import math
import json
from torch import nn
from queue import Queue
from dgl.data.utils import save_graphs, load_graphs

class TaxoDataManager():
  def __init__(self, root, taxonomy_file, data_name):
    self.taxonomy_file = root + taxonomy_file
    self.root = root
    self.data_name = data_name
    self.child2parent = {}
    self.parent2child = {}
    self.id2label = {}
    self.label2id = {}
    self.g = dgl.DGLGraph()
    self.label_id = 0

  def get_graph(self):
    return self.g
  
  def parent_from_child(self, child):
    return self.child2parent.get(child)
  
  def label_from_id(self, id):
    return self.get_id2label.get(id)
  
  def id_from_label(self, label):
    return self.label2id.get(label)
  
  def child_from_parent(self, parent):
    return self.parent2child.get(parent)

  def load_from_taxofile(self):
    que = Queue()

    with open(self.taxonomy_file, 'r') as taxo:
      taxonomy_data = json.load(taxo)

    que.put((taxonomy_data, [0]))

    self.label_id = 1
    self.label2id = {}
    self.id2label = {}
    self.child2parent = {}
    self.parent2child = {0:[]}

    while ( not que.empty() ):
      (childs, label) = que.get()

      for child in childs:
        que.put((childs[child], label + [self.label_id]))
         #Some labels have the same name.
        if self.label2id.get(child) != None:
          self.label2id[child].append(self.label_id)
        else :
          self.label2id[child] = [self.label_id]
    
        if self.id2label.get(self.label_id) !=None:
          print("Error: label ID should be unique.")
        self.id2label[self.label_id] = child
        self.parent2child[self.label_id] = []

        self.child2parent[self.label_id] = label[-1]
        self.parent2child[label[-1]].append(self.label_id)
        self.label_id  = self.label_id +1
    

    self.g = dgl.DGLGraph()

    self.g.add_nodes(self.label_id)

    #save taxonomy in taxo_graph, and id - label pair in taxonomy_id.txt

    with open(self.root + self.data_name + '_taxonomy_id.txt', "w") as fout:
      for id in self.id2label:
        label_name = self.id2label[id]
        fout.write(f"{id}\t{label_name}\n")
        self.g.add_edges(id,id)
    
        parent_id = self.child2parent.get(id)
        if parent_id!=None:
          self.g.add_edges(id, parent_id)
          self.g.add_edges(parent_id,id)
  
    with open(self.root + self.data_name + '_child2parent.txt', "w") as fout:
      for id in self.id2label:
        parent_id = self.child2parent.get(id)
        if parent_id != None:
          fout.write(f"{id}\t{parent_id}\n")


  def load_graph(self):
    try: 
      glist, label_dict = load_graphs(self.root + self.data_name + '_taxo_graph.bin')
      self.g = glist[0]
      print("load success")
    except:
      self.load_from_taxofile()
    return

  def load_dict(self):
    try:
      self.label2id = {}
      self.id2label = {}
      self.child2parent = {}
      self.parent2child = {0:[]}
      with open(self.root + self.data_name + '_taxonomy_id.txt', "r") as fin:
        for line in fin:
          line = line.strip()
          if line:
            segs = line.split("\t")
            if len(segs) == 1:
              segs.append('')

            self.id2label[int(segs[0])] = segs[1]
            self.parent2child[int(segs[0])] = []
            self.label_id = int(segs[0]) + 1
            if self.label2id.get(segs[1]) != None:
              self.label2id[segs[1]].append(int(segs[0]))
            else:
              self.label2id[segs[1]] = [int(segs[0])]
      
      print("load taxonomy success")

      with open(self.root + self.data_name + '_child2parent.txt', "r") as fin:
        for line in fin:
          line = line.strip()
          if line:
            segs = line.split("\t")
            self.child2parent[int(segs[0])] = int(segs[1])
            self.parent2child[int(segs[1])].append(int(segs[0]))

      print("load dict success")
    except:
      self.load_from_taxofile()

  def load_all(self):    
    self.load_graph()
    self.load_dict()

  def save_graph(self):
    save_graphs(self.root + self.data_name + '_taxo_graph.bin',[self.g])

  def save_dict(self):
      with open(self.root + self.data_name + '_taxonomy_id.txt', "w") as fout:
        for id in self.id2label:
          label_name = self.id2label[id]
          fout.write(f"{id}\t{label_name}\n")

      with open(self.root + self.data_name + '_child2parent.txt', "w") as fout:
        for id in self.id2label:
          parent_id = self.child2parent.get(id)
          if parent_id != None:
            fout.write(f"{id}\t{parent_id}\n")

  def save_all(self):
    self.save_graph()
    self.save_dict()
  
class DocumentManager():
  def __init__(self, file_name, root, dataset_name, tokenizer, encoder, taxo_manager):
    self.file_name = file_name
    self.root = root
    self.dataset_name = dataset_name
    self.tokenizer = tokenizer
    self.encoder = encoder
    self.id2tokens = {}
    self.id2core = {}
    self.id2category = {}
    self.id2pos = {}
    self.id2nonneg = {}
    self.taxo_manager = taxo_manager

  def get_ids(self):
    return self.id2tokens.keys()

  def get_tokens (self, id) :
    return self.id2tokens.get(id)

  def get_output_label(self, id):
    return (self.id2pos[id], self.id2nonneg[id])

  def load_from_raw (self):
    with open(self.root + self.file_name, "r") as fin:
      for line in fin:
        data = json.loads(line)
        id = data["asin"]
        raw_text = data["reviewText"]
        core = data["core_classes"]
        category = data["categories"]
        tokens = self.tokenizer(raw_text)

        #find positive, nonnegative set
        self.id2pos[id] = []
        self.id2nonneg[id] = []

        for core_class in core:
          parent = 0
          for node in core_class:
              if node == "root":
                parent = 0
                continue
              childs = self.taxo_manager.child_from_parent(parent)
              label_list = self.taxo_manager.id_from_label(node)
              label_id = ( list(set(childs) & set(label_list)) )[0]
              parent = label_id

          self.id2pos[id] = self.id2pos[id] + [parent] + [self.taxo_manager.parent_from_child(parent)]
          self.id2nonneg[id] = self.id2nonneg[id] + [parent] + self.taxo_manager.child_from_parent(parent)


        cls = self.encoder(tokens)
        #truncate with 500 words
        #token_list = tokens['input_ids'].tolist()[0]
        #token_list = token_list[:500]  

        self.id2tokens[id] = cls.tolist()
        self.id2core[id] = core
        self.id2category[id] = category
        self.save_tokens()

  def load_tokens (self):
    with open(self.root + self.dataset_name + '_tokens.json', "r") as fin:
      data = json.load(fin)
      self.id2tokens = data

  def load_dicts (self):
    with open(self.root + self.file_name, "r") as fin:
      for line in fin:
        data = json.loads(line)
        id = data["asin"]
        core = data["core_classes"]
        category = data["categories"]

        #find positive, nonnegative set
        self.id2pos[id] = []
        self.id2nonneg[id] = []

        for core_class in core:
          parent = 0
          for node in core_class:
              if node == "root":
                parent = 0
                continue
              childs = self.taxo_manager.child_from_parent(parent)
              label_list = self.taxo_manager.id_from_label(node)
              label_id = ( list(set(childs) & set(label_list)) )[0]
              parent = label_id

          self.id2pos[id] = self.id2pos[id] + [parent] + [self.taxo_manager.parent_from_child(parent)]
          self.id2nonneg[id] = self.id2nonneg[id] + [parent] + self.taxo_manager.child_from_parent(parent)

        self.id2core[id] = core
        self.id2category[id] = category

  def save_tokens (self):
    with open(self.root + self.dataset_name + '_tokens.json', "w") as fout:
      data = json.dumps(self.id2tokens)
      fout.write(f"{data}")
