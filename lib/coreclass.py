import json, time, os, logging
from statistics import median

from torch.utils.data import get_worker_info

from .simcalc import SimCalc
from .common import flatten
from .tree import ClassTree

class Doc:
  def __init__(self, text, calc):
    self.text = text
    self.cache = dict()
    self.candidates = None
    self.tags = set()
    self.calc = calc
    
  def name_similarity(self, class_name):
    text = self.text
    cache = self.cache

    if cache is None:
      return self.calc.similarity(text, class_name)
    elif class_name not in cache:
      cache[class_name] = self.calc.similarity(text, class_name)
    return cache[class_name]

  def similarity(self, class_node):
    text = self.text
    cache = self.cache
    class_name = class_node.name

    if cache is None:
      return self.calc.similarity(text, class_name)
    elif class_name not in cache:
      cache[class_name] = self.calc.similarity(text, class_name)
    return cache[class_name]

  def path_score(self, class_node):
    parent = class_node.parent
    base_score = 1 if parent is None else self.path_score(parent)
    return base_score * self.similarity(class_node)

  def confidence(self, class_node):
    parent = class_node.parent
    assert parent is not None
    siblings = [c for c in parent.children() if c.name != class_node.name]
    competitors = [parent] + siblings
    return self.similarity(class_node) - max([self.similarity(n) for n in competitors])

  def filter_children_by_similarity(self, class_node):
    children = class_node.children()
    depth = class_node.depth
    return sorted(children, key=self.similarity, reverse=True)[:(depth + 2)]

  def filter_children_groups_by_path_score(self, node_groups):
    children = flatten(node_groups)
    if not children:
      return children
    depth = children[0].depth
    return sorted(children, key=self.path_score, reverse=True)[:((depth + 1) ** 2)]

  def find_candidates(self):
    candidates = []
    nodes = self.filter_children_by_similarity(self.tree.root)

    while nodes:
      candidates += nodes
      node_groups = [self.filter_children_by_similarity(n) for n in nodes]
      nodes = self.filter_children_groups_by_path_score(node_groups)

    return candidates

  def tagged_with(self, name):
    return name in self.tags

class CCMiner:
  def __init__(self, texts, tree_json, calc = None):
    self.calc = calc if calc is not None else SimCalc()
    self.documents = [Doc(t, self.calc) for t in texts]
    self.tree = ClassTree(tree_json)
    self.logger = logging.getLogger(str(os.getpid()))

  def candidates_of(self, doc):
    candidates = []
    nodes = doc.filter_children_by_similarity(self.tree.root)

    while nodes:
      candidates += nodes
      node_groups = [doc.filter_children_by_similarity(n) for n in nodes]
      nodes = doc.filter_children_groups_by_path_score(node_groups)

    return candidates

  def find_candidates(self):
    for i, doc in enumerate(self.documents):
      start = time.time()
      doc.candidates = self.candidates_of(doc)
      doc.tags = {n.name for n in doc.candidates}
      self.logger.info(str(i) + f' finding candidates took {round(time.time() - start, 3)} seconds')

  def confidence_threshold(self, class_node):
    return median([doc.confidence(class_node) for doc in self.documents if doc.tagged_with(class_node.name)])

  def coreclasses(self, doc):
    def keyFunc(class_node):
      return [doc.name_similarity(name) for name in class_node.full_path()]

    doc.candidates = sorted(doc.candidates, key=keyFunc, reverse=True)
    
    for n in doc.candidates:
      threshold = self.confidence_threshold(n)
      if n.name and doc.confidence(n) >= threshold:
        yield n.full_path()

def handle_batch(batch, OUTPUT_PATH):
  with open(OUTPUT_PATH, 'a') as outfile:
    for review in batch:
      outfile.write(json.dumps(review) + '\n')
