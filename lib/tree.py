class ClassNode:
  def __init__(self, name, parent, depth):
    self.name = name
    self.parent = parent
    self.depth = depth
    self.children_map = dict()

  def propagate(self, dic):
    for k, v in dic.items():
      self.children_map[k] = ClassNode(k, self, self.depth + 1)
      self.children_map[k].propagate(v)

  def full_path(self):
    path = []
    cursor = self
    while cursor.parent is not None:
      path.append(cursor.name)
      cursor = cursor.parent
    path.reverse()
    return path

  def children(self):
    return self.children_map.values()
    
class ClassTree:
  def __init__(self, tree_json):
    self.root = ClassNode('root', None, 0)
    self.root.propagate(tree_json)
