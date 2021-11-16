from nltk.corpus import words, brown
from nltk.tokenize import wordpunct_tokenize
from nltk.stem import WordNetLemmatizer
import json

all_words = set(brown.words())
all_words.add('CDs')
lemmatizer = WordNetLemmatizer()

def tokenize(category):
  words = wordpunct_tokenize(category)
  return [w for w in words if w.isalnum()]

def lower_first(s):
  return s[0].lower() + s[1:]

def is_valid_word(w):
  if len(w) < 2 or not w.isalpha():
    return False
  pos_list = ['n', 'v', 'a', 'r', 's']
  lemmas = [w] + [lemmatizer.lemmatize(w, pos) for pos in pos_list] + [lemmatizer.lemmatize(lower_first(w), pos) for pos in pos_list]
  valid_lemmas = [l for l in lemmas if l in all_words]
  return len(valid_lemmas) > 0

def is_valid_category(category):
  words = tokenize(category)
  is_valid = True
  for w in words:
    if not is_valid_word(w):
      is_valid = False
  return is_valid

with open('../data/amazon/unique-category-paths.csv') as c_file:
  taxo = dict()
  for index, l in enumerate(c_file):
    categories = l.strip().split('\t')[:3]
    valid_categories = [c for c in categories if is_valid_category(c)]
    if len(categories) != len(valid_categories):
      continue
    history = []
    for c in categories:
      cursor = taxo
      for t in history:
        cursor = cursor.get(t, dict())
      if not cursor.get(c):
        cursor[c] = dict()
      history.append(c)

  with open('../data/amazon/taxonomy.small.json', 'w') as taxoFile:
    taxoFile.write(json.dumps(taxo))
