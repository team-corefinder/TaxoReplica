import json, math, itertools, os, time, logging

import torch
from torch.utils.data import IterableDataset, get_worker_info
from torch.utils.data.dataloader import DataLoader
import torch.autograd.profiler as profiler

from lib.common import PROJECT_PATH
from lib.coreclass import CCMiner

TAXO_PATH_AMAZON = os.path.join(PROJECT_PATH, 'taxonomy/amazon-344.json')
TAXO_PATH_DBPEDIA = os.path.join(PROJECT_PATH, 'data/dbpedia/taxonomy.json')

class JsonDataset(IterableDataset):
  def __init__(self, files):
    self.files = files

  def __iter__(self):
    for json_file in self.files:
      with open(json_file) as f:
        for index, line in enumerate(f):
          review = json.loads(line)
          yield review

class MultiProcessableJsonDataset(JsonDataset):
  def __init__(self, files, start, end):
    super().__init__(files)
    assert end > start, "this example code only works with end >= start"
    self.start = start
    self.end = end

  def __iter__(self):
    worker_info = get_worker_info()
    if worker_info is None:  # single-process data loading, return the full iterator
      print('single process')
      iter_start = self.start
      iter_end = self.end
    else:  # in a worker process
      # split workload
      per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
      worker_id = worker_info.id
      print(f'multi process: {worker_id}')
      iter_start = self.start + worker_id * per_worker
      iter_end = min(iter_start + per_worker, self.end)
    return iter(itertools.islice(super().__iter__(), iter_start, iter_end))
  
def collate_fn(batch):
  taxo_json = dict()
  # TAXO_PATH = TAXO_PATH_AMAZON
  TAXO_PATH = TAXO_PATH_DBPEDIA
  with open(TAXO_PATH) as f:
    taxo_json = json.load(f)
  get_text = lambda d: d.get('reviewText', d.get('text', None))
  ccminer = CCMiner([get_text(d) for d in batch], taxo_json)
  ccminer.find_candidates()
  for i, r in enumerate(batch):
    start = time.time()
    doc = ccminer.documents[i]
    r['coreclasses'] = list(ccminer.coreclasses(doc))
    ccminer.logger.info(str(i) + f' mining core classes took {round(time.time() - start, 3)} seconds')

  return batch

def load_data(filepath, num_data, batch_size=32, num_workers=0):
  dataset = MultiProcessableJsonDataset([filepath], 0, num_data)
  if num_workers > 0:
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn, multiprocessing_context='spawn')
  else:
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn)

