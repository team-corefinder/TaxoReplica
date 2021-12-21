from os import mkdir
from os.path import abspath, dirname, isdir

import gdown

filepath = abspath(__file__)
PROJECT_PATH = dirname(dirname(filepath))

def flatten(list_of_lists):
  return [item for l in list_of_lists for item in l]

def cached_download(url, output):
    if not isdir(output):
      mkdir(output)
      gdown.download_folder(url, output=output)
    else:
      print(f'이미 다운로드되어 있습니다')
