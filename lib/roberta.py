import os, pathlib

import torch

from lib.common import PROJECT_PATH

roberta = None

def get_roberta():
    global roberta
    if roberta is None:
        project_path = pathlib.Path(PROJECT_PATH)
        repo_or_dir = os.path.join(str(project_path.parent), 'fairseq')
        roberta = torch.hub.load(repo_or_dir, 'roberta.large.mnli', source='local')
        try:
            roberta.cuda()
        except Exception as e:
            print(e)
        roberta.eval()  # disable dropout (or leave in train mode to finetune)

    return roberta
