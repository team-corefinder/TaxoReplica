import torch

from lib.roberta import get_roberta


class SimCalc:
  def __init__(self, roberta = None):
    self.roberta = roberta if roberta is not None else get_roberta()

  def similarity(self, text, class_name):
    with torch.inference_mode():
      tokens = self.roberta.encode(text, f'this document is about {class_name.lower()}')
      logits = self.roberta.predict('mnli', tokens[:512], return_logits=True)
      probabilities = logits.softmax(dim=-1).tolist()[0]
      entailment_probability = probabilities[2]
      return entailment_probability
