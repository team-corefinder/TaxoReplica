import json, random

if __name__ == '__main__':
  with open('data/amazon/train.json') as f:
    reviews = json.load(f)
    samples = random.choices(reviews, k=1000)

    with open('data/amazon/train-1000.jsonl', 'w') as ff:
      ff.write('\n'.join([json.dumps(r) for r in samples]))
