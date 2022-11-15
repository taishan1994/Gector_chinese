import json
from pprint import pprint
from utils.metric_utils import final_f1_score

def get_metric(path):
  with open(path, "r") as fp:
    data = json.loads(fp.read())

  sources = []
  preds = []
  targets = []
  for d in data:
    sources.append(d["source"])
    preds.append(d["pred"])
    targets.append(d["target"])

  pprint(final_f1_score(sources, preds, targets, log_fp='logs/f1_score.log'))