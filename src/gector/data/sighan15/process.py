import json

with open("../ctc_vocab/common_tags.txt", "r", encoding="utf-8") as fp:
  tags = fp.read().strip().split("\n")

prefix = "$REPLACE_"

def get_data(path, out_path):
  with open(path, "r", encoding="utf-8") as fp:
    data = fp.read().strip().split("\n")
  res = []
  for i,d in enumerate(data):
    tmp = {}
    d = d.split(" ")
    if len(d) == 2:
      source = d[0]
      target = d[1]
      tmp["id"] = i
      tmp["source"] = source
      tmp["target"] = target
      if source == target:
        tmp["type"] = "positive"
      else:
        tmp["type"] = "negative"
      for trg, src in zip(target, source):
        if trg != src:
          tag = prefix + trg
          if tag not in tags:
            tags.append(tag)
      res.append(tmp)

  with open(out_path, "w", encoding="utf-8") as fp:
    json.dump(res, fp, ensure_ascii=False)
    


get_data("train.txt", "train.json")
get_data("test.txt", "test.json")


with open("../ctc_vocab/ctc_correct_sighan15_tags.txt", "w", encoding="utf-8") as fp:
  fp.write("\n".join(tags))