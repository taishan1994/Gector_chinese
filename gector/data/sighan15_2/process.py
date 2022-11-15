import json

with open("../ctc_vocab/common_tags.txt", "r", encoding="utf-8") as fp:
  tags = fp.read().strip().split("\n")

prefix = "$REPLACE_"

def get_data(path, out_path):
  with open(path, "r", encoding="utf-8") as fp:
    data = fp.read().strip().split("\n")
  res = []
  for i in range(0, len(data), 2):
    tmp = {}
    source = data[i]
    tmp["id"] = i
    tmp["source"] = source
    labels = data[i+1]
    if not labels.strip() or labels == "0":
      tmp["type"] = "positive"
      tmp["target"] = source
      res.append(tmp)
      continue 
    else:
      tmp["type"] = "negative"
    labels = labels.split(";")
    labels = [label.split(",") for label in labels if label]
    target = [i for i in source]
    for label in labels:
      ind = label[0]
      t = label[2]
      target[int(ind)-1] = t
      tag = prefix + t
      if tag not in tags:
        tags.append(tag)
    target = "".join(target)
    tmp["target"] = target
   
    res.append(tmp)

  with open(out_path, "w", encoding="utf-8") as fp:
    json.dump(res, fp, ensure_ascii=False)
    


get_data("train.txt", "train.json")
get_data("test.txt", "test.json")


with open("../ctc_vocab/ctc_correct_sighan15_tags.txt", "w", encoding="utf-8") as fp:
  fp.write("\n".join(tags))