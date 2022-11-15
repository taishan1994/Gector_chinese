import json
from difflib import SequenceMatcher

def get_data(in_path, out_path):
    with open(in_path, "r", encoding="utf-8") as fp:
        data = json.loads(fp.read())

    res = []
    for d in data:
        tmp = {}
        did = d["pid"]
        source = d["source"]
        labels = d["target"]
        tmp["id"] = did
        tmp["source"] = source
        tmp["type"] = "positive" if len(labels) == 0 else "negative"a
        target = [i for i in source]

        new_labels = []
        for label in labels:
            for k, v in label.items():
                if k == "pos":
                    label[k] = int(v)
            new_labels.append(label)
        new_labels = sorted(labels, key=lambda x: x["pos"])

        pre_end = 0
        edit_tmp = []
        for label in new_labels:
            # {"pos": "22", "ori": "熟", "edit": "束", "type": "char"}
            pos = label["pos"]
            ori = label["ori"]
            edit = label["edit"]
            etype = label["type"]
            start = pos
            end = int(pos) + len(ori)
            if etype == "char" or etype == "disorder":
                edit_tmp += target[pre_end:start]
                edit_tmp += [i for i in edit]
                pre_end = end
            elif etype == "miss":
                edit_tmp += target[pre_end:start]
                edit_tmp += [i for i in edit]
                pre_end = end
            elif etype == "redund":
                edit_tmp += target[pre_end:start]
                pre_end = end

        if pre_end <= len(source) - 1:
            edit_tmp += target[pre_end:]

        target = "".join(edit_tmp)
        tmp["target"] = target
        res.append(tmp)

    with open(out_path, "w", encoding="utf-8")  as fp:
        json.dump(res, fp, ensure_ascii=False)


def get_vocab(out_path):
    """
    'replace'：a[i1:i2] 应替换为 b[j1:j2] 。
    'delete'：a[i1:i2] 应该被删除。请注意，在这种情况下，j1 == j2。
    'insert'：b[j1:j2] 应该插入到 a[i1:i1] 。请注意，在这种情况下，i1 == i2。
    'equal'：a[i1:i2] == b[j1:j2](sub-sequences 相等)。
    :return:
    """
    with open("../ctc_vocab/common_tags.txt", "r", encoding="utf-8") as fp:
        tags = fp.read().strip().split("\n")
    with open("train.json", "r", encoding="utf-8") as fp:
        train_data = json.loads(fp.read())
    with open("dev.json", "r", encoding="utf-8") as fp:
        dev_data = json.loads(fp.read())
    data = train_data + dev_data
    src_text = [d["source"] for d in data]
    trg_text = [d["target"] for d in data]
    for src, trg in zip(src_text, trg_text):
        r = SequenceMatcher(None, src, trg)
        diffs = r.get_opcodes()
        for diff in diffs:
            tag, i1, i2, j1, j2 = diff
            labels = None
            if tag == "delete":
                prefix = "$DELETE_"
                labels = src[i1:i2]
            elif tag == "replace":
                prefix = "$REPLACE_"
                labels = trg[j1:j2]
            elif tag == "insert":
                prefix = "$APPEND_"
                labels = trg[j1:j2]
            if labels:
                for label in labels:
                    if label not in tags:
                        tags.append(prefix + label)

    with open(out_path, "w", encoding="utf-8")  as fp:
        fp.write("\n".join(tags))


get_data("train_ori.json", "train.json")
get_data("test_ori.json", "test.json")

get_vocab("../ctc_vocab/ctc_correct_cail2022_tags.txt")