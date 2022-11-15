import json
import os
import logging

import torch
import numpy as np
from torch.cuda.random import device_count
from tqdm import tqdm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from config import Args
from dataset import DatasetCTC
from model import ModelingCtcBert
from utils.utils import set_seed, set_logger
from utils.tokenizer import CtcTokenizer
from utils.train_utils import build_optimizer_and_scheduler
from utils.metric_utils import ctc_f1

args = Args()
set_seed(args.seed)
logger = logging.getLogger(__name__)

if args.use_tensorboard:
    writer = SummaryWriter(log_dir='./tensorboard')


class Trainer:
    def __init__(self,
                 args,
                 train_loader,
                 dev_loader,
                 test_loader,
                 model,
                 device,
                 _loss_ignore_id=-100,
                 _keep_id_in_ctag=1):
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.args = args
        self.model = model
        self.device = device
        self.epochs = self.args.epochs
        self._loss_ignore_id = _loss_ignore_id
        self._keep_id_in_ctag = _keep_id_in_ctag
        if train_loader is not None:
            self.t_total = len(self.train_loader) * self.epochs
            # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.getfloat("train", "learning_rate"))
            self.optimizer, self.scheduler = build_optimizer_and_scheduler(args, model, self.t_total)

    def train(self):
        # Train
        global_step = 1
        best_d_f1 = 0.
        best_c_f1 = 0.
        self.model.zero_grad()
        eval_step = self.args.eval_step  # 每多少个step打印损失及进行验证
        for epoch in range(1, self.epochs + 1):
            for step, batch_data in enumerate(self.train_loader):
                self.model.train()
                for k, v in batch_data.items():
                    batch_data[k] = v.to(self.device)
                detect_labels = batch_data["d_tags"]
                correct_labels = batch_data["c_tags"]
                # 训练过程可能有些许数据出错，跳过
                try:
                    output = self.model(batch_data["input_ids"],
                                        batch_data["attention_mask"],
                                        batch_data["token_type_ids"],
                                        detect_labels,
                                        correct_labels)
                except Exception as e:
                    logger.error('ignore training step error!!')
                    logger.exception(e)
                    continue
                batch_loss = output["loss"]
                batch_loss = batch_loss.mean()
                batch_detect_loss = output["detect_loss"]
                batch_correct_loss = output["correct_loss"]
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.model.zero_grad()
                logger.info('【train】 epoch:{}/{} step:{}/{} detect_loss:{:.6f} correct_loss:{:.6f} loss:{:.6f}'.format(
                    epoch, self.args.epochs, global_step, self.t_total,
                    batch_detect_loss.item(), batch_correct_loss.item(), batch_loss.item()))

                global_step += 1
                if self.args.use_tensorboard:
                    writer.add_scalar('data/detect_loss', batch_detect_loss.item(), global_step)
                    writer.add_scalar('data/correct_loss', batch_correct_loss.item(), global_step)
                    writer.add_scalar('data/loss', batch_loss.item(), global_step)

                if global_step % eval_step == 0:
                    dev_metric = self.dev()
                    d_f1 = dev_metric["d_f1"]
                    c_f1 = dev_metric["c_f1"]
                    logger.info("【dev】 loss:{:.6f} d_precision:{:.4f} d_recall:{:.4f} "
                                "d_f1:{:.4f} c_precision:{:.4f} c_recall:{:.4f} c_f1:{:.4f}".format(
                        dev_metric["loss"], dev_metric["d_precision"], dev_metric["d_recall"],
                        dev_metric["d_f1"], dev_metric["c_precision"], dev_metric["c_recall"],
                        dev_metric["c_f1"]
                    ))
                    if c_f1 > best_c_f1:
                        best_d_f1 = d_f1
                        best_c_f1 = c_f1
                        logger.info("【best】 detect_f1:{:.6f} correct_f1:{:.6f}".format(
                            d_f1, c_f1
                        ))
                        if self.args.use_tensorboard:
                            writer.add_scalar("detect_f1", d_f1)
                            writer.add_scalar("correct_f1", c_f1)
                        if not os.path.exists(self.args.output_dir):
                            os.makedirs(self.args.output_dir, exist_ok=True)

                        # take care of model distributed / parallel training
                        model_to_save = (
                            self.model.module if hasattr(self.model, "module") else self.model
                        )
                        output_dir = os.path.join(self.args.output_dir, self.args.data_name)
                        logger.info('Saving model checkpoint to {}'.format(output_dir))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        torch.save(model_to_save.state_dict(),
                                   os.path.join(output_dir, '{}_model.pt'.format(self.args.model_name)))

    def dev(self):
        self.model.eval()
        preds, gold_labels, src = [], [], []
        losses = 0.
        with torch.no_grad():
            for i, batch_data in enumerate(self.dev_loader):
                for k, v in batch_data.items():
                    batch_data[k] = v.to(self.device)
                detect_labels = batch_data["d_tags"]
                correct_labels = batch_data["c_tags"]
                output = self.model(batch_data["input_ids"],
                                    batch_data["attention_mask"],
                                    batch_data["token_type_ids"],
                                    detect_labels,
                                    correct_labels)
                batch_loss = output["loss"]
                batch_gold = correct_labels.view(-1).cpu().numpy()
                correct_output = output["correct_outputs"]
                batch_pred = torch.argmax(correct_output, dim=-1).view(-1).cpu().numpy()
                batch_src = batch_data['input_ids'].view(-1).cpu().numpy()

                seq_true_idx = np.argwhere(batch_gold != self._loss_ignore_id)  # 获取非pad部分的标签
                batch_gold = batch_gold[seq_true_idx].squeeze()
                batch_pred = batch_pred[seq_true_idx].squeeze()
                batch_src = batch_src[seq_true_idx].squeeze()

                src += list(batch_src)
                gold_labels += list(batch_gold)
                preds += list(batch_pred)
                losses += batch_loss.item()
        "因为输出和输入空间不一样，所以计算指标要对应输出空间，原字符对应输出空间的keep"
        src = [self._keep_id_in_ctag] * len(src)

        (d_precision, d_recall, d_f1), (c_precision, c_recall, c_f1) = ctc_f1(
            src_texts=[src], trg_texts=[gold_labels], pred_texts=[preds])
        result = {
            "loss": losses,
            "c_precision": c_precision,
            "c_recall": c_recall,
            "c_f1": c_f1,
            "d_precision": d_precision,
            "d_recall": d_recall,
            "d_f1": d_f1,
        }
        return result

    def test(self, model):
        model.to(self.device)
        model.eval()
        preds, gold_labels, src = [], [], []
        losses = 0.
        with torch.no_grad():
            for i, batch_data in enumerate(self.test_loader):
                for k, v in batch_data.items():
                    batch_data[k] = v.to(self.device)
                detect_labels = batch_data["d_tags"]
                correct_labels = batch_data["c_tags"]
                output = model(batch_data["input_ids"],
                               batch_data["attention_mask"],
                               batch_data["token_type_ids"],
                               detect_labels,
                               correct_labels)
                batch_loss = output["loss"]
                batch_gold = correct_labels.view(-1).cpu().numpy()
                correct_output = output["correct_outputs"]
                batch_pred = torch.argmax(correct_output, dim=-1).view(-1).cpu().numpy()
                batch_src = batch_data['input_ids'].view(-1).cpu().numpy()

                seq_true_idx = np.argwhere(batch_gold != self._loss_ignore_id)  # 获取非pad部分的标签
                batch_gold = batch_gold[seq_true_idx].squeeze()
                batch_pred = batch_pred[seq_true_idx].squeeze()
                batch_src = batch_src[seq_true_idx].squeeze()

                src += list(batch_src)
                gold_labels += list(batch_gold)
                preds += list(batch_pred)
                losses += batch_loss.item()
        "因为输出和输入空间不一样，所以计算指标要对应输出空间，原字符对应输出空间的keep"
        src = [self._keep_id_in_ctag] * len(src)

        (d_precision, d_recall, d_f1), (c_precision, c_recall, c_f1) = ctc_f1(
            src_texts=[src], trg_texts=[gold_labels], pred_texts=[preds])
        result = {
            "loss": losses,
            "c_precision": c_precision,
            "c_recall": c_recall,
            "c_f1": c_f1,
            "d_precision": d_precision,
            "d_recall": d_recall,
            "d_f1": d_f1,
        }
        return result

    def predict(self,
                text,
                tokenizer,
                model,
                device,
                id2label):
        model.to(device)
        model.eval()
        inputs = tokenizer(text,
                           max_len=self.args.max_seq_len,
                           return_batch=True)
        text = [i for i in text]
        real_length = 1 + len(text)
        with torch.no_grad():
            input_ids = torch.LongTensor(inputs["input_ids"]).to(device)
            real_lenth = input_ids
            attention_mask = torch.LongTensor(inputs["attention_mask"]).to(device)
            token_type_ids = torch.LongTensor(inputs["token_type_ids"]).to(device)
            output = model(input_ids, attention_mask, token_type_ids)
            correct_outputs = output["correct_outputs"]
            correct_outputs = correct_outputs.detach().cpu().numpy()
            detect_outputs = output["detect_outputs"]
            detect_outputs = detect_outputs.detach().cpu().numpy()
            detect_outputs = np.argmax(detect_outputs, axis=-1).squeeze()[:real_length]
            correct_outputs = np.argmax(correct_outputs, axis=-1).squeeze()[:real_length]
            # print(detect_outputs)
            # print(correct_outputs)
            res = {}
            pre_text = []
            for d, c, t in zip(detect_outputs, correct_outputs, ["始"] + text):
                clabel = id2label[c]
                if "APPEND" in clabel:
                    pre_text.append(clabel)
                    insert = clabel.split("_")[-1]
                    pre_text.append(insert)
                elif "DELETE" in clabel:
                    continue
                elif "$REPLACE" in clabel:
                    replace = clabel.split("_")[-1]
                    pre_text.append(replace)
                else:
                    pre_text.append(t)
            res["source"] = "".join(text)
            res["pred"] = "".join(pre_text)[1:]
        return res


def load_texts_from_fp(file_path):
    trg_texts, src_texts = [], []
    if '.txt' in file_path:
        for line in open(file_path, 'r', encoding='utf-8'):
            line = line.strip().split('\t')
            if line:
                # 需注意txt文件中src和trg前后关系
                src_texts.append(line[0])
                trg_texts.append(line[1])
    elif '.json' in file_path:
        json_data = json.load(open(file_path, 'r', encoding='utf-8'))
        for line in tqdm(json_data, ncols=100):
            if isinstance(line['source'], str) and \
                    isinstance(line['target'], str):
                src_texts.append(line['source'])
                trg_texts.append(line['target'])
    return src_texts, trg_texts


dataset_mappings = {
    "midu2022": {
        "train": args.midu_train_path,
        "dev": args.midu_dev_path,
        "correct_tags_file": args.midu_correct_tags_path,
        "detect_tags_file": args.midu_detect_tags_path,
    },
    "sighan13": {
        "train": args.sighan13_train_path,
        "dev": args.sighan13_dev_path,
        "correct_tags_file": args.sighan13_correct_tags_path,
        "detect_tags_file": args.sighan13_detect_tags_path,
    },
    "sighan14": {
        "train": args.sighan14_train_path,
        "dev": args.sighan14_dev_path,
        "correct_tags_file": args.sighan14_correct_tags_path,
        "detect_tags_file": args.sighan14_detect_tags_path,
    },
    "sighan15": {
        "train": args.sighan15_train_path,
        "dev": args.sighan15_dev_path,
        "correct_tags_file": args.sighan15_correct_tags_path,
        "detect_tags_file": args.sighan15_detect_tags_path,
    },
    "sighan15_2": {
        "train": args.sighan15_2_train_path,
        "dev": args.sighan15_2_dev_path,
        "correct_tags_file": args.sighan15_2_correct_tags_path,
        "detect_tags_file": args.sighan15_2_detect_tags_path,
    },
    "cail2022": {
        "train": args.cail2022_train_path,
        "dev": args.cail2022_dev_path,
        "correct_tags_file": args.cail2022_correct_tags_path,
        "detect_tags_file": args.cail2022_detect_tags_path,
    },
}

if __name__ == '__main__':
    # data_name = 'sighan13'
    # model_name = "roberta"
    # args.data_name = data_name
    # args.model_name = model_name
    set_logger(os.path.join(args.log_dir, "main.log"))

    with open(os.path.join(args.ctc_vocab_dir, dataset_mappings[args.data_name]["correct_tags_file"]), "r") as fp:
        vocab_szie = len(fp.read().strip().split("\n"))
    args.correct_vocab_size = vocab_szie

    logger.info("使用模型【{}】，使用数据集：【{}】".format(args.model_name, args.data_name))
    ctc_tokenizer = CtcTokenizer.from_pretrained(args.bert_dir)

    train_src_texts, train_trg_texts = load_texts_from_fp(dataset_mappings[args.data_name]["train"])
    dev_src_texts, dev_trg_texts = load_texts_from_fp(dataset_mappings[args.data_name]["dev"])
    train_dataset = DatasetCTC(in_model_dir=args.bert_dir,
                               src_texts=train_src_texts,
                               trg_texts=train_trg_texts,
                               max_seq_len=args.max_seq_len,
                               ctc_label_vocab_dir=args.ctc_vocab_dir,
                               correct_tags_file=dataset_mappings[args.data_name]["correct_tags_file"],
                               detect_tags_file=dataset_mappings[args.data_name]["detect_tags_file"],
                               _loss_ignore_id=-100)

    dev_dataset = DatasetCTC(in_model_dir=args.bert_dir,
                             src_texts=dev_src_texts,
                             trg_texts=dev_trg_texts,
                             max_seq_len=args.max_seq_len,
                             ctc_label_vocab_dir=args.ctc_vocab_dir,
                             correct_tags_file=dataset_mappings[args.data_name]["correct_tags_file"],
                             detect_tags_file=dataset_mappings[args.data_name]["detect_tags_file"],
                             _loss_ignore_id=-100)

    logger.info("训练集数据：{}条 验证集数据：{}条".format(len(train_dataset), len(dev_dataset)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("使用【{}】".format(device))
    batch_size = args.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    model = ModelingCtcBert(args)
    model.to(device)

    trainer = Trainer(
        args,
        train_loader,
        dev_loader,
        dev_loader,
        model,
        device
    )

    trainer.train()

    id2label = train_dataset.id2ctag
    ckpt_path = "./checkpoints/{}/roberta_model.pt".format(args.data_name)
    model = ModelingCtcBert(args)
    model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu')))

    result = trainer.test(model)
    print(result)

    texts = [
        "应为他学得很好，所以同班同学都喜欢问他问题。",
        "我听说这个礼拜六你要开一个误会。可是那天我会很忙，因为我男朋友要回国来看我。",
        "我以前想要高诉你，可是我忘了。我真户秃。",
        "即然学生在学校老师的照顾下上课，他们的父母不需要一直看。",
        "为了小孩的自由跟安全还有长大得很好，我完全还对我对某一所市立小学校长的建议「用网站看到小孩在教室里的情况」。",
        "我觉得装了电影机是很好的作法，因为这机器有很多好处。",
        "我看建设这种机器是很值得的事情，所有的学校已该建设。",
        "看完那段文张，我是反对的",
        "在三岁时，一场大火将家里烧得面目全飞",
        "希望你生体好",
        '感帽了',
        '你儿字今年几岁了',
        '少先队员因该为老人让坐',
        '随然今天很热',
        '传然给我',
        '呕土不止',
        '哈蜜瓜',
        '广州黄浦',
        '在 上 上面 上面 那 什么 啊',
        '呃 。 呃 ,啊,那用户名称是叫什么呢？',
        '我生病了,咳数了好几天',
        '对京东新人度大打折扣',
        '我想买哥苹果手机',
        "#淮安消防[超话]#长夜漫漫，独愿汝安晚安[心]​​",
        "妻子遭国民党联保“打地雷公”的酷刑，生活无依无靠，沿村乞讨度日。",
        "2.风力预报5日白天起风力逐渐加大，预计5～7日高海拔山区最大风力可达6～7级阵风8～9级。",
    ]

    for text in texts:
        pred_text = trainer.predict(text, ctc_tokenizer, model, device, id2label)
        print("输入：", text)
        print("输出：", pred_text)
        print("=" * 100)

    results = []
    for src, trg in zip(dev_src_texts, dev_trg_texts):
        res_tmp = {}
        res = trainer.predict(src, ctc_tokenizer, model, device, id2label)
        source = res["source"]
        pred = res["pred"]
        target = trg
        res_tmp["source"] = source
        res_tmp["pred"] = pred
        res_tmp["target"] = target
        if source != target:
            res_tmp["type"] = "negative"
        else:
            res_tmp["type"] = "positive"
        results.append(res_tmp)

    result_path = "{}_results.json".format(args.data_name)
    with open(result_path, "w") as fp:
        json.dump(results, fp, ensure_ascii=False)

    from evaluate import get_metric

    get_metric(result_path)


