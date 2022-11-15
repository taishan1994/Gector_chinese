#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys

sys.path.append("..")
import torch
from transformers.models.bert import BertModel, BertPreTrainedModel, BertConfig
from torch.nn import CrossEntropyLoss, Module

from gector.layers.loss import LabelSmoothingLoss


class ModelingCtcBert(Module):

    def __init__(self, args):
        super(ModelingCtcBert, self).__init__()
        self.args = args
        bert_config = BertConfig.from_pretrained(args.bert_dir)
        self.bert = BertModel.from_pretrained(args.bert_dir)
        self.dropout = torch.nn.Dropout(bert_config.hidden_dropout_prob)
        self.tag_detect_projection_layer = torch.nn.Linear(
            bert_config.hidden_size, args.detect_vocab_size)
        self.tag_label_projection_layer = torch.nn.Linear(
            bert_config.hidden_size, args.correct_vocab_size)
        self._detect_criterion = CrossEntropyLoss(ignore_index=-100)
        self._correct_criterion = LabelSmoothingLoss(smoothing=0.1, ignore_index=-100)

    @staticmethod
    def build_dummpy_inputs():
        inputs = {}
        inputs['input_ids'] = torch.LongTensor(
            torch.randint(low=1, high=10, size=(8, 56)))
        inputs['attention_mask'] = torch.ones(size=(8, 56)).long()
        inputs['token_type_ids'] = torch.zeros(size=(8, 56)).long()
        inputs['detect_labels'] = torch.zeros(size=(8, 56)).long()
        inputs['correct_labels'] = torch.zeros(size=(8, 56)).long()
        return inputs

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            detect_labels=None,
            correct_labels=None
    ):

        hidden_states = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids)[0]
        detect_outputs = self.tag_detect_projection_layer(hidden_states)
        correct_outputs = self.tag_label_projection_layer(hidden_states)

        result = {
            "detect_outputs": detect_outputs,
            "correct_outputs": correct_outputs,
            "detect_loss": None,
            "correct_loss": None,
            "loss": None,
        }

        loss = None
        if detect_labels is not None and correct_labels is not None:
            detect_loss = self._detect_criterion(
                detect_outputs.view(-1, self.args.detect_vocab_size), detect_labels.view(-1))
            correct_loss = self._correct_criterion(
                correct_outputs.view(-1, self.args.correct_vocab_size), correct_labels.view(-1))
            loss = detect_loss + correct_loss
            result["detect_loss"] = detect_loss
            result["correct_loss"] = correct_loss

        elif detect_labels is not None:
            loss = self._detect_criterion(
                detect_outputs.view(-1, self.args.detect_vocab_size), detect_labels.view(-1))
        elif correct_labels is not None:
            loss = self._correct_criterion(
                correct_outputs.view(-1, self.args.correct_vocab_size), correct_labels.view(-1))

        result["loss"] = loss
        return result


if __name__ == '__main__':
    class Args:
       bert_dir = "../model_hub/chinese-bert-wwm-ext"
       detect_vocab_size = 2
       correct_vocab_size = 20675

    args = Args()
    ctc = ModelingCtcBert(args)
    inputs = ctc.build_dummpy_inputs()
    input_ids = inputs["input_ids"]
    attention_mask = inputs['attention_mask']
    token_type_ids = inputs['token_type_ids']
    detect_labels = inputs['detect_labels']
    correct_labels = inputs['correct_labels']

    output = ctc(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        detect_labels=detect_labels,
        correct_labels=correct_labels
    )
    print(output)