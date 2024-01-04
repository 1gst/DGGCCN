import random

import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import load_dataset, DownloadMode
from transformers import AutoTokenizer, AutoConfig, BertTokenizer, BertTokenizerFast

from utils.pub_utils import GlobalVariable

# from utils.triple_extractor import TripleExtractor

bio_labels = [
    "O",
    "B-明知",
    "I-明知",
    "B-投案",
    "I-投案",
    "B-供述",
    "I-供述",
    "B-谅解",
    "I-谅解",
    "B-赔偿",
    "I-赔偿",
    "B-退赃",
    "I-退赃",
    "B-销赃",
    "I-销赃",
    "B-分赃",
    "I-分赃",
    "B-搜查/扣押",
    "I-搜查/扣押",
    "B-举报",
    "I-举报",
    "B-拘捕",
    "I-拘捕",
    "B-报警/报案",
    "I-报警/报案",
    "B-鉴定",
    "I-鉴定",
    "B-冲突",
    "I-冲突",
    "B-言语冲突",
    "I-言语冲突",
    "B-肢体冲突",
    "I-肢体冲突",
    "B-买卖",
    "I-买卖",
    "B-卖出",
    "I-卖出",
    "B-买入",
    "I-买入",
    "B-租/借",
    "I-租/借",
    "B-出租/出借",
    "I-出租/出借",
    "B-租用/借用",
    "I-租用/借用",
    "B-归还/偿还",
    "I-归还/偿还",
    "B-获利",
    "I-获利",
    "B-雇佣",
    "I-雇佣",
    "B-放贷",
    "I-放贷",
    "B-集资",
    "I-集资",
    "B-支付/给付",
    "I-支付/给付",
    "B-签订合同/订立协议",
    "I-签订合同/订立协议",
    "B-制造",
    "I-制造",
    "B-遗弃",
    "I-遗弃",
    "B-运输/运送",
    "I-运输/运送",
    "B-邮寄",
    "I-邮寄",
    "B-组织/安排",
    "I-组织/安排",
    "B-散布",
    "I-散布",
    "B-联络",
    "I-联络",
    "B-通知/提醒",
    "I-通知/提醒",
    "B-介绍/引荐",
    "I-介绍/引荐",
    "B-邀请/招揽",
    "I-邀请/招揽",
    "B-纠集",
    "I-纠集",
    "B-阻止/妨碍",
    "I-阻止/妨碍",
    "B-挑衅/挑拨",
    "I-挑衅/挑拨",
    "B-帮助/救助",
    "I-帮助/救助",
    "B-提供",
    "I-提供",
    "B-放纵",
    "I-放纵",
    "B-跟踪",
    "I-跟踪",
    "B-同意/接受",
    "I-同意/接受",
    "B-拒绝/抗拒",
    "I-拒绝/抗拒",
    "B-放弃/停止",
    "I-放弃/停止",
    "B-要求/请求",
    "I-要求/请求",
    "B-建议",
    "I-建议",
    "B-约定",
    "I-约定",
    "B-饮酒",
    "I-饮酒",
    "B-自然灾害",
    "I-自然灾害",
    "B-洪涝",
    "I-洪涝",
    "B-干旱",
    "I-干旱",
    "B-山体滑坡",
    "I-山体滑坡",
    "B-事故",
    "I-事故",
    "B-交通事故",
    "I-交通事故",
    "B-火灾事故",
    "I-火灾事故",
    "B-爆炸事故",
    "I-爆炸事故",
    "B-暴力",
    "I-暴力",
    "B-杀害",
    "I-杀害",
    "B-伤害人身",
    "I-伤害人身",
    "B-言语辱骂",
    "I-言语辱骂",
    "B-敲诈勒索",
    "I-敲诈勒索",
    "B-威胁/强迫",
    "I-威胁/强迫",
    "B-持械/持枪",
    "I-持械/持枪",
    "B-拘束/拘禁",
    "I-拘束/拘禁",
    "B-绑架",
    "I-绑架",
    "B-欺骗",
    "I-欺骗",
    "B-拐骗",
    "I-拐骗",
    "B-冒充",
    "I-冒充",
    "B-伪造",
    "I-伪造",
    "B-变造",
    "I-变造",
    "B-盗窃财物",
    "I-盗窃财物",
    "B-抢夺财物",
    "I-抢夺财物",
    "B-抢劫财物",
    "I-抢劫财物",
    "B-挪用财物",
    "I-挪用财物",
    "B-侵占财物",
    "I-侵占财物",
    "B-毁坏财物",
    "I-毁坏财物",
    "B-猥亵",
    "I-猥亵",
    "B-强奸",
    "I-强奸",
    "B-卖淫",
    "I-卖淫",
    "B-嫖娼",
    "I-嫖娼",
    "B-吸毒",
    "I-吸毒",
    "B-贩卖毒品",
    "I-贩卖毒品",
    "B-赌博",
    "I-赌博",
    "B-开设赌场",
    "I-开设赌场",
    "B-指使/教唆",
    "I-指使/教唆",
    "B-共谋",
    "I-共谋",
    "B-违章驾驶",
    "I-违章驾驶",
    "B-泄露信息",
    "I-泄露信息",
    "B-私藏/藏匿",
    "I-私藏/藏匿",
    "B-入室/入户",
    "I-入室/入户",
    "B-贿赂",
    "I-贿赂",
    "B-逃匿",
    "I-逃匿",
    "B-放火",
    "I-放火",
    "B-走私",
    "I-走私",
    "B-投毒",
    "I-投毒",
    "B-自杀",
    "I-自杀",
    "B-死亡",
    "I-死亡",
    "B-受伤",
    "I-受伤",
    "B-被困",
    "I-被困",
    "B-中毒",
    "I-中毒",
    "B-昏迷",
    "I-昏迷",
    "B-遗失",
    "I-遗失",
    "B-受损",
    "I-受损"
]
pure_event2id = {
    "None": 0,
    "明知": 1,
    "投案": 2,
    "供述": 3,
    "谅解": 4,
    "赔偿": 5,
    "退赃": 6,
    "销赃": 7,
    "分赃": 8,
    "搜查/扣押": 9,
    "举报": 10,
    "拘捕": 11,
    "报警/报案": 12,
    "鉴定": 13,
    "冲突": 14,
    "言语冲突": 15,
    "肢体冲突": 16,
    "买卖": 17,
    "卖出": 18,
    "买入": 19,
    "租/借": 20,
    "出租/出借": 21,
    "租用/借用": 22,
    "归还/偿还": 23,
    "获利": 24,
    "雇佣": 25,
    "放贷": 26,
    "集资": 27,
    "支付/给付": 28,
    "签订合同/订立协议": 29,
    "制造": 30,
    "遗弃": 31,
    "运输/运送": 32,
    "邮寄": 33,
    "组织/安排": 34,
    "散布": 35,
    "联络": 36,
    "通知/提醒": 37,
    "介绍/引荐": 38,
    "邀请/招揽": 39,
    "纠集": 40,
    "阻止/妨碍": 41,
    "挑衅/挑拨": 42,
    "帮助/救助": 43,
    "提供": 44,
    "放纵": 45,
    "跟踪": 46,
    "同意/接受": 47,
    "拒绝/抗拒": 48,
    "放弃/停止": 49,
    "要求/请求": 50,
    "建议": 51,
    "约定": 52,
    "饮酒": 53,
    "自然灾害": 54,
    "洪涝": 55,
    "干旱": 56,
    "山体滑坡": 57,
    "事故": 58,
    "交通事故": 59,
    "火灾事故": 60,
    "爆炸事故": 61,
    "暴力": 62,
    "杀害": 63,
    "伤害人身": 64,
    "言语辱骂": 65,
    "敲诈勒索": 66,
    "威胁/强迫": 67,
    "持械/持枪": 68,
    "拘束/拘禁": 69,
    "绑架": 70,
    "欺骗": 71,
    "拐骗": 72,
    "冒充": 73,
    "伪造": 74,
    "变造": 75,
    "盗窃财物": 76,
    "抢夺财物": 77,
    "抢劫财物": 78,
    "挪用财物": 79,
    "侵占财物": 80,
    "毁坏财物": 81,
    "猥亵": 82,
    "强奸": 83,
    "卖淫": 84,
    "嫖娼": 85,
    "吸毒": 86,
    "贩卖毒品": 87,
    "赌博": 88,
    "开设赌场": 89,
    "指使/教唆": 90,
    "共谋": 91,
    "违章驾驶": 92,
    "泄露信息": 93,
    "私藏/藏匿": 94,
    "入室/入户": 95,
    "贿赂": 96,
    "逃匿": 97,
    "放火": 98,
    "走私": 99,
    "投毒": 100,
    "自杀": 101,
    "死亡": 102,
    "受伤": 103,
    "被困": 104,
    "中毒": 105,
    "昏迷": 106,
    "遗失": 107,
    "受损": 108
}


class CAILDataset():
    def __init__(self, max_len, tokenizer_path, train_data_path=None, dev_data_path=None, test_data_path=None):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.train_data_path = train_data_path
        self.dev_data_path = dev_data_path
        self.test_data_path = test_data_path
        self.mode = 'train'
        self.max_len = max_len
        self.gp_mask = True
        self.tokenize_function = self.tokenize_function_gp_gccn
        if self.tokenize_function == self.tokenize_function_gp_gccn:
            self.CreateDisInputs() #没有用上
        self.get_event_input()

    def get_event_input(self):
        with open('corpus/type_match_v3.json', 'r', encoding='GBK') as f:
            type_match_dict = json.load(f)
        event_text = []
        # 类别名加上类别解释
        with open('corpus/type_explain.json', 'r', encoding='UTF-8') as f:
            type_match_dict = json.load(f)
        for key, value in pure_event2id.items():
            key = type_match_dict['type_' + str(value)]
            event_text.append(key)
        event_input = self.tokenizer(event_text[1:], padding=True, truncation=True, max_length=150)
        # 类别名加上此类触发词
        # for key, value in pure_event2id.items():
        #     for word in type_match_dict['type_' + str(value)]:
        #         if len(key + "," + word) > 100:
        #             break
        #         key = key + "," + word
        #     event_text.append(key)
        # event_input = self.tokenizer(event_text[1:], padding=True, truncation=True, max_length=100)
        # 只有类别名
        # event_text = []
        # for key, value in pure_event2id.items():
        #     event_text.append(key)
        # event_input = self.tokenizer(event_text[1:], padding=True)
        for k, v in event_input.items():
            event_input[k] = torch.tensor(v, dtype=torch.int64, device="cuda:0")
        GlobalVariable.event_text_input = event_input

    # 加载训练数据集
    def CreateDisInputs(self):
        # 距离矩阵索引
        dis2idx = np.zeros((1000), dtype='int64')
        dis2idx[1] = 1
        dis2idx[2:] = 2
        dis2idx[4:] = 3
        dis2idx[8:] = 4
        dis2idx[16:] = 5
        dis2idx[32:] = 6
        dis2idx[64:] = 7
        dis2idx[128:] = 8
        dis2idx[256:] = 9
        # 构造一个dist_inputs
        self.dist_inputs = torch.zeros((512, 512), dtype=torch.int8)
        for k in range(512):
            self.dist_inputs[k, :] += k
            self.dist_inputs[:, k] -= k

        for i in range(512):
            for j in range(512):
                if self.dist_inputs[i, j] < 0:
                    self.dist_inputs[i, j] = dis2idx[-self.dist_inputs[i, j]] + 9
                else:
                    self.dist_inputs[i, j] = dis2idx[self.dist_inputs[i, j]]
        self.dist_inputs[self.dist_inputs == 0] = 19

    def LoadTrainDataset(self):
        self.mode = 'train'
        # 加载训练和验证的原始数据，成为一个DatasetDict对象，里面的每一个数据集为Dataset可迭代对象
        train_datasets = load_dataset("json", data_files={"train": self.train_data_path}, split='train[:]',
                                      cache_dir="opt/ml/input")
        # tokenizer映射到原始数据上
        tokenized_train_datasets = train_datasets.map(self.tokenize_function, load_from_cache_file=False,
                                                      remove_columns=train_datasets.column_names, batched=True)
        return tokenized_train_datasets

        # 加载验证数据集

    def LoadDevDataset(self):
        self.mode = 'dev'
        # 加载训练和验证的原始数据，成为一个DatasetDict对象，里面的每一个数据集为Dataset可迭代对象
        dev_datasets = load_dataset("json", data_files={"validation": self.dev_data_path}, split='validation[:]',
                                    cache_dir="opt/ml/input")
        # tokenizer映射到原始数据上
        tokenized_dev_datasets = dev_datasets.map(self.tokenize_function, load_from_cache_file=False,
                                                  remove_columns=dev_datasets.column_names, batched=True,
                                                  keep_in_memory=True)
        return tokenized_dev_datasets

        # 加载测试数据集

    def LoadTestDataset(self):
        self.mode = 'test'
        # 加载测试的原始数据，成为一个DatasetDict对象，里面的每一个数据集为Dataset可迭代对象
        test_datasets = load_dataset("json", data_files={"test": self.test_data_path}, split='test[:]',
                                     cache_dir="opt/ml/input")
        with open('corpus/type_match1.json', 'r', encoding='GBK') as f:
            self.type_match_dict = json.load(f)
            self.all_trig = []
            for key, value in self.type_match_dict.items():
                for tirg in value:
                    self.all_trig.append(tirg)
        # tokenizer映射到原始数据上
        tokenized_test_datasets = test_datasets.map(self.tokenize_function, load_from_cache_file=False,
                                                    remove_columns=test_datasets.column_names, batched=True)
        return tokenized_test_datasets

    # 获取分词器
    def get_tokenizer(self):
        return self.tokenizer

    # 看做序列标注任务需要对数据处理的方法
    def tokenize_function_seq(self, examples):
        tokenize_example = {}
        tokenize_example['words'] = []
        tokenize_example['labels'] = []
        tokenize_example['label_ids'] = []
        temp_labels = []
        temp_words = []
        type_labels = []
        # 提取数据
        for contents in examples['content']:
            words = [c['tokens'] for c in contents]
            temp_words.append(words)
            labels = [['O'] * len(c['tokens']) for c in contents]
            temp_labels.append(labels)
        # 构造label标签
        if self.mode != 'test':
            for id, events, mentions, labels in zip(examples['id'], examples['events'], examples['negative_triggers'],
                                                    temp_labels):
                for event in events:
                    for mention in event['mention']:
                        labels[mention['sent_id']][mention['offset'][0]] = "B-" + event['type']
                        for i in range(mention['offset'][0] + 1, mention['offset'][1]):
                            labels[mention['sent_id']][i] = "I-" + event['type']
                for mention in mentions:
                    labels[mention['sent_id']][mention['offset'][0]] = "O"
                    for i in range(mention['offset'][0] + 1, mention['offset'][1]):
                        labels[mention['sent_id']][i] = "O"
                type_labels.append(labels)
            # 把words和labels追加到列表中
            for words, labels in zip(temp_words, type_labels):
                for i in range(0, len(words)):
                    tokenize_example['words'].append(words[i])
                    tokenize_example['labels'].append(labels[i])
        else:
            for words, labels in zip(temp_words, temp_labels):
                for i in range(0, len(words)):
                    tokenize_example['words'].append(words[i])
                    tokenize_example['labels'].append(labels[i])
        # 我们有两种对齐label的方式：
        # 多个subtokens对齐一个word，对齐一个label
        # 多个subtokens的第一个subtoken对齐word，对齐一个label，其他subtokens直接赋予 - 100.
        # 我们提供这两种方式，通过label_all_tokens = True切换。
        label_all_tokens = True
        # 转换labels
        label_map = {label: i for i, label in enumerate(bio_labels)}
        for labels in tokenize_example['labels']:
            tokenize_example['label_ids'].append([label_map[label] for label in labels])
        # 开始处理输入的数据
        tokenized_inputs = self.tokenizer(tokenize_example["words"], truncation=True, padding='max_length',
                                          max_length=self.max_len,
                                          is_split_into_words=True)
        labels = []
        for i, label in enumerate(tokenize_example['label_ids']):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(label[word_idx] if label_all_tokens else -100)
                previous_word_idx = word_idx

            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        if self.mode == 'test':
            tokenized_inputs['word_ids'] = []
            for i, label in enumerate(tokenize_example['label_ids']):
                word_ids = []
                for word_id in tokenized_inputs.word_ids(batch_index=i):
                    if word_id != None:
                        word_ids.append(word_id)
                    else:
                        word_ids.append(-1)
                tokenized_inputs['word_ids'].append(word_ids)
        return tokenized_inputs

    # 在使用序列标注任务的基础上使用罪名及其解释增强数据
    def tokenize_function_enhance(self, examples):
        """
        加入罪名相关的信息
        Args:
            examples:

        Returns:

        """
        tokenize_example = {}
        tokenize_example['words'] = []
        tokenize_example['labels'] = []
        tokenize_example['label_ids'] = []
        tokenize_example['crime'] = []
        temp_labels = []
        temp_words = []
        type_labels = []
        crime_path = os.path.split(self.train_data_path)[0] + '/crime.json'
        with open(crime_path, 'r', encoding='UTF-8') as f:
            crime_dic = json.load(f)
        # 提取数据
        for contents in examples['content']:
            words = [c['tokens'] for c in contents]
            temp_words.append(words)
            labels = [['O'] * len(c['tokens']) for c in contents]
            temp_labels.append(labels)
        for contents, crime in zip(examples['content'], examples['crime']):
            for c in contents:
                crimes = [crime_dic[crime]]
                tokenize_example['crime'].append(crimes)
        # 构造label标签
        if self.mode != 'test':
            for crime, events, mentions, labels in zip(examples['crime'], examples['events'],
                                                       examples['negative_triggers'],
                                                       temp_labels):
                for event in events:
                    for mention in event['mention']:
                        labels[mention['sent_id']][mention['offset'][0]] = "B-" + event['type']
                        for i in range(mention['offset'][0] + 1, mention['offset'][1]):
                            labels[mention['sent_id']][i] = "I-" + event['type']
                for mention in mentions:
                    labels[mention['sent_id']][mention['offset'][0]] = "O"
                    for i in range(mention['offset'][0] + 1, mention['offset'][1]):
                        labels[mention['sent_id']][i] = "O"
                type_labels.append(labels)
            # 把words和labels追加到列表中
            for words, labels in zip(temp_words, type_labels):
                for i in range(0, len(words)):
                    tokenize_example['words'].append(words[i])
                    tokenize_example['labels'].append(labels[i])
        else:
            for words, labels in zip(temp_words, temp_labels):
                for i in range(0, len(words)):
                    tokenize_example['words'].append(words[i])
                    tokenize_example['labels'].append(labels[i])
        # 我们有两种对齐label的方式：
        # 多个subtokens对齐一个word，对齐一个label
        # 多个subtokens的第一个subtoken对齐word，对齐一个label，其他subtokens直接赋予 - 100.
        # 我们提供这两种方式，通过label_all_tokens = True切换。
        label_all_tokens = True
        # 转换labels
        label_map = {label: i for i, label in enumerate(bio_labels)}
        for labels in tokenize_example['labels']:
            tokenize_example['label_ids'].append([label_map[label] for label in labels])
        # 开始处理输入的数据
        tokenized_inputs = self.tokenizer(tokenize_example["words"], tokenize_example['crime'], truncation=True,
                                          padding='max_length',
                                          max_length=self.max_len,
                                          is_split_into_words=True)
        labels = []
        for i, label in enumerate(tokenize_example['label_ids']):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            is_next_none = 0
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                # 如果是第二个None，那么后面就是拼接的罪名解释，不需要标签
                if is_next_none == 2:
                    label_ids.append(-100)
                elif word_idx is None:
                    is_next_none += 1
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(label[word_idx] if label_all_tokens else -100)
                previous_word_idx = word_idx

            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        if self.mode == 'test':
            tokenized_inputs['word_ids'] = []
            for i, label in enumerate(tokenize_example['label_ids']):
                word_ids = []
                is_next_none = 0
                for word_id in tokenized_inputs.word_ids(batch_index=i):
                    if is_next_none == 2:
                        word_ids.append(-1)
                    elif word_id != None:
                        word_ids.append(word_id)
                    else:
                        is_next_none += 1
                        word_ids.append(-1)
                tokenized_inputs['word_ids'].append(word_ids)
        return tokenized_inputs

    # 使用global pointer时需要对数据处理的方法
    def tokenize_function_gp(self, examples):
        # 构造一个返回的字典
        tokenized_inputs = {}
        tokenized_inputs['input_ids'] = []
        tokenized_inputs['token_type_ids'] = []
        tokenized_inputs['attention_mask'] = []
        tokenized_inputs['word_ids'] = []
        # tokenized_inputs['sentence']=[]
        # 如果不是训练集需要构造一个传入global pointer中的mask
        if self.mode != 'test':
            tokenized_inputs['labels'] = []
            if self.gp_mask:
                tokenized_inputs['gp_masks'] = []
            # 提取数据
            # 处理每一个案件对应的内容
            for one_contents, one_events, one_negative_triggers in zip(examples['content'], examples['events'],
                                                                       examples['negative_triggers']):
                # 临时保存每个案件的数据
                temp_example = {}
                temp_example['input_ids'] = []
                temp_example['token_type_ids'] = []
                temp_example['attention_mask'] = []
                temp_example['word_ids'] = []
                temp_example['labels'] = []
                temp_example['gp_masks'] = []
                # 处理的是每一个contents，contents中包含了这个案件的所有句子内容，content是一个句子的内容
                for content in one_contents:
                    tokenize_example = self.tokenizer(content["tokens"], truncation=True, max_length=self.max_len,
                                                      is_split_into_words=True)
                    temp_example['input_ids'].append(tokenize_example.input_ids)
                    temp_example['token_type_ids'].append(tokenize_example.token_type_ids)
                    temp_example['attention_mask'].append(tokenize_example.attention_mask)
                    # 处理word_ids，将None替换成-1
                    word_ids = []
                    for word_id in tokenize_example.word_ids():
                        if word_id != None:
                            word_ids.append(word_id)
                        else:
                            word_ids.append(-1)
                    temp_example['word_ids'].append(word_ids)
                    # 先构造一个全为0的label标签
                    temp_example['labels'].append(torch.zeros((len(word_ids), len(word_ids)), dtype=torch.int8))
                    # 构造一个全为0的gp_mask
                    temp_example['gp_masks'].append(torch.zeros((len(word_ids)), dtype=torch.int8))
                    # 把确定的处理好的值赋给返回的字典
                    # tokenized_inputs['sentence'].append(content['sentence'])
                    tokenized_inputs['input_ids'].append(tokenize_example.input_ids)
                    tokenized_inputs['token_type_ids'].append(tokenize_example.token_type_ids)
                    tokenized_inputs['attention_mask'].append(tokenize_example.attention_mask)
                    tokenized_inputs['word_ids'].append(word_ids)

                # 开始处理每个案件的所有触发词,events包括了这个案件的所有触发词，event是一个触发词事件提及
                labels = temp_example['labels']
                gp_masks = temp_example['gp_masks']
                # 在原来全为0的labels中构造类别id
                for event in one_events:
                    sent_id = event['mention'][0]['sent_id']
                    offset = event['mention'][0]['offset']
                    type_id = event['type_id']
                    word_id = temp_example['word_ids'][sent_id]
                    if offset[0] <= max(word_id):
                        start = word_id.index(offset[0])
                        end = start + word_id.count(offset[0]) - 1
                        labels[sent_id][start][end] = type_id
                        for i in range(start, end + 1):
                            gp_masks[sent_id][i] = 1

                # 同时构造一个传入global pointer的mask
                for negative_trigger in one_negative_triggers:
                    sent_id = negative_trigger['sent_id']
                    offset = negative_trigger['offset']
                    word_id = temp_example['word_ids'][sent_id]
                    if offset[0] <= max(word_id):
                        start = word_id.index(offset[0])
                        end = start + word_id.count(offset[0]) - 1
                        for i in range(start, end + 1):
                            gp_masks[sent_id][i] = 1

                # 把处理好的label添加到返回字典中
                for label, gp_mask in zip(labels, gp_masks):
                    tokenized_inputs["labels"].append(label)
                    if self.gp_mask:
                        tokenized_inputs["gp_masks"].append(gp_mask)
        else:
            if self.gp_mask:
                tokenized_inputs['gp_masks'] = []
            # 提取数据
            # 处理每一个案件对应的内容
            for one_contents, one_candidates in zip(examples['content'], examples['candidates']):
                temp_example = {}
                temp_example['word_ids'] = []
                temp_example['gp_masks'] = []
                # 处理的是每一个contents，contents中包含了这个案件的所有句子内容，content是一个句子的内容
                for content in one_contents:
                    tokenize_example = self.tokenizer(content["tokens"], truncation=True, max_length=self.max_len,
                                                      is_split_into_words=True)
                    # 处理word_ids，将None替换成-1
                    word_ids = []
                    for word_id in tokenize_example.word_ids():
                        if word_id != None:
                            word_ids.append(word_id)
                        else:
                            word_ids.append(-1)
                    temp_example['word_ids'].append(word_ids)
                    # 构造一个全为0的gp_mask
                    temp_example['gp_masks'].append(torch.zeros((len(word_ids)), dtype=torch.int8))
                    # 把确定的处理好的值赋给返回的字典
                    tokenized_inputs['input_ids'].append(tokenize_example.input_ids)
                    tokenized_inputs['token_type_ids'].append(tokenize_example.token_type_ids)
                    tokenized_inputs['attention_mask'].append(tokenize_example.attention_mask)
                    tokenized_inputs['word_ids'].append(word_ids)
                gp_masks = temp_example['gp_masks']
                # 在原来全为0的labels中构造类别id
                for candidate in one_candidates:
                    sent_id = candidate['sent_id']
                    offset = candidate['offset']
                    word_id = temp_example['word_ids'][sent_id]
                    if offset[0] <= max(word_id):
                        start = word_id.index(offset[0])
                        end = start + word_id.count(offset[0]) - 1
                        for i in range(start, end + 1):
                            gp_masks[sent_id][i] = 1
                # 添加gp_mask
                if self.gp_mask:
                    for gp_mask in gp_masks:
                        tokenized_inputs["gp_masks"].append(gp_mask)
        return tokenized_inputs

    # 在使用global pinter的基础上使用罪名及其解释增强数据
    def tokenize_function_gp_enhance(self, examples):
        crime_path = os.path.split(self.train_data_path)[0] + '/crime.json'
        with open(crime_path, 'r', encoding='UTF-8') as f:
            crime_dic = json.load(f)
        # 实例化一个三元组抽取器
        # extractor = TripleExtractor()
        # 构造一个返回的字典
        tokenized_inputs = {}
        tokenized_inputs['input_ids'] = []
        tokenized_inputs['token_type_ids'] = []
        tokenized_inputs['attention_mask'] = []
        tokenized_inputs['word_ids'] = []
        # tokenized_inputs['sentence']=[]
        # 如果不是训练集需要构造一个传入global pointer中的mask
        if self.mode != 'test':
            tokenized_inputs['labels'] = []
            if self.gp_mask:
                tokenized_inputs['gp_masks'] = []
            # 提取数据
            # 处理每一个案件对应的内容
            for crime, one_contents, one_events, one_negative_triggers in zip(examples['crime'], examples['content'],
                                                                              examples['events'],
                                                                              examples['negative_triggers']):
                crime_text = crime_dic[crime]
                # 临时保存每个案件的数据
                temp_example = {}
                temp_example['input_ids'] = []
                temp_example['token_type_ids'] = []
                temp_example['attention_mask'] = []
                temp_example['word_ids'] = []
                temp_example['labels'] = []
                temp_example['gp_masks'] = []
                # 处理的是每一个contents，contents中包含了这个案件的所有句子内容，content是一个句子的内容
                for content in one_contents:
                    # triples = sum(extractor.triples_main(content["sentence"]), [])
                    tokenize_example = self.tokenizer(content["tokens"], crime_text, truncation=True,
                                                      max_length=self.max_len,
                                                      is_split_into_words=True)
                    temp_example['input_ids'].append(tokenize_example.input_ids)
                    temp_example['token_type_ids'].append(tokenize_example.token_type_ids)
                    temp_example['attention_mask'].append(tokenize_example.attention_mask)
                    # 处理word_ids，将None替换成-1
                    word_ids = []
                    for word_id in tokenize_example.word_ids():
                        if word_id != None:
                            word_ids.append(word_id)
                        else:
                            word_ids.append(-1)
                    temp_example['word_ids'].append(word_ids)
                    # 先构造一个全为0的label标签
                    temp_example['labels'].append(torch.zeros((len(word_ids), len(word_ids)), dtype=torch.int8))
                    # 构造一个gp_mask
                    temp_example['gp_masks'].append(
                        torch.ones((len(word_ids)), dtype=torch.int8) - torch.tensor(tokenize_example.token_type_ids,
                                                                                     dtype=torch.int8))
                    # 把确定的处理好的值赋给返回的字典
                    # tokenized_inputs['sentence'].append(content['sentence'])
                    tokenized_inputs['input_ids'].append(tokenize_example.input_ids)
                    tokenized_inputs['token_type_ids'].append(tokenize_example.token_type_ids)
                    tokenized_inputs['attention_mask'].append(tokenize_example.attention_mask)
                    tokenized_inputs['word_ids'].append(word_ids)

                # 开始处理每个案件的所有触发词,events包括了这个案件的所有触发词，event是一个触发词事件提及
                labels = temp_example['labels']
                gp_masks = temp_example['gp_masks']
                # 在原来全为0的labels中构造类别id
                for event in one_events:
                    sent_id = event['mention'][0]['sent_id']
                    offset = event['mention'][0]['offset']
                    type_id = event['type_id']
                    word_id = temp_example['word_ids'][sent_id]
                    if offset[0] <= max(word_id):
                        start = word_id.index(offset[0])
                        end = start + word_id.count(offset[0]) - 1
                        labels[sent_id][start][end] = type_id

                # 把处理好的label添加到返回字典中
                for label, gp_mask in zip(labels, gp_masks):
                    tokenized_inputs["labels"].append(label)
                    if self.gp_mask:
                        tokenized_inputs["gp_masks"].append(gp_mask)
        else:
            # 提取数据
            # 处理每一个案件对应的内容
            for crime, one_contents, one_candidates in zip(examples['crime'], examples['content'],
                                                           examples['candidates']):
                crime_text = crime_dic[crime]
                # 处理的是每一个contents，contents中包含了这个案件的所有句子内容，content是一个句子的内容
                for content in one_contents:
                    tokenize_example = self.tokenizer(content["tokens"], crime_text, truncation=True,
                                                      max_length=self.max_len,
                                                      is_split_into_words=True)
                    # 处理word_ids，将None替换成-1
                    word_ids = []
                    for word_id in tokenize_example.word_ids():
                        if word_id != None:
                            word_ids.append(word_id)
                        else:
                            word_ids.append(-1)
                    # 把确定的处理好的值赋给返回的字典
                    tokenized_inputs['input_ids'].append(tokenize_example.input_ids)
                    tokenized_inputs['token_type_ids'].append(tokenize_example.token_type_ids)
                    tokenized_inputs['attention_mask'].append(tokenize_example.attention_mask)
                    tokenized_inputs['word_ids'].append(word_ids)
                    # 构造一个gp_mask
                    if self.gp_mask:
                        tokenized_inputs['gp_masks'].append(
                            torch.ones((len(word_ids)), dtype=torch.int8) - torch.tensor(
                                tokenize_example.token_type_ids,
                                dtype=torch.int8))
        return tokenized_inputs

    # 在使用global pinter的基础上添加了触发词的相关信息，用于类型判断的后处理
    def tokenize_function_gp_trig(self, examples):
        # 构造一个返回的字典
        tokenized_inputs = {}
        tokenized_inputs['input_ids'] = []
        tokenized_inputs['token_type_ids'] = []
        tokenized_inputs['attention_mask'] = []
        tokenized_inputs['word_ids'] = []
        # tokenized_inputs['sentence']=[]
        # 如果不是训练集需要构造一个传入global pointer中的mask
        if self.mode != 'test':
            tokenized_inputs['labels'] = []
            if self.gp_mask:
                tokenized_inputs['gp_masks'] = []
            # 提取数据
            # 处理每一个案件对应的内容
            for one_contents, one_events, one_negative_triggers in zip(examples['content'], examples['events'],
                                                                       examples['negative_triggers']):
                # 临时保存每个案件的数据
                temp_example = {}
                temp_example['input_ids'] = []
                temp_example['token_type_ids'] = []
                temp_example['attention_mask'] = []
                temp_example['word_ids'] = []
                temp_example['labels'] = []
                temp_example['gp_masks'] = []
                # 处理的是每一个contents，contents中包含了这个案件的所有句子内容，content是一个句子的内容
                for content in one_contents:
                    tokenize_example = self.tokenizer(content["tokens"], truncation=True, max_length=self.max_len,
                                                      is_split_into_words=True)
                    temp_example['input_ids'].append(tokenize_example.input_ids)
                    temp_example['token_type_ids'].append(tokenize_example.token_type_ids)
                    temp_example['attention_mask'].append(tokenize_example.attention_mask)
                    # 处理word_ids，将None替换成-1
                    word_ids = []
                    for word_id in tokenize_example.word_ids():
                        if word_id != None:
                            word_ids.append(word_id)
                        else:
                            word_ids.append(-1)
                    temp_example['word_ids'].append(word_ids)
                    # 先构造一个全为0的label标签
                    temp_example['labels'].append(torch.zeros((len(word_ids), len(word_ids)), dtype=torch.int8))
                    # 构造一个全为0的gp_mask
                    temp_example['gp_masks'].append(torch.zeros((len(word_ids)), dtype=torch.int8))
                    # 把确定的处理好的值赋给返回的字典
                    # tokenized_inputs['sentence'].append(content['sentence'])
                    tokenized_inputs['input_ids'].append(tokenize_example.input_ids)
                    tokenized_inputs['token_type_ids'].append(tokenize_example.token_type_ids)
                    tokenized_inputs['attention_mask'].append(tokenize_example.attention_mask)
                    tokenized_inputs['word_ids'].append(word_ids)

                # 开始处理每个案件的所有触发词,events包括了这个案件的所有触发词，event是一个触发词事件提及
                labels = temp_example['labels']
                gp_masks = temp_example['gp_masks']
                # 在原来全为0的labels中构造类别id
                for event in one_events:
                    sent_id = event['mention'][0]['sent_id']
                    offset = event['mention'][0]['offset']
                    type_id = event['type_id']
                    word_id = temp_example['word_ids'][sent_id]
                    if offset[0] <= max(word_id):
                        start = word_id.index(offset[0])
                        end = start + word_id.count(offset[0]) - 1
                        labels[sent_id][start][end] = type_id
                        for i in range(start, end + 1):
                            gp_masks[sent_id][i] = 1
                # 同时构造一个传入global pointer的mask
                for negative_trigger in one_negative_triggers:
                    sent_id = negative_trigger['sent_id']
                    offset = negative_trigger['offset']
                    word_id = temp_example['word_ids'][sent_id]
                    if offset[0] <= max(word_id):
                        start = word_id.index(offset[0])
                        end = start + word_id.count(offset[0]) - 1
                        for i in range(start, end + 1):
                            gp_masks[sent_id][i] = 1

                # 把处理好的label添加到返回字典中
                for label, gp_mask in zip(labels, gp_masks):
                    tokenized_inputs["labels"].append(label)
                    if self.gp_mask:
                        tokenized_inputs["gp_masks"].append(gp_mask)
        else:
            tokenized_inputs['triggers'] = []
            if self.gp_mask:
                tokenized_inputs['gp_masks'] = []
            # 提取数据
            # 处理每一个案件对应的内容
            for crime, one_contents, one_candidates in zip(examples['crime'], examples['content'],
                                                           examples['candidates']):
                temp_example = {}
                temp_example['word_ids'] = []
                temp_example['gp_masks'] = []
                temp_example['triggers'] = []
                # 处理的是每一个contents，contents中包含了这个案件的所有句子内容，content是一个句子的内容
                for content in one_contents:
                    tokenize_example = self.tokenizer(content["tokens"], truncation=True, max_length=self.max_len,
                                                      is_split_into_words=True)
                    # 处理word_ids，将None替换成-1
                    word_ids = []
                    for word_id in tokenize_example.word_ids():
                        if word_id != None:
                            word_ids.append(word_id)
                        else:
                            word_ids.append(-1)
                    temp_example['word_ids'].append(word_ids)
                    # 构造一个全为0的gp_mask
                    temp_example['gp_masks'].append(torch.zeros((len(word_ids)), dtype=torch.int8))
                    # 把每句话的触发词存一个列表
                    temp_example['triggers'].append([])
                    # 把确定的处理好的值赋给返回的字典
                    tokenized_inputs['input_ids'].append(tokenize_example.input_ids)
                    tokenized_inputs['token_type_ids'].append(tokenize_example.token_type_ids)
                    tokenized_inputs['attention_mask'].append(tokenize_example.attention_mask)
                    tokenized_inputs['word_ids'].append(word_ids)
                gp_masks = temp_example['gp_masks']
                # 填充全为0的gp_mask
                for candidate in one_candidates:
                    sent_id = candidate['sent_id']
                    offset = candidate['offset']
                    word_id = temp_example['word_ids'][sent_id]
                    if offset[0] <= max(word_id):
                        start = word_id.index(offset[0])
                        end = start + word_id.count(offset[0]) - 1
                        trigger = (candidate['trigger_word'], str(start), str(end), crime)
                        temp_example['triggers'][sent_id].append(trigger)
                        for i in range(start, end + 1):
                            gp_masks[sent_id][i] = 1
                for trigger in temp_example['triggers']:
                    tokenized_inputs['triggers'].append(trigger)
                # 添加gp_mask
                if self.gp_mask:
                    for gp_mask in gp_masks:
                        tokenized_inputs["gp_masks"].append(gp_mask)
        return tokenized_inputs

    # 在使用global pinter的基础上添加了距离矩阵的相关信息，用于w2ner和gccn模型中
    def tokenize_function_gp_gccn(self, examples):
        # 构造一个返回的字典
        tokenized_inputs = {}
        tokenized_inputs['input_ids'] = []
        tokenized_inputs['token_type_ids'] = []
        tokenized_inputs['attention_mask'] = []
        tokenized_inputs['word_ids'] = []
        tokenized_inputs['dist_inputs'] = []
        tokenized_inputs['adjs'] = []
        # 如果不是训练集需要构造一个传入global pointer中的mask
        if self.mode != 'test':
            tokenized_inputs['triggers'] = []
            tokenized_inputs['labels'] = []
            tokenized_inputs['sentence'] = []
            if self.gp_mask:
                tokenized_inputs['gp_masks'] = []
            # 提取数据
            # 处理每一个案件对应的内容
            for crime, one_contents, one_events, one_negative_triggers in zip(examples['crime'],examples['content'], examples['events'],
                                                                       examples['negative_triggers']):
                # 临时保存每个案件的数据
                temp_example = {}
                temp_example['input_ids'] = []
                temp_example['token_type_ids'] = []
                temp_example['attention_mask'] = []
                temp_example['word_ids'] = []
                temp_example['labels'] = []
                temp_example['gp_masks'] = []
                temp_example['triggers'] = []
                temp_example['adjs'] = []
                # 处理的是每一个contents，contents中包含了这个案件的所有句子内容，content是一个句子的内容
                for content in one_contents:
                    tokenize_example = self.tokenizer(content["tokens"], truncation=True, max_length=self.max_len,
                                                      is_split_into_words=True)
                    temp_example['input_ids'].append(tokenize_example.input_ids)
                    temp_example['token_type_ids'].append(tokenize_example.token_type_ids)
                    temp_example['attention_mask'].append(tokenize_example.attention_mask)
                    # 处理word_ids，将None替换成-1
                    word_ids = []
                    for word_id in tokenize_example.word_ids():
                        if word_id != None:
                            word_ids.append(word_id)
                        else:
                            word_ids.append(-1)
                    temp_example['word_ids'].append(word_ids)
                    # 先构造一个全为0的label标签
                    temp_example['labels'].append(torch.zeros((len(word_ids), len(word_ids)), dtype=torch.int8))
                    # 构造一个全为0的gp_mask
                    temp_example['gp_masks'].append(torch.zeros((len(word_ids)), dtype=torch.int8))
                    # 构造一个全为0的adj_mask
                    temp_example['adjs'].append(torch.eye(len(word_ids), dtype=torch.float))
                    # 添加dist_inputs
                    tokenized_inputs['dist_inputs'].append(self.dist_inputs[:len(word_ids), :len(word_ids)])
                    # 把确定的处理好的值赋给返回的字典
                    tokenized_inputs['sentence'].append(content['sentence'])
                    tokenized_inputs['input_ids'].append(tokenize_example.input_ids)
                    tokenized_inputs['token_type_ids'].append(tokenize_example.token_type_ids)
                    tokenized_inputs['attention_mask'].append(tokenize_example.attention_mask)
                    temp_example['triggers'].append([])
                    tokenized_inputs['word_ids'].append(word_ids)

                # 开始处理每个案件的所有触发词,events包括了这个案件的所有触发词，event是一个触发词事件提及
                labels = temp_example['labels']
                gp_masks = temp_example['gp_masks']
                adjs = temp_example['adjs']
                # 在原来全为0的labels中构造类别id
                for event in one_events:
                    sent_id = event['mention'][0]['sent_id']
                    offset = event['mention'][0]['offset']
                    type_id = event['type_id']
                    word_id = temp_example['word_ids'][sent_id]

                    if offset[0] <= max(word_id):
                        start = word_id.index(offset[0])
                        end = start + word_id.count(offset[0]) - 1
                        labels[sent_id][start][end] = type_id
                        adjs[sent_id][start][end] = 1.
                        trigger = (event['mention'][0]['trigger_word'], str(start), str(end), crime)
                        temp_example['triggers'][sent_id].append(trigger)
                        for i in range(start, end + 1):
                            gp_masks[sent_id][i] = 1
                            # adjs[sent_id][start][i] = 1.

                # 同时构造一个传入global pointer的mask
                for negative_trigger in one_negative_triggers:
                    sent_id = negative_trigger['sent_id']
                    offset = negative_trigger['offset']
                    word_id = temp_example['word_ids'][sent_id]
                    if offset[0] <= max(word_id):
                        start = word_id.index(offset[0])
                        end = start + word_id.count(offset[0]) - 1
                        labels[sent_id][start][end] = -1
                        adjs[sent_id][start][end] = 1.
                        trigger = (negative_trigger['trigger_word'], str(start), str(end), crime)
                        temp_example['triggers'][sent_id].append(trigger)
                        for i in range(start, end + 1):
                            gp_masks[sent_id][i] = 1
                            # adjs[sent_id][start][i] = 1.
                for trigger in temp_example['triggers']:
                    tokenized_inputs['triggers'].append(trigger)
                # 把处理好的label添加到返回字典中
                for label, gp_mask, adj in zip(labels, gp_masks, adjs):
                    tokenized_inputs["labels"].append(label)
                    tokenized_inputs["adjs"].append(adj)
                    if self.gp_mask:
                        # 按概率将gp_mask中为0的值变为1
                        # zeros_pos = (gp_mask == 0)
                        # mask = torch.bernoulli(torch.full(gp_mask.shape, 1.0)).bool()
                        # gp_mask[zeros_pos & mask] = 1
                        tokenized_inputs["gp_masks"].append(gp_mask)
        else:
            tokenized_inputs['triggers'] = []
            tokenized_inputs['sentence'] = []
            if self.gp_mask:
                tokenized_inputs['gp_masks'] = []
            # 提取数据
            # 处理每一个案件对应的内容
            for crime, one_contents, one_candidates in zip(examples['crime'], examples['content'],
                                                           examples['candidates']):
                temp_example = {}
                temp_example['word_ids'] = []
                temp_example['gp_masks'] = []
                temp_example['triggers'] = []
                temp_example['adjs'] = []
                # 处理的是每一个contents，contents中包含了这个案件的所有句子内容，content是一个句子的内容
                for content in one_contents:
                    tokenize_example = self.tokenizer(content["tokens"], truncation=True, max_length=self.max_len,
                                                      is_split_into_words=True)
                    # 处理word_ids，将None替换成-1
                    word_ids = []
                    for word_id in tokenize_example.word_ids():
                        if word_id != None:
                            word_ids.append(word_id)
                        else:
                            word_ids.append(-1)
                    temp_example['word_ids'].append(word_ids)
                    # 构造一个全为0的gp_mask
                    temp_example['gp_masks'].append(torch.zeros((len(word_ids)), dtype=torch.int8))
                    # 构造一个全为0的adj_mask
                    temp_example['adjs'].append(torch.eye(len(word_ids), dtype=torch.float))
                    # 把每句话的触发词存一个列表
                    temp_example['triggers'].append([])
                    # 添加dist_inputs
                    tokenized_inputs['dist_inputs'].append(self.dist_inputs[:len(word_ids), :len(word_ids)])
                    # 把确定的处理好的值赋给返回的字典
                    tokenized_inputs['sentence'].append(content['sentence'])
                    tokenized_inputs['input_ids'].append(tokenize_example.input_ids)
                    tokenized_inputs['token_type_ids'].append(tokenize_example.token_type_ids)
                    tokenized_inputs['attention_mask'].append(tokenize_example.attention_mask)
                    tokenized_inputs['word_ids'].append(word_ids)
                gp_masks = temp_example['gp_masks']
                adjs = temp_example['adjs']
                # 填充全为0的gp_mask
                for candidate in one_candidates:
                    sent_id = candidate['sent_id']
                    offset = candidate['offset']
                    word_id = temp_example['word_ids'][sent_id]
                    if offset[0] <= max(word_id):
                        start = word_id.index(offset[0])
                        end = start + word_id.count(offset[0]) - 1
                        # 构造邻接矩阵
                        adjs[sent_id][start][end] = 1.
                        trigger = (candidate['trigger_word'], str(start), str(end), crime)
                        temp_example['triggers'][sent_id].append(trigger)
                        for i in range(start, end + 1):
                            gp_masks[sent_id][i] = 1
                            # adjs[sent_id][start][i] = 1.
                            # if candidate['trigger_word'] in self.all_trig:
                            #     adjs[sent_id][start][i] = 1.
                for trigger in temp_example['triggers']:
                    tokenized_inputs['triggers'].append(trigger)
                for adj in adjs:
                    tokenized_inputs["adjs"].append(adj)
                # 添加gp_mask
                if self.gp_mask:
                    for gp_mask in gp_masks:
                        # 按概率将gp_mask中为0的值变为1
                        # zeros_pos = (gp_mask == 0)
                        # mask = torch.bernoulli(torch.full(gp_mask.shape, 1.0)).bool()
                        # gp_mask[zeros_pos & mask] = 1
                        tokenized_inputs["gp_masks"].append(gp_mask)
        return tokenized_inputs


from transformers.data import DataCollatorForTokenClassification


class MyDataCollatorForTokenClassification(DataCollatorForTokenClassification):
    def torch_call(self, features):
        import torch

        label_name = "label" if "label" in features[0].keys() else "labels"
        mask_name = 'gp_masks'
        dist_input_name = 'dist_inputs'
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None
        adjs = [feature['adjs'] for feature in features] if 'adjs' in features[0].keys() else None
        gp_masks = [feature[mask_name] for feature in features] if mask_name in features[0].keys() else None
        triggers = [feature['triggers'] for feature in features] if 'triggers' in features[0].keys() else None
        dist_inputs = [feature[dist_input_name] for feature in features] if dist_input_name in features[
            0].keys() else None

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors="pt" if labels is None and gp_masks is None and triggers is None else None,
        )
        if labels is None and gp_masks is None and triggers is None:
            return batch

        sequence_length = torch.tensor(batch["input_ids"]).shape[1]
        padding_side = self.tokenizer.padding_side
        pad_labels = None
        # 填充labels
        if labels is not None:
            for label in labels:
                pad_len = sequence_length - len(label)
                label = torch.tensor(label, dtype=torch.int64)
                if padding_side == "right":
                    pad = nn.ZeroPad2d((0, pad_len, 0, pad_len))
                else:
                    pad = nn.ZeroPad2d((0, pad_len, pad_len, 0))
                label = pad(label).unsqueeze(0)
                if pad_labels == None:
                    pad_labels = label
                else:
                    pad_labels = torch.cat((pad_labels, label), dim=0)
            batch[label_name] = pad_labels.numpy()
        # 填充adj
        pad_adjs = None
        if adjs is not None:
            for adj in adjs:
                pad_len = sequence_length - len(adj)
                adj = torch.tensor(adj, dtype=torch.float)
                if padding_side == "right":
                    pad = nn.ZeroPad2d((0, pad_len, 0, pad_len))
                else:
                    pad = nn.ZeroPad2d((0, pad_len, pad_len, 0))
                adj = pad(adj).unsqueeze(0)
                if pad_adjs == None:
                    pad_adjs = adj
                else:
                    pad_adjs = torch.cat((pad_adjs, adj), dim=0)
            batch['adjs'] = pad_adjs.numpy()
        # 填充dis_inputs
        pad_dist_inputs = None
        if dist_inputs is not None:
            for dist_input in dist_inputs:
                pad_len = sequence_length - len(dist_input)
                dist_input = torch.tensor(dist_input, dtype=torch.int64)
                if padding_side == "right":
                    pad = nn.ZeroPad2d((0, pad_len, 0, pad_len))
                else:
                    pad = nn.ZeroPad2d((0, pad_len, pad_len, 0))
                dist_input = pad(dist_input).unsqueeze(0)
                if pad_dist_inputs == None:
                    pad_dist_inputs = dist_input
                else:
                    pad_dist_inputs = torch.cat((pad_dist_inputs, dist_input), dim=0)
            batch[dist_input_name] = pad_dist_inputs.numpy()
        # 填充gb_masks
        if gp_masks is not None:
            pad_masks = []
            for gp_mask in gp_masks:
                pad_len = sequence_length - len(gp_mask)
                gp_mask = gp_mask + pad_len * [0]
                pad_masks.append(gp_mask)
            batch[mask_name] = pad_masks

        for k, v in batch.items():
            if k == 'adjs':
                batch[k] = torch.tensor(v, dtype=torch.float)
            elif k != 'triggers':
                batch[k] = torch.tensor(v, dtype=torch.int64)
        return batch


if __name__ == '__main__':
    import numpy as np

    config = AutoConfig.from_pretrained('../../Models/bert-base-chinese')
    config.max_len = 250
    dataset = CAILDataset(max_len=250, tokenizer_path='../../Models/bert-base-chinese/',
                          train_data_path='../corpus/train.jsonl',
                          dev_data_path='../corpus/valid.jsonl', test_data_path='../corpus/test.jsonl')
    dataset.LoadTrainDataset()
    # dataset.LoadDevDataset()
    dataset.LoadTestDataset()
