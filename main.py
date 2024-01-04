import copy
import json
from enum import Enum
from typing import Union, Optional, Dict
import numpy as np
import pandas as pd
import torch
from datasets import load_metric
from tqdm import tqdm
from transformers import TrainingArguments, DataCollatorWithPadding, AutoConfig, IntervalStrategy, \
    EarlyStoppingCallback, EvalPrediction, DataCollatorForTokenClassification

from dataset.cail_dataset import CAILDataset, bio_labels, pure_event2id, MyDataCollatorForTokenClassification
from models.net import CAILNet as Net
from models.net_crf import CAILNet as NetCRF
from models.net_bilstm_crf import CAILNet as NetBilstmCRF
from models.net_gp import CAILNet as NetGP
from models.net_gp_dot import CAILNet as NetGPDot
from models.net_gp_attn import CAILNet as NetGPAttn
from models.net_rnn_gp import CAILNet as NetRNNGP
from models.net_bilstm_gp import CAILNet as NetBilstmGP
from models.net_gp_cnn import CAILNet as NetGPCNN
from models.net_w2gp import CAILNet as NetW2GP
from models.net_gp_gccn import CAILNet as NetGPGCCN
from models.model_fusion import CAILNet_Fusion as NetFusion
from utils.my_trainer import MyTrainer
import torch.nn.functional as F
import os

from utils.pub_utils import dp_nested_concat, save_training_args, regularization_match


class Net_Enhanced(Enum):
    USE_CRF = 'use_crf'
    USE_LSTM_CRF = 'use_lstm_crf'
    USE_GP = 'use_global_pointer'
    USE_LSTM_GP = 'use_bilstm_global_pointer'
    USE_GP_CNN = 'use_global_pointer_cnn'
    USE_Net_Fusion = 'use_net_fusion'
    USE_RNN_GP = 'use_rnn_global_pointer'
    USE_GP_Dot = 'use_global_pointer_dot'
    USE_GP_Attn = 'use_global_pointer_attn'
    USE_W2GP = 'use_w2_global_pointer'
    USE_GP_GCCN = 'use_global_pointer_gccn'


# 任务器类:封装方法，可训练模型，测试模型
class Tasker():
    # 是否使用动态填充技术

    def __init__(self, model='bert-base-chinese', net_enhanced=None, best_model=False, use_dynamic_padding=False):
        """
        初始化任务器
        use_cnn和use_plus只能使用一个，他们是不同的网络结构，如果都是False，那么只在基模型后面加分类层
        :param model: 基模型名字
        :param use_cnn: 为True则基模型后面加了cnn网络
        :param use_plus: 为True则基模型后面其他的复杂网络
        :param best_model: 是否加载的是保存下来的最好的模型
        """
        self.use_dynamic_padding = use_dynamic_padding
        self.use_crf = False
        self.use_lstm_crf = False
        self.use_global_pointer = False
        self.net_enhanced = net_enhanced
        # 拼接路径所用的模型名称
        self.model_name = model
        self.base_model = model
        self.best_model = best_model
        # 选择自己搭建的网络结构
        self.net = Net
        if net_enhanced == Net_Enhanced.USE_CRF:
            self.use_crf = True
            self.net = NetCRF
            self.model_name = model + '-crf'
        if net_enhanced == Net_Enhanced.USE_LSTM_CRF:
            self.use_lstm_crf = True
            self.net = NetBilstmCRF
            self.model_name = model + '-bilstm-crf'
        if net_enhanced == Net_Enhanced.USE_GP:
            self.use_global_pointer = True
            self.net = NetGP
            self.model_name = model + '-gp'
        if net_enhanced == Net_Enhanced.USE_LSTM_GP:
            self.use_global_pointer = True
            self.net = NetBilstmGP
            self.model_name = model + '-bilstm-gp'
        if net_enhanced == Net_Enhanced.USE_RNN_GP:
            self.use_global_pointer = True
            self.net = NetRNNGP
            self.model_name = model + '-rnn-gp'
        if net_enhanced == Net_Enhanced.USE_GP_CNN:
            self.use_global_pointer = True
            self.net = NetGPCNN
            self.model_name = model + '-gp-cnn'
        if net_enhanced == Net_Enhanced.USE_Net_Fusion:
            self.use_global_pointer = True
            self.net = NetFusion
            self.model_name = model + '-net-fusion'
            self.best_model = True
        if net_enhanced == Net_Enhanced.USE_GP_Dot:
            self.use_global_pointer = True
            self.net = NetGPDot
            self.model_name = model + '-gp-dot'
        if net_enhanced == Net_Enhanced.USE_GP_Attn:
            self.use_global_pointer = True
            self.net = NetGPAttn
            self.model_name = model + '-gp-attn'
        if net_enhanced == Net_Enhanced.USE_W2GP:
            self.use_global_pointer = True
            self.net = NetW2GP
            self.model_name = model + '-w2gp'
        if net_enhanced == Net_Enhanced.USE_GP_GCCN:
            self.use_global_pointer = True
            self.net = NetGPGCCN
            # self.model_name = model + '-gp-gccn'
            self.model_name = model + '-gp-gccn (lr 5e-3)'
            # self.model_name = model + '-gp-gccn (gcn 3)'
        self.trainer = None
        self.max_len = 350
        # 初始化所有路径
        self.init_path()
        # 初始化数据和模型
        self.init_data_and_model()

    def init_path(self):
        # 训练过程输出文件存放路径
        # self.output_dir_path = "/root/autodl-tmp/output/" + self.model_name
        self.output_dir_path = "output/" + self.model_name
        # 日志存放路径
        self.logging_dir_path = "log/" + self.model_name
        # 读取模型路径
        # self.model_path = "models/" + self.base_model
        self.model_path = "../Models/" + self.base_model
        self.best_model_path = "best_models/" + self.model_name
        # 配置文件读取路径
        # self.config_path = "models/" + self.base_model + "/config.json"
        self.config_path = "../Models/" + self.base_model + "/config.json"
        self.best_config_path = "best_models/" + self.model_name + "/config.json"
        # 分词器读取路径--注意路径最后有反斜杆
        # self.tokenizer_path = "models/" + self.base_model + "/"
        self.tokenizer_path = "../Models/" + self.base_model + "/"
        self.best_tokenizer_path = "best_models/" + self.model_name + "/"
        # 保存最优模型位置
        self.save_model_path = "best_models/" + self.model_name
        # 训练数据集的路径
        self.train_data_path = "corpus/train.jsonl"
        # 验证数据集的路径
        self.dev_data_path = "corpus/valid.jsonl"
        # 测试数据集的路径
        self.test_data_path = "corpus/test_stage2.jsonl"

    # 初始化数据和模型
    def init_data_and_model(self):
        # 加载模型
        if self.net_enhanced == Net_Enhanced.USE_Net_Fusion:
            self.model = NetFusion(models_name=self.base_model)
        elif self.best_model:
            # 加载模型配置 output_hidden_states是否获取所有隐藏层的输出
            self.config = AutoConfig.from_pretrained(self.best_config_path, output_hidden_states=True)
            self.config.max_len = self.max_len
            # 从保存的模型里面加载，包括了分类层的权重
            self.model = self.net.from_pretrained(pretrained_model_name_or_path=self.best_model_path,
                                                  config=self.config)
        else:
            # 加载模型配置 output_hidden_states是否获取所有隐藏层的输出
            self.config = AutoConfig.from_pretrained(self.config_path, output_hidden_states=True)
            self.config.max_len = self.max_len
            # 从基模型里面加载，不包括分类层的权重
            self.model = self.net(config=self.config)
            # 初始化基模型的权重
            self.model.init_base_model(model_path=self.model_path)
        if self.net_enhanced == Net_Enhanced.USE_Net_Fusion:
            model1 = self.base_model.split('+')[0]
            path = os.path.join('best_models', model1 + '/')
            self.Datasets = CAILDataset(max_len=self.max_len, tokenizer_path=path,
                                        train_data_path=self.train_data_path,
                                        dev_data_path=self.dev_data_path, test_data_path=self.test_data_path)
        elif self.best_model:
            self.Datasets = CAILDataset(max_len=self.max_len, tokenizer_path=self.best_tokenizer_path,
                                        train_data_path=self.train_data_path,
                                        dev_data_path=self.dev_data_path, test_data_path=self.test_data_path)
        else:
            self.Datasets = CAILDataset(max_len=self.max_len, tokenizer_path=self.tokenizer_path,
                                        train_data_path=self.train_data_path,
                                        dev_data_path=self.dev_data_path, test_data_path=self.test_data_path)
        self.tokenizer = self.Datasets.get_tokenizer()
        # 动态填充，即将每个批次的输入序列填充到一样的长度
        self.data_collator = MyDataCollatorForTokenClassification(
            tokenizer=self.tokenizer) if self.use_global_pointer else DataCollatorForTokenClassification(
            tokenizer=self.tokenizer)
        # 初始化优化器，设置不同层的学习率
        param_group = self.get_group_parameters()
        self.optimizer = torch.optim.AdamW(param_group, lr=5e-5, eps=1e-7)
        lr_lambda = lambda epoch: 1 / (epoch + 1)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda, last_epoch=-1)

    # 设置不同层的学习率
    def get_group_parameters(self):
        params = list(self.model.named_parameters())
        no_decay = ['bias,', 'LayerNorm']
        # other = ['lstm', 'classifier', 'crf', 'gcn', 'convLayer','cln','predictor','global_pointer']
        other = ['gcn', 'convLayer', 'cln', 'predictor']
        no_main = no_decay + other
        param_group = [
            {'params': [p for n, p in params if not any(nd in n for nd in no_main)], 'weight_decay': 1e-5, 'lr': 1e-5},
            {'params': [p for n, p in params if not any(nd in n for nd in other) and any(nd in n for nd in no_decay)],
             'weight_decay': 0, 'lr': 1e-5},
            {'params': [p for n, p in params if any(nd in n for nd in other) and any(nd in n for nd in no_decay)],
             'weight_decay': 0, 'lr': 1e-4},
            {'params': [p for n, p in params if any(nd in n for nd in other) and not any(nd in n for nd in no_decay)],
             'weight_decay': 1e-5, 'lr': 1e-4},
        ]
        return param_group

    # 设置训练器的超参数
    # 详细参数说明见https://zhuanlan.zhihu.com/p/363670628
    def get_training_args(self):
        training_args = TrainingArguments(
            output_dir=self.output_dir_path,
            evaluation_strategy=IntervalStrategy.STEPS,
            eval_steps=500,
            save_strategy=IntervalStrategy.STEPS,
            save_steps=500,
            save_total_limit=10,
            gradient_accumulation_steps=8,
            eval_accumulation_steps=500,
            learning_rate=5e-3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=20,
            weight_decay=1e-5,
            # warmup_ratio=0.05,
            logging_dir=self.logging_dir_path,
            logging_strategy=IntervalStrategy.EPOCH,
            load_best_model_at_end=True,
            metric_for_best_model='f1',
            # 如果网络前向计算的时候没用到labels则需要指定标签，否则trainer会在网络需要的参数中找不到labels，从而排除labels
            label_names=['labels'],
            # fp16=True,
            # fp16_full_eval=True,
            group_by_length=True if self.use_dynamic_padding else False,
            dataloader_pin_memory=True,
            warmup_steps=1000,
            # dataloader_num_workers=4,
            # no_cuda=True,
            seed=35,
            optim='adamw_torch'
            # adafactor=True
        )
        return training_args

    def train(self, use_fgm=False, resume_from_checkpoint: Optional[Union[str, bool]] = None):
        """
        训练模型
        :param use_fgm: 是否使用对抗网络FGM
        :param resume_from_checkpoint:如果是 str，则为由上一个 Trainer 实例保存的已保存检查点的本地路径。
        如果一个 bool 并且等于 True，则在args.output_dir加载最后一个检查点，
        如上一个 Trainer 实例所保存的那样。
        如果存在，训练将从此处加载的模型/优化器/调度程序状态恢复。
        :return:
        """
        # 加载训练数据
        train_datasets = self.Datasets.LoadTrainDataset()
        # 加载验证数据
        dev_datasets = self.Datasets.LoadDevDataset()
        # 得到训练器的超参数
        training_args = self.get_training_args()
        # 构造训练器
        trainer = MyTrainer(
            self.model,
            training_args,
            train_dataset=train_datasets,
            eval_dataset=dev_datasets,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics_gp if self.use_global_pointer else self.compute_metrics,
            optimizers=(self.optimizer, None),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=4)],
            use_fgm=use_fgm,
            use_dynamic_padding=self.use_dynamic_padding
        )

        # 训练模型
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        # 保存模型
        trainer.save_model(self.save_model_path)
        # trainer.model.save_pretrained(self.save_model_path)
        # trainer.tokenizer.save_pretrained(self.save_model_path)
        # 保存一下trainer,方便训练完之后紧接着测试
        trainer._load_best_model()
        # trainer.evaluate()
        self.trainer = trainer

    def evaluate(self):
        # 加载训练数据
        train_datasets = self.Datasets.LoadTrainDataset()
        # 加载验证数据
        dev_datasets = self.Datasets.LoadDevDataset()
        # 得到训练器的超参数
        training_args = self.get_training_args()
        # 构造训练器
        trainer = MyTrainer(
            self.model,
            training_args,
            train_dataset=train_datasets,
            eval_dataset=dev_datasets,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics_gp if self.use_global_pointer else self.compute_metrics,
            optimizers=(self.optimizer, None),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=4)],
            use_dynamic_padding=self.use_dynamic_padding
        )
        trainer.evaluate()

    def test(self):
        """
        使用保存下来的最好的模型进行测试
        :return:
        """
        if self.trainer == None:
            # 加载测试数据
            test_datasets = self.Datasets.LoadTestDataset()
            # 得到训练器的超参数
            training_args = self.get_training_args()
            # 构造训练器
            trainer = MyTrainer(
                self.model,
                args=training_args,
                data_collator=self.data_collator,
                tokenizer=self.tokenizer,
                compute_metrics=self.compute_metrics_gp if self.use_global_pointer else self.compute_metrics,
                use_dynamic_padding=self.use_dynamic_padding
            )
            self.trainer = trainer
            # 使用训练器预测模型
            trainer.args.per_device_eval_batch_size = 2

            # test_dataloader = trainer.get_test_dataloader(test_datasets)
            # predictions_logits = self.predict(test_dataloader)
            self.trainer.args.eval_accumulation_steps = 100
            # self.trainer.args.use_legacy_prediction_loop=True
            predictions_logits, _, _ = self.trainer.predict(test_datasets, ignore_keys=['labels'])
            # 生成提交结果的文件
            if self.use_global_pointer:
                self.save_submition_gp(predictions_logits, test_datasets)
            else:
                self.save_submition(predictions_logits, test_datasets)
        else:
            # 加载测试数据
            test_datasets = self.Datasets.LoadTestDataset()
            # 使用训练器预测模型
            self.trainer.args.per_device_eval_batch_size = 2

            # test_dataloader = self.trainer.get_test_dataloader(test_datasets)
            # predictions_logits = self.predict(test_dataloader)
            self.trainer.args.eval_accumulation_steps = 100
            # self.trainer.args.eval_accumulation_steps = None
            predictions_logits, _, _ = self.trainer.predict(test_datasets, ignore_keys=['labels'])
            # 生成提交结果的文件
            if self.use_global_pointer:
                self.save_submition_gp(predictions_logits, test_datasets)
            else:
                self.save_submition(predictions_logits, test_datasets)

    def predict(self, test_dataloader):
        """
        预测模型的输出结果
        """
        self.model.cuda()
        self.model.eval()
        predictions_labels = None
        predictions_logits = None
        labels = None
        for step, batch in enumerate(tqdm(test_dataloader, desc="Predict")):
            # if hasattr(torch.cuda, 'empty_cache'):
            #     torch.cuda.empty_cache()
            with torch.no_grad():
                inputs = {"input_ids": batch['input_ids'].cuda(),
                          "token_type_ids": batch['token_type_ids'].cuda(),
                          "attention_mask": batch['attention_mask'].cuda(),
                          }
                output = self.model(**inputs)
                if self.use_global_pointer:
                    predictions_logit = output['logits'].cpu().numpy()
                    label = batch['labels'].cpu().numpy()
                    if labels is None:
                        labels = label
                    else:
                        labels = dp_nested_concat(labels, label, 0)
                    if predictions_logits is None:
                        predictions_logits = predictions_logit
                    else:
                        predictions_logits = dp_nested_concat(predictions_logits, predictions_logit, 0)
                    continue
                if self.use_crf or self.use_lstm_crf:
                    predictions_label = output['tag'].detach()
                else:
                    logits = output['logits'].detach()
                    predictions_label = np.argmax(logits, axis=2)
                if predictions_labels is None:
                    predictions_labels = predictions_label
                else:
                    predictions_labels = torch.cat([predictions_labels, predictions_label], dim=0)
        if self.use_global_pointer:
            # return predictions_logits
            return predictions_logits, labels
        predictions_labels = predictions_labels.cpu().numpy()
        return predictions_labels

    # 搜索参数需要每次模型初始化一个新实例
    def model_init(self):
        # 加载模型配置 output_hidden_states是否获取所有隐藏层的输出
        config = AutoConfig.from_pretrained(self.config_path, output_hidden_states=True)
        config.max_len = self.max_len
        # 加载模型
        model = self.net(config=config)
        model.init_base_model(model_path=self.model_path)
        return model

    # 搜索超参数
    def search_hyperparameter_train(self):
        # 得到训练器的超参数
        training_args = self.get_training_args()
        training_args.fp16_full_eval = False
        # 加载训练数据
        train_datasets = self.Datasets.LoadTrainDataset()
        # 加载验证数据
        dev_datasets = self.Datasets.LoadDevDataset()
        # 构造训练器
        trainer = MyTrainer(
            model_init=self.model_init,
            args=training_args,
            train_dataset=train_datasets,
            eval_dataset=dev_datasets,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics_gp if self.use_global_pointer else self.compute_metrics,
            use_dynamic_padding=self.use_dynamic_padding
        )
        best_run = trainer.hyperparameter_search(n_trials=15, direction="maximize")
        print(best_run)
        print("*" * 200)
        # 设置超参数并训练
        for n, v in best_run.hyperparameters.items():
            setattr(trainer.args, n, v)
        trainer.train()
        # 保存模型
        trainer.save_model(self.save_model_path)
        self.trainer = trainer
        return best_run

    # 打印模型参数
    def print_model(self):
        f = open('log1.txt', 'w')
        # print(self.optimizer.state_dict()['param_groups'][2])
        # print(self.model, file=f)
        for k,v in self.model.named_parameters():  # 打印出参数矩阵及值
            print(k,v, file=f)

        # 生成提交数据

    # 生成提交文件
    def save_submition(self, predictions_labels, test_datasets):
        label_map = {i: label for i, label in enumerate(bio_labels)}
        label_map[-100] = 'PAD'
        predictions = [[label_map[label] for label in labels]
                       for labels in predictions_labels]
        word_ids = test_datasets['word_ids']  # 分词后的每个token对应最原始数据中的token位置
        # Save predictions
        output_test_predictions_file = os.path.join(self.best_model_path, "results2.jsonl")
        with open(output_test_predictions_file, "w") as writer:
            Cnt = 0
            levenTypes = list(pure_event2id.keys())
            # 记录数量
            write_count = 0
            read_count = 0
            # line_count = 10
            with open(self.test_data_path, "r", encoding='UTF-8') as fin:
                lines = fin.readlines()
                for line in lines:
                    doc = json.loads(line)
                    # line_count -= 1
                    # if line_count == -1:
                    #     break
                    res = {}
                    res['id'] = doc['id']
                    res['predictions'] = []
                    for mention in doc['candidates']:
                        read_count += 1
                        # 需要预测触发词的位置超出了句子的长度，打印长度并结束本次循环
                        if mention['offset'][0] > max(word_ids[Cnt + mention['sent_id']]):
                            print(len(doc['content'][mention['sent_id']]['tokens']),
                                  max(word_ids[Cnt + mention['sent_id']]), "id：", mention['id'])
                            res['predictions'].append({"id": mention['id'], "type_id": 0})
                            write_count += 1
                            continue
                        # 获取触发词在predictions中的开始位置
                        start_index = word_ids[Cnt + mention['sent_id']].index(mention['offset'][0])
                        is_NA = False if predictions[Cnt + mention['sent_id']][start_index].startswith(
                            "B") else True
                        if not is_NA:
                            Type = predictions[Cnt + mention['sent_id']][start_index][2:]
                            # if mention['offset'][1] in word_ids[Cnt + mention['sent_id']]:
                            #     end_index = word_ids[Cnt + mention['sent_id']].index(mention['offset'][1]) - 1
                            # else:
                            #     count = word_ids[Cnt + mention['sent_id']].count(mention['offset'][0])
                            #     end_index = word_ids[Cnt + mention['sent_id']].index(mention['offset'][0]) + count
                            # for i in range(start_index + 1, end_index):
                            #     if predictions[Cnt + mention['sent_id']][i][2:] != Type:
                            #         is_NA = True
                            #         break
                            if not is_NA:
                                res['predictions'].append({"id": mention['id'], "type_id": levenTypes.index(Type)})
                                write_count += 1
                        if is_NA:
                            res['predictions'].append({"id": mention['id'], "type_id": 0})
                            write_count += 1
                    writer.write(json.dumps(res) + "\n")
                    Cnt += len(doc['content'])
            print('读入触发词总数：', read_count, '\t写入触发词总数：', write_count)

    def save_submition_gp(self, predictions_labels, test_datasets):
        word_ids = test_datasets['word_ids']  # 分词后的每个token对应最原始数据中的token位置
        # Save predictions
        if not os.path.exists(self.best_model_path):
            os.makedirs(self.best_model_path)
        output_test_predictions_file = os.path.join(self.best_model_path, "results2.jsonl")
        with open(output_test_predictions_file, "w") as writer:
            Cnt = 0
            # 记录数量
            write_count = 0
            read_count = 0
            # line_count = 10
            with open(self.test_data_path, "r", encoding='UTF-8') as fin:
                lines = fin.readlines()
                for line in lines:
                    doc = json.loads(line)
                    # line_count -= 1
                    # if line_count == -1:
                    #     break
                    res = {}
                    res['id'] = doc['id']
                    res['predictions'] = []
                    for mention in doc['candidates']:
                        read_count += 1
                        # 需要预测触发词的位置超出了句子的长度，打印长度并结束本次循环
                        if mention['offset'][0] > max(word_ids[Cnt + mention['sent_id']]):
                            print(len(doc['content'][mention['sent_id']]['tokens']),
                                  max(word_ids[Cnt + mention['sent_id']]), "id：", mention['id'])
                            res['predictions'].append({"id": mention['id'], "type_id": type_id})
                            write_count += 1
                            continue
                        # 获取触发词在predictions中的开始位置和结束位置
                        start_index = word_ids[Cnt + mention['sent_id']].index(mention['offset'][0])
                        count = word_ids[Cnt + mention['sent_id']].count(mention['offset'][0])
                        end_index = word_ids[Cnt + mention['sent_id']].index(mention['offset'][0]) + count - 1
                        type_id = predictions_labels[Cnt + mention['sent_id']][start_index][end_index].item()
                        res['predictions'].append({"id": mention['id'], "type_id": type_id})
                        # res['predictions'].append({"id": mention['id'], "type_id": predictions_labels})
                        write_count += 1
                    writer.write(json.dumps(res) + "\n")
                    Cnt += len(doc['content'])
            print('读入触发词总数：', read_count, '\t写入触发词总数：', write_count)

    # 筛选出验证集判断错误的数据
    def check_dev(self):
        # 加载验证数据
        dev_datasets = self.Datasets.LoadDevDataset()
        # 得到训练器的超参数
        training_args = self.get_training_args()
        # 构造训练器
        trainer = MyTrainer(
            self.model,
            args=training_args,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics_gp if self.use_global_pointer else self.compute_metrics,
            use_dynamic_padding=self.use_dynamic_padding
        )
        # metrics=trainer.evaluate(dev_datasets)
        # print(metrics)
        self.trainer = trainer
        # 使用训练器预测模型
        trainer.args.per_device_eval_batch_size = 8
        # dev_dataloader = trainer.get_eval_dataloader(dev_datasets)
        # predictions_labels = self.predict(dev_dataloader)
        self.trainer.args.eval_accumulation_steps = 100
        predictions_labels, labels, metrics = self.trainer.predict(dev_datasets)
        # metrics = self.compute_metrics_gp(EvalPrediction(predictions=predictions_labels, label_ids=labels))
        print(metrics)
        word_ids = dev_datasets['word_ids']
        sentences = dev_datasets['sentence']  # 分词后的每个token对应最原始数据中的token位置
        labels_map = {key: value for value, key in pure_event2id.items()}
        # Save predictions
        output_dev_predictions_file = os.path.join(self.best_model_path, "check_dev.jsonl")
        with open(output_dev_predictions_file, "w", encoding='UTF-8') as writer:
            Cnt = 0
            # 记录数量
            write_count = 0
            read_count = 0
            nan_count = 0
            # line_count = 10
            with open(self.dev_data_path, "r", encoding='UTF-8') as fin:
                lines = fin.readlines()
                for line in lines:
                    doc = json.loads(line)
                    # line_count -= 1
                    # if line_count == -1:
                    #     break
                    res = {}
                    res['id'] = doc['id']
                    for event, negative_trigger in zip(doc['events'], doc['negative_triggers']):
                        mention = event['mention'][0]
                        read_count += 1
                        # 需要预测触发词的位置超出了句子的长度，打印长度并结束本次循环
                        if mention['offset'][0] > max(word_ids[Cnt + mention['sent_id']]):
                            print(len(doc['content'][mention['sent_id']]['tokens']),
                                  max(word_ids[Cnt + mention['sent_id']]), "id：", mention['id'])
                            res['predictions'] = {"id": mention['id'], "pre_type_id": type_id,
                                                  'pre_type_name': labels_map[0],
                                                  "type_id": event['type_id'], 'type_name': event['type'],
                                                  "trigger_word": mention['trigger_word'],
                                                  "sentence": sentences[Cnt + mention['sent_id']]}
                            write_count += 1
                            continue
                        # 获取触发词在predictions中的开始位置和结束位置
                        start_index = word_ids[Cnt + mention['sent_id']].index(mention['offset'][0])
                        count = word_ids[Cnt + mention['sent_id']].count(mention['offset'][0])
                        end_index = word_ids[Cnt + mention['sent_id']].index(mention['offset'][0]) + count - 1
                        type_id = predictions_labels[Cnt + mention['sent_id']][start_index][end_index].item()
                        res['predictions'] = {"id": mention['id'],
                                              "pre_type_id": type_id, 'pre_type_name': labels_map[type_id],
                                              "type_id": event['type_id'], 'type_name': event['type'],
                                              "trigger_word": mention['trigger_word'],
                                              "sentence": sentences[Cnt + mention['sent_id']]}
                        if type_id != event['type_id']:
                            write_count += 1
                            writer.write(json.dumps(res, ensure_ascii=False) + "\n")
                    Cnt += len(doc['content'])
        print('读入触发词总数：', read_count, '\t写入触发词总数：', write_count, '\t预测为空并且预测正确了的数量：', nan_count)

    # 指标评估
    def compute_metrics(self, eval_preds):
        # 没网络时从本地加载
        metric = load_metric("utils/f1.py")
        logits, labels = eval_preds
        if self.use_crf or self.use_lstm_crf:
            predictions = logits
        else:
            predictions = np.argmax(logits, axis=2)
        # 移除忽略的标签
        true_predictions = []
        true_labels = []
        for prediction, label in zip(predictions, labels):
            for (p, l) in zip(prediction, label):
                if l != -100:
                    true_predictions.append(p)
                    true_labels.append(l)
        macro_f1 = metric.compute(predictions=true_predictions, references=true_labels, average="macro")
        micro_f1 = metric.compute(predictions=true_predictions, references=true_labels, average="micro")
        f1 = (macro_f1['f1'] + micro_f1['f1']) / 2
        return {'f1': f1, 'macro_f1': macro_f1['f1'], 'micro_f1': micro_f1['f1']}

    # global pointer指标评估
    def compute_metrics_gp(self, eval_preds):
        # 没网络时从本地加载
        metric = load_metric("utils/f1.py")
        logits, labels = eval_preds
        predictions = logits
        # np.seterr(invalid='ignore')
        # # 为每个类别统计 TP,FN,FP
        # all_F1 = 0.
        # all_TP = 0.
        # all_FP = 0.
        # all_FN = 0.
        # for label in tqdm(range(1, 109), desc='Computing F1...'):
        #     TP = np.sum(np.where(predictions == label, 1., 0.) * np.where(labels == label, 1., 0.))
        #     FP = np.sum(np.where(predictions == label, 1., 0.) * np.where(labels != label, 1., 0.))
        #     FN = np.sum(np.where(predictions != label, 1., 0.) * np.where(labels == label, 1., 0.))
        #     if (TP + FP) == 0:
        #         P = 0.
        #     else:
        #         P = TP / (TP + FP)
        #     if (TP + FN) == 0:
        #         R = 0.
        #     else:
        #         R = TP / (TP + FN)
        #     if (P + R) == 0:
        #         F1 = 0.
        #     else:
        #         F1 = (2 * P * R) / (P + R)
        #     all_F1 += F1
        #     all_TP += TP
        #     all_FP += FP
        #     all_FN += FN
        # macro_f1 = all_F1 / 108
        # if (all_TP + all_FP) == 0:
        #     tem_P = 0.
        # else:
        #     tem_P = all_TP / (all_TP + all_FP)
        # if (all_TP + all_FN) == 0:
        #     tem_R = 0.
        # else:
        #     tem_R = all_TP / (all_TP + all_FN)
        # if (tem_P + tem_R) == 0:
        #     micro_f1 = 0.
        # else:
        #     micro_f1 = (2 * tem_P * tem_R) / (tem_P + tem_R)
        # f1 = (macro_f1 + micro_f1) / 2
        # 构造预测标签和真实标签
        true_predictions = []
        true_labels = []
        for i, (prediction, label) in tqdm(enumerate(zip(predictions, labels)), desc="compute metrics "):
            if isinstance(label, torch.Tensor):
                label = label.numpy()
            # 得到所有预测错误标签的位置
            label_index = np.argwhere(label != 0).tolist()
            # pre_index=np.argwhere(prediction!=0).tolist()
            # for index in pre_index:
            #     if index not in label_index:
            #         label_index.append(index)
            for index in label_index:
                label_id = label[index[0], index[1]]
                true_labels.append(0 if label_id == -1 else label_id)  # 将负触发词也替换成0标签
                pre_label = prediction[index[0], index[1]]
                true_predictions.append(pre_label)
        #将结果写入文件
        # self.write_result(true_predictions,true_labels)
        macro_f1 = metric.compute(predictions=true_predictions, references=true_labels, average="macro")
        micro_f1 = metric.compute(predictions=true_predictions, references=true_labels, average="micro")
        f1 = (macro_f1['f1'] + micro_f1['f1']) / 2
        # print({'f1': f1, 'macro_f1': macro_f1['f1'], 'micro_f1': micro_f1['f1']})
        return {'f1': f1, 'macro_f1': macro_f1['f1'], 'micro_f1': micro_f1['f1']}
    #将预测结果写入文件
    def write_result(self,predictions,labels):
        predictions = [int(prediction) for prediction in predictions]
        labels = [int(label) for label in labels]
        result = {'predictions': predictions, 'labels': labels}
        with open('model_results.json', 'w') as f:
            json.dump(result, f)

# 超参数搜索目标值
def f1_compute_objective(metrics: Dict[str, float]) -> float:
    """
    The default objective to maximize/minimize when doing an hyperparameter search. It is the evaluation loss if no
    metrics are provided to the [`Trainer`], the sum of all metrics otherwise.
    Args:
        metrics (`Dict[str, float]`): The metrics returned by the evaluate method.
    Return:
        `float`: The objective to minimize or maximize
    """
    metrics = copy.deepcopy(metrics)
    f1 = metrics.pop("eval_f1", None)
    _ = metrics.pop("epoch", None)
    # Remove speed metrics
    speed_metrics = [m for m in metrics.keys() if m.endswith("_runtime") or m.endswith("_per_second")]
    for sm in speed_metrics:
        _ = metrics.pop(sm, None)
    return f1


if __name__ == '__main__':
    import os

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
    # os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    # 清空gpu空闲内存
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    # 不使用wandb记录变化
    os.environ["WANDB_DISABLED"] = "true"
    # wandb记录在本地
    # os.environ['WANDB_MODE'] = 'dryrun'
    tasker = Tasker(model='deberta-v2-chinese',
                    net_enhanced=Net_Enhanced.USE_GP_GCCN, best_model=True,
                    use_dynamic_padding=True)
    # state_dict = tasker.model.state_dict()
    # 搜索超参数
    # tasker.search_hyperparameter_train()
    # 训练
    # tasker.train(use_fgm=True, resume_from_checkpoint=None)
    # 检查模型的判断
    # tasker.check_dev()
    # 打印最好模型的参数
    # arg_path = tasker.best_model_path
    # save_training_args(trainer=tasker.trainer)
    # 测试
    tasker.test()
    #评估
    # tasker.evaluate()
    # tasker.print_model()
    # test_dataset = tasker.Datasets.LoadTestDataset()
    # tasker.save_submition_gp(19, test_dataset)
    #pipreqs . --encoding=utf8 --force 导出依赖包