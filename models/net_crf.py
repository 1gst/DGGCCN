import numpy as np
import pandas as pd
import torch
from torch.nn import CrossEntropyLoss
from torchcrf import CRF
from transformers.modeling_utils import PreTrainedModel
from torch import nn
from transformers import AutoModel, AutoModelForMaskedLM, BertModel


# from utils.crf import CRF, to_crf_pad, unpad_crf


class CAILNet(PreTrainedModel):
    def __init__(self, config, dropout_rate=0.2, class_num=217):
        super(CAILNet, self).__init__(config)
        self.max_len = config.max_len
        self.config = config

        # 加载预训练模型
        # self.bert = AutoModelForMaskedLM.from_pretrained(model_path, config=self.config)
        self.bert = AutoModel.from_config(config=self.config)
        # dropout与分类网络
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.config.hidden_size, class_num)
        self._init_weights(self.classifier)
        # crf层
        # self.crf = CRF(class_num)
        self.crf = CRF(class_num, batch_first=True)

    def init_base_model(self, model_path):
        self.bert = AutoModel.from_pretrained(pretrained_model_name_or_path=model_path, config=self.config)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            module.weight.data = torch.nn.init.xavier_uniform_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        """
        前向计算
        :param input_ids:[batch_size,seq_len]
        :param token_type_ids:[batch_size,seq_len]
        :param attention_mask:[batch_size,seq_len]
        :return:
        """
        # np.set_printoptions(threshold=np.inf)
        #
        # print(input_ids.cpu().numpy())
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                            return_dict=True)
        # 得到模型的最后一层没有softmax的隐藏层输出
        hidden_state = outputs.get('last_hidden_state')
        # 得到模型的最后一层没有softmax的cls输出
        # cls_output = outputs.get('pooler_output').unsqueeze(1).repeat(1, hidden_state.size()[1], 1)
        # hidden_state = torch.cat([hidden_state, cls_output], dim=-1)
        # hidden_state = torch.add(hidden_state, cls_output.unsqueeze(1))
        emissions = self.classifier(self.dropout(hidden_state))
        if labels is not None:
            # 使用pytorch-crf库
            attention_mask = attention_mask.clone().bool().detach()
            pad_mask = (labels != -100)
            pad_mask[:, 0] = True
            if attention_mask is not None:
                loss_mask = ((attention_mask == 1) & pad_mask)
            else:
                loss_mask = ((torch.ones(emissions.shape) == 1) & pad_mask)
            loss = -self.crf.forward(emissions, labels, loss_mask, reduction='sum')
            tag = self.crf.decode(emissions)
            best_path = torch.tensor(tag)
            # pad_mask = (labels != -100)
            # if attention_mask is not None:
            #     loss_mask = ((attention_mask == 1) & pad_mask)
            # else:
            #     loss_mask = ((torch.ones(emissions.shape) == 1) & pad_mask)
            #
            # crf_labels, crf_mask = to_crf_pad(labels, loss_mask, -100)
            # crf_logits, _ = to_crf_pad(emissions, loss_mask, -100)
            #
            # loss = self.crf.neg_log_likelihood(crf_logits, crf_mask, crf_labels)
            # # removing mask stuff from the output path is done later in my_crf_ner but it should be kept away
            # # when calculating loss
            # best_path = self.crf(crf_logits, crf_mask)  # (torch.ones(logits.shape) == 1)
            # best_path = unpad_crf(best_path, crf_mask, labels, pad_mask)
            return {'loss': loss, 'tag': best_path, 'logits': emissions}
        else:
            tag = self.crf.decode(emissions)
            best_path = torch.tensor(tag)
            # removing mask stuff from the output path is done later in my_crf_ner but it should be kept away
            # when calculating loss
            # if attention_mask is not None:
            #     mask = (attention_mask == 1)  # & (labels!=-100))
            # else:
            #     mask = torch.ones(emissions.shape).bool()  # (labels!=-100)
            # crf_logits, crf_mask = to_crf_pad(emissions, mask, -100)
            # crf_mask = crf_mask.sum(axis=2) == crf_mask.shape[2]
            # best_path = self.crf(crf_logits, crf_mask)
            # temp_labels = torch.ones(mask.shape, dtype=torch.long, device=best_path.device) * -100
            # best_path = unpad_crf(best_path, crf_mask, temp_labels, mask)
            return {'tag': best_path,'logits': emissions}
