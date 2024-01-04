import torch
from torch.nn import CrossEntropyLoss
from transformers.modeling_utils import PreTrainedModel
from torch import nn
from transformers import AutoModel, AutoModelForMaskedLM, BertModel

from utils.focal_loss import FocalLoss


class CAILNet(PreTrainedModel):
    def __init__(self, config, dropout_rate=0.2, class_num=217):
        super(CAILNet, self).__init__(config)
        self.max_len = config.max_len
        self.config = config
        self.class_num=class_num

        # 加载预训练模型
        # self.bert = AutoModelForMaskedLM.from_pretrained(model_path, config=self.config)
        self.bert = AutoModel.from_config(config=self.config)
        # dropout与分类网络
        self.fc_dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(self.config.hidden_size * 2, class_num)
        self._init_weights(self.fc)

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
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                            return_dict=True)
        # 得到模型的最后一层没有softmax的隐藏层输出
        hidden_state = outputs.get('last_hidden_state')
        # 得到模型的最后一层没有softmax的cls输出
        cls_output = outputs.get('pooler_output').unsqueeze(1).repeat(1, 512, 1)
        hidden_state = torch.cat([hidden_state, cls_output], dim=-1)
        # hidden_state = torch.add(hidden_state, cls_output.unsqueeze(1))
        logits = self.fc(self.fc_dropout(hidden_state))
        if labels is not None:
            # 先声明损失函数
            # loss_fct = CrossEntropyLoss()
            # pre_labels = logits.permute(0, 2, 1)
            # loss = loss_fct(pre_labels, labels)
            #使用FocalLoss
            labels_mask=labels!=-100
            labels=labels.masked_select(mask=labels_mask)
            logits_mask=labels_mask.unsqueeze(-1).expand(logits.size())
            logits_masked=logits.masked_select(mask=logits_mask)
            loss_fct = FocalLoss(alpha=0.25,num_classes=self.class_num)
            pre_labels = logits_masked.view(-1,self.class_num)
            labels=labels.view(-1)
            loss = loss_fct(pre_labels, labels)
            return {'loss': loss, 'logits': logits}
        else:
            return {'logits': logits}
