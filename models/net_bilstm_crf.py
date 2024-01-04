import torch
from torch.nn import CrossEntropyLoss
from transformers.modeling_utils import PreTrainedModel
from torch import nn
from transformers import AutoModel, AutoModelForMaskedLM, BertModel
from torchcrf import CRF


# from utils.crf import to_crf_pad, unpad_crf


class CAILNet(PreTrainedModel):
    def __init__(self, config, dropout_rate=0.2, class_num=217):
        super(CAILNet, self).__init__(config)
        self.max_len = config.max_len
        self.config = config
        self.lstm_out = int(config.hidden_size / 2)
        # 加载预训练模型
        # self.bert = AutoModelForMaskedLM.from_pretrained(model_path, config=self.config)
        self.bert = AutoModel.from_config(config=self.config)
        self.lstm = nn.LSTM(input_size=config.hidden_size, hidden_size=self.lstm_out, num_layers=1,
                            bidirectional=True,
                            batch_first=True)
        # dropout与分类网络
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.config.hidden_size, class_num)
        self._init_weights(self.classifier)
        # crf层
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
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                            return_dict=True)
        # 得到模型的最后一层没有softmax的隐藏层输出
        hidden_state = outputs.get('last_hidden_state')
        hidden_state, _ = self.lstm(hidden_state)
        # 得到模型的最后一层没有softmax的cls输出
        # cls_output = outputs.get('pooler_output').unsqueeze(1).repeat(1, hidden_state.size()[1], 1)
        # hidden_state = torch.cat([hidden_state, cls_output], dim=-1)
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
            return {'loss': loss, 'tag': best_path, 'logits': emissions}
        else:
            # removing mask stuff from the output path is done later in my_crf_ner but it should be kept away
            # when calculating loss
            tag = self.crf.decode(emissions)
            best_path = torch.tensor(tag)
            return {'tag': best_path, 'logits': emissions}
