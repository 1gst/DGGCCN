import numpy as np
import torch
from transformers.modeling_utils import PreTrainedModel
from torch import nn
from transformers import AutoModel

from utils.module.global_pointer_attn import MyLoss, GlobalPointer
from utils.pub_utils import GlobalVariable
from utils.pub_utils import regularization_match


class CAILNet(PreTrainedModel):
    def __init__(self, config, dropout_rate=0.2, class_num=108):
        super(CAILNet, self).__init__(config)
        self.max_len = config.max_len
        self.config = config
        self.class_num = class_num
        # 加载预训练模型
        # self.bert = AutoModelForMaskedLM.from_pretrained(model_path, config=self.config)
        self.bert = AutoModel.from_config(config=self.config)
        # dropout与分类网络
        self.dropout = nn.Dropout(dropout_rate)
        self.criterion = MyLoss()
        if self.config.model_type != 'deberta-v2' and self.config.model_type != 'roformer':
            self.global_pointer = GlobalPointer(hidden_size=self.config.hidden_size * 2,
                                                event_hidden_size=self.config.hidden_size, heads=class_num,
                                                head_size=64)
        else:
            self.global_pointer = GlobalPointer(hidden_size=self.config.hidden_size,
                                                event_hidden_size=self.config.hidden_size, heads=class_num,
                                                head_size=64)

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

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None, gp_masks=None, triggers=None):
        """
        前向计算
        :param input_ids:[batch_size,seq_len]
        :param token_type_ids:[batch_size,seq_len]
        :param attention_mask:[batch_size,seq_len]
        :return:
        """
        # 清空gpu空闲内存
        # if hasattr(torch.cuda, 'empty_cache'):
        #     torch.cuda.empty_cache()
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                            return_dict=True)
        event_inputs = GlobalVariable.event_text_input
        event_input_ids = event_inputs['input_ids']
        event_attention_mask = event_inputs['attention_mask']
        event_token_type_ids = event_inputs['token_type_ids']
        event_outputs = self.bert(input_ids=event_input_ids, attention_mask=event_attention_mask,
                                  token_type_ids=event_token_type_ids,
                                  return_dict=True)
        event_hidden_state = event_outputs.get('pooler_output')
        # 得到模型的最后一层没有softmax的隐藏层输出
        hidden_state = outputs.get('last_hidden_state')
        if self.config.model_type != 'deberta-v2' and self.config.model_type != 'roformer':
            cls_output = outputs.get('pooler_output').unsqueeze(1).repeat(1, hidden_state.size()[1], 1)
            hidden_state = torch.cat([hidden_state, cls_output], dim=-1)
        # 使用global_pointer
        if gp_masks is None:
            logits = self.global_pointer(self.dropout(hidden_state), event_hidden_state,
                                         attention_mask.gt(0).long()).contiguous()
        else:
            logits = self.global_pointer(self.dropout(hidden_state), event_hidden_state,
                                         gp_masks.gt(0).long()).contiguous()
        if labels is None:
            # 提取出logit中的结果，与数据构造的标签对应
            # logits = logits.cpu().detach().numpy()
            # logits_max = np.max(logits, 1)
            # select_labels = np.where(logits_max < 0, 0, 1)
            # predictions_logits = (np.argmax(logits, axis=1) + 1) * select_labels
            # del logits_max, select_labels, logits
            predictions_logits = regularization_match(logits, triggers)
            return {'logits': torch.tensor(predictions_logits, dtype=torch.int8)}
        extend_labels = torch.zeros((labels.size()[0], labels.size()[1], labels.size()[2], self.class_num + 1),
                                    device=labels.device, dtype=torch.int8)
        labels = labels.unsqueeze(-1)
        extend_labels.scatter_(-1, labels, 1)
        # 交换维度
        extend_labels = extend_labels.permute(0, 3, 1, 2)
        extend_labels = extend_labels[:, 1:, :, :].contiguous()
        loss = self.criterion(logits, extend_labels)
        # 提取出logit中的结果，与数据构造的标签对应
        logits = logits.cpu().detach().numpy()
        logits_max = np.max(logits, 1)
        select_labels = np.where(logits_max < 0, 0, 1)
        predictions_logits = (np.argmax(logits, axis=1) + 1) * select_labels
        del logits_max, select_labels, extend_labels, logits
        return {'loss': loss, 'logits': torch.tensor(predictions_logits)}
