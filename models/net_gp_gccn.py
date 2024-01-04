import numpy as np
import torch
from transformers.modeling_utils import PreTrainedModel
from torch import nn
from transformers import AutoModel
import torch.nn.functional as F

from utils.module.GCN import GCN
from utils.module.global_pointer import MyLoss, GlobalPointer
from utils.pub_utils import regularization_match


class ConLayerNorm(nn.Module):
    def __init__(self, input_dim, cond_dim=0, center=True, scale=True, epsilon=None, conditional=False,
                 hidden_units=None, hidden_activation='linear', hidden_initializer='xaiver'):
        super().__init__()
        """
        input_dim: inputs.shape[-1]
        cond_dim: cond.shape[-1]
        """
        self.center = center
        self.scale = scale
        self.conditional = conditional
        self.hidden_units = hidden_units
        self.hidden_initializer = hidden_initializer
        self.epsilon = epsilon or 1e-12
        self.input_dim = input_dim
        self.cond_dim = cond_dim

        if self.center:
            self.bias = nn.Parameter(torch.zeros(input_dim))
        if self.scale:
            self.weight = nn.Parameter(torch.ones(input_dim))

        if self.conditional:
            if self.hidden_units is not None:
                self.hidden_dense = nn.Linear(in_features=self.cond_dim, out_features=self.hidden_units, bias=False)
            if self.center:
                self.bias_dense = nn.Linear(in_features=self.cond_dim, out_features=input_dim, bias=False)
            if self.scale:
                self.weight_dense = nn.Linear(in_features=self.cond_dim, out_features=input_dim, bias=False)

        self.initialize_weights()

    def initialize_weights(self):

        if self.conditional:
            if self.hidden_units is not None:
                if self.hidden_initializer == 'normal':
                    torch.nn.init.normal(self.hidden_dense.weight)
                elif self.hidden_initializer == 'xavier':  # glorot_uniform
                    torch.nn.init.xavier_uniform_(self.hidden_dense.weight)

            if self.center:
                torch.nn.init.constant_(self.bias_dense.weight, 0)
            if self.scale:
                torch.nn.init.constant_(self.weight_dense.weight, 0)

    def forward(self, inputs, cond=None):
        if self.conditional:
            if self.hidden_units is not None:
                cond = self.hidden_dense(cond)

            for _ in range(len(inputs.shape) - len(cond.shape)):
                cond = cond.unsqueeze(1)  # cond = K.expand_dims(cond, 1)

            if self.center:
                beta = self.bias_dense(cond) + self.bias
            if self.scale:
                gamma = self.weight_dense(cond) + self.weight
        else:
            if self.center:
                beta = self.bias
            if self.scale:
                gamma = self.weight

        outputs = inputs
        if self.center:
            mean = torch.mean(outputs, dim=-1).unsqueeze(-1)
            outputs = outputs - mean
        if self.scale:
            variance = torch.mean(outputs ** 2, dim=-1).unsqueeze(-1)
            std = (variance + self.epsilon) ** 0.5
            outputs = outputs / std
            outputs = outputs * gamma
        if self.center:
            outputs = outputs + beta

        return outputs


class ConvolutionLayer(nn.Module):
    def __init__(self, input_size, channels, dilation, dropout=0.2):
        '''

        :param input_size: B,L,L,808
        :param channels:96
        :param dilation:[1, 2, 3]
        :param dropout:0.5
        '''
        super(ConvolutionLayer, self).__init__()
        self.base = nn.Sequential(
            nn.Dropout2d(dropout),
            nn.Conv2d(input_size, channels, kernel_size=1),
            nn.GELU(),
        )

        self.convs = nn.ModuleList(
            [nn.Conv2d(channels, channels, kernel_size=3, groups=channels, dilation=d, padding=d) for d in dilation])

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.base(x)

        outputs = []
        for conv in self.convs:
            x = conv(x)
            x = F.gelu(x)
            outputs.append(x)
        outputs = torch.cat(outputs, dim=1)
        outputs = outputs.permute(0, 2, 3, 1).contiguous()
        return outputs


class MLP(nn.Module):
    def __init__(self, n_in, n_out, dropout=0.2):
        super().__init__()

        self.linear = nn.Linear(n_in, n_out)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear(x)
        x = self.activation(x)
        return x


class CoPredictor(nn.Module):
    def __init__(self, cls_num, channels, ffnn_hid_size, dropout=0.2):
        super().__init__()
        self.mlp_rel = MLP(channels, ffnn_hid_size, dropout=dropout)
        self.linear = nn.Linear(ffnn_hid_size, cls_num)
        self.dropout = nn.Dropout(dropout)

    def forward(self, z):
        z = self.dropout(self.mlp_rel(z))
        o2 = self.linear(z)
        o2 = o2.permute(0, 3, 1, 2)
        return o2


class CAILNet(PreTrainedModel):
    def __init__(self, config, dropout_rate=0.2, class_num=108):
        super(CAILNet, self).__init__(config)

        # 设置网络的超参数
        self.max_len = config.max_len
        self.config = config
        self.class_num = class_num
        self.conv_dropout = 0.2
        self.dilation = [1, 2, 5]
        # self.dilation = [1, 3, 4]
        self.dist_emb_size = 64
        # self.ffnn_hid_size = 128
        self.ffnn_hid_size = 768
        self.conv_hid_size = 128
        self.out_dropout = 0.2
        self.encoder_hidden_size = config.hidden_size*2
        self.lstm_out = config.hidden_size // 2

        # 加载预训练模型
        self.bert = AutoModel.from_config(config=self.config)

        # self.lstm = nn.LSTM(input_size=config.hidden_size, hidden_size=self.lstm_out, num_layers=1,
        #                     bidirectional=True,
        #                     batch_first=True)
        # dropout与其他网络模块
        # 嵌入和gcn层
        # self.dist_embs = nn.Embedding(20, self.dist_emb_size)
        self.gcn = GCN(self.encoder_hidden_size, 768, self.conv_dropout, self.encoder_hidden_size)
        # self.gat = GAT(self.encoder_hidden_size, 768, self.encoder_hidden_size, 0.2, 0.2, 4)
        self.cln = ConLayerNorm(self.encoder_hidden_size, self.encoder_hidden_size, epsilon=1e-12, conditional=True)
        # self.cnn = MaskCNN(self.encoder_hidden_size, self.encoder_hidden_size, kernel_size=5, depth=1)
        # self.attention = CBAM(gate_channels=class_num)
        self.convLayer = ConvolutionLayer(self.encoder_hidden_size, self.conv_hid_size,
                                          self.dilation,
                                          self.conv_dropout)
        # self.cbma = CBAM(gate_channels=class_num)
        # # 卷积层的预测器
        self.predictor = CoPredictor(class_num, self.conv_hid_size * len(self.dilation), self.ffnn_hid_size,
                                     self.out_dropout)
        # self.predictor = CoPredictor(class_num, self.encoder_hidden_size, self.ffnn_hid_size,
        #                              self.out_dropout)

        self.criterion = MyLoss()
        self.dropout = nn.Dropout(dropout_rate)
        self.global_pointer = GlobalPointer(hidden_size=self.encoder_hidden_size, heads=class_num,
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

    def forward(self, input_ids, attention_mask, dist_inputs, token_type_ids=None, labels=None, gp_masks=None,
                adjs=None,
                triggers=None):
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
        last_hidden_state = outputs.get('last_hidden_state')
        if self.config.model_type != 'deberta-v2' and self.config.model_type != 'roformer':
            cls_output = outputs.get('pooler_output').unsqueeze(1).repeat(1, last_hidden_state.size()[1], 1)
            hidden_state = torch.cat([last_hidden_state, cls_output], dim=-1)
        else:
            cls_output = last_hidden_state[:, 0, :].unsqueeze(1).repeat(1, last_hidden_state.size()[1], 1)
            hidden_state = torch.cat([last_hidden_state, cls_output], dim=-1)
        # 使用global_pointer，得到global pointer的输出
        gp_logits = self.global_pointer(self.dropout(hidden_state), gp_masks.gt(0).long()).contiguous()
        gcn_logits = self.gcn(hidden_state, adjs)
        cln = self.cln(gcn_logits.unsqueeze(2), gcn_logits)  # [batch_size,seq_len,seq_len,1536]
        conv_outputs = self.convLayer(cln)  # [batch_size,seq_len,seq_len,128*3]
        # conv_logits = self.predictor(conv_outputs)
        conv_logits = self.predictor(F.gelu(conv_outputs)) # [batch_size,108,seq_len,seq_len]

        # 使用CBMA
        # cbma_inputs = pre_outputs.permute(0, 3, 1, 2)
        # conv_logits = self.cbma(pre_outputs)
        # conv_logits = cnov_inputs.permute(0, 2, 3, 1)

        # 合并两个的输出
        # adjs = adjs.unsqueeze(1) - torch.eye(adjs.size()[1], device=adjs.device)
        # logits = conv_logits.contiguous()
        logits = conv_logits.contiguous() + gp_logits.contiguous()

        if labels is None:
            # 提取出logit中的结果，与数据构造的标签对应
            logits = logits.cpu().detach().numpy()
            logits_max = np.max(logits, 1)
            select_labels = np.where(logits_max > 0, 1, 0)
            predictions_logits = (np.argmax(logits, axis=1) + 1) * select_labels
            del logits_max, select_labels
            # predictions_logits = regularization_match(logits, triggers)
            return {'logits': torch.tensor(predictions_logits, dtype=torch.int8), 'output': torch.tensor(logits)}
        extend_labels = torch.zeros((labels.size()[0], labels.size()[1], labels.size()[2], self.class_num + 1),
                                    device=labels.device, dtype=torch.int8)
        temp_labels=labels.clone()
        temp_labels[temp_labels == -1] = 0  # 将负触发词也替换成0标签
        temp_labels = temp_labels.unsqueeze(-1)
        extend_labels.scatter_(-1, temp_labels, 1)
        # 交换维度
        extend_labels = extend_labels.permute(0, 3, 1, 2)
        extend_labels = extend_labels[:, 1:, :, :].contiguous()

        loss = self.criterion(logits, extend_labels)

        # 使用两个loss
        # loss1 = self.criterion(gp_logits, extend_labels)
        # attention_mask1 = 1 - attention_mask.unsqueeze(1).unsqueeze(3)  # [btz, 1, seq_len, 1]
        # attention_mask2 = 1 - attention_mask.unsqueeze(1).unsqueeze(2)  # [btz, 1, 1, seq_len]
        # conv_logits = conv_logits.masked_fill(attention_mask1.bool(), value=-float('inf'))
        # conv_logits = conv_logits.masked_fill(attention_mask2.bool(), value=-float('inf'))
        # loss2 = self.criterion(conv_logits.contiguous(), extend_labels)
        # loss = loss1 + loss2

        # 提取出logit中的结果，与数据构造的标签对应
        logits = logits.cpu().detach().numpy()
        logits_max = np.max(logits, 1)
        select_labels = np.where(logits_max > 0, 1, 0)
        predictions_logits = (np.argmax(logits, axis=1) + 1) * select_labels
        del logits_max, select_labels, extend_labels, logits
        # predictions_logits = regularization_match(logits, triggers)
        return {'loss': loss, 'logits': torch.tensor(predictions_logits)}
