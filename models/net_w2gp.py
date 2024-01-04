import math

import numpy as np
import torch
from transformers.modeling_utils import PreTrainedModel
from torch import nn
from transformers import AutoModel
import torch.nn.functional as F
from utils.module.global_pointer import MyLoss, GlobalPointer
from utils.pub_utils import regularization_match


class RoPEPositionEncoding(nn.Module):
    """旋转式位置编码: https://kexue.fm/archives/8265
    """

    def get_sinusoid_encoding_table(self, n_position, d_hid, padding_idx=None):
        '''Returns: [seq_len, d_hid]
        '''
        position = torch.arange(0, n_position, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_hid, 2).float() * (-math.log(10000.0) / d_hid))
        embeddings_table = torch.zeros(n_position, d_hid)
        embeddings_table[:, 0::2] = torch.sin(position * div_term)
        embeddings_table[:, 1::2] = torch.cos(position * div_term)
        return embeddings_table

    def __init__(self, max_position, embedding_size):
        super(RoPEPositionEncoding, self).__init__()
        position_embeddings = self.get_sinusoid_encoding_table(max_position, embedding_size)  # [seq_len, hdsz]
        cos_position = position_embeddings[:, 1::2].repeat_interleave(2, dim=-1)
        sin_position = position_embeddings[:, ::2].repeat_interleave(2, dim=-1)
        # register_buffer是为了最外层model.to(device)，不用内部指定device
        self.register_buffer('cos_position', cos_position)
        self.register_buffer('sin_position', sin_position)

    def forward(self, qw, seq_dim=1):
        # 默认最后两个维度为[seq_len, hdsz]
        seq_len = qw.shape[seq_dim]
        qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], dim=-1).reshape_as(qw)
        return qw * self.cos_position[:seq_len] + qw2 * self.sin_position[:seq_len]


class LayerNorm(nn.Module):
    def __init__(self, input_dim, cond_dim=0, center=True, scale=True, epsilon=None, conditional=False,
                 hidden_units=None, hidden_activation='linear', hidden_initializer='xaiver', **kwargs):
        super(LayerNorm, self).__init__()
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
        self.position_embedding = RoPEPositionEncoding(512, input_dim)

        if self.center:
            self.beta = nn.Parameter(torch.zeros(input_dim))
        if self.scale:
            self.gamma = nn.Parameter(torch.ones(input_dim))

        if self.conditional:
            if self.hidden_units is not None:
                self.hidden_dense = nn.Linear(in_features=self.cond_dim, out_features=self.hidden_units, bias=False)
            if self.center:
                self.beta_dense = nn.Linear(in_features=self.cond_dim, out_features=input_dim, bias=False)
            if self.scale:
                self.gamma_dense = nn.Linear(in_features=self.cond_dim, out_features=input_dim, bias=False)

        self.initialize_weights()

    def initialize_weights(self):

        if self.conditional:
            if self.hidden_units is not None:
                if self.hidden_initializer == 'normal':
                    torch.nn.init.normal(self.hidden_dense.weight)
                elif self.hidden_initializer == 'xavier':  # glorot_uniform
                    torch.nn.init.xavier_uniform_(self.hidden_dense.weight)

            if self.center:
                torch.nn.init.constant_(self.beta_dense.weight, 0)
            if self.scale:
                torch.nn.init.constant_(self.gamma_dense.weight, 0)

    def forward(self, inputs, cond=None):
        if self.conditional:
            if self.hidden_units is not None:
                cond = self.hidden_dense(cond)

            for _ in range(len(inputs.shape) - len(cond.shape)):
                cond = cond.unsqueeze(1)  # cond = K.expand_dims(cond, 1)

            if self.center:
                beta = self.beta_dense(cond) + self.beta
            if self.scale:
                gamma = self.gamma_dense(cond) + self.gamma
        else:
            if self.center:
                beta = self.beta
            if self.scale:
                gamma = self.gamma

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
    def __init__(self, input_size, channels, dilation, dropout=0.1):
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
    def __init__(self, n_in, n_out, dropout=0.):
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
    def __init__(self, cls_num, channels, ffnn_hid_size, dropout=0.):
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
        self.conv_hid_size = 128
        self.dilation = [1, 2, 3]
        self.conv_dropout = 0.2
        self.dist_emb_size = 64
        self.ffnn_hid_size = 128
        self.out_dropout = 0.2
        self.encoder_hidden_size = config.hidden_size * 2

        # 加载预训练模型
        self.bert = AutoModel.from_config(config=self.config)
        # dropout与其他网络模块
        # 嵌入和卷积层
        self.dist_embs = nn.Embedding(20, self.dist_emb_size)
        self.convLayer = ConvolutionLayer(self.encoder_hidden_size + self.dist_emb_size, self.conv_hid_size,
                                          self.dilation,
                                          self.conv_dropout)
        self.cln = LayerNorm(self.encoder_hidden_size, self.encoder_hidden_size, conditional=True)
        # 卷积层的预测器
        self.predictor = CoPredictor(class_num, self.conv_hid_size * len(self.dilation), self.ffnn_hid_size,
                                     self.out_dropout)

        self.dropout = nn.Dropout(dropout_rate)
        self.criterion = MyLoss()
        if self.config.model_type != 'deberta-v2' and self.config.model_type != 'roformer':
            self.global_pointer = GlobalPointer(hidden_size=self.encoder_hidden_size, heads=class_num,
                                                head_size=64)
        else:
            self.global_pointer = GlobalPointer(hidden_size=self.config.hidden_size, heads=class_num,
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
        hidden_state = outputs.get('last_hidden_state')
        if self.config.model_type != 'deberta-v2' and self.config.model_type != 'roformer':
            cls_output = outputs.get('pooler_output').unsqueeze(1).repeat(1, hidden_state.size()[1], 1)
            hidden_state = torch.cat([hidden_state, cls_output], dim=-1)
        # 使用global_pointer，得到global pointer的输出
        if gp_masks is None:
            gp_logits = self.global_pointer(self.dropout(hidden_state), attention_mask.gt(0).long()).contiguous()
        else:
            gp_logits = self.global_pointer(self.dropout(hidden_state), gp_masks.gt(0).long()).contiguous()
        # 得到卷积和卷积预测层的输出
        cln = self.cln(hidden_state.unsqueeze(2), hidden_state)# [batch_size,seq_len,seq_len,1536]
        dis_emb = self.dist_embs(dist_inputs) # [batch_size,seq_len,seq_len,64]
        conv_inputs = torch.cat([dis_emb, cln], dim=-1) # [batch_size,seq_len,seq_len,1600]
        conv_outputs = self.convLayer(conv_inputs) # [batch_size,seq_len,seq_len,128*3]
        conv_logits = self.predictor(conv_outputs)# [batch_size,108,seq_len,seq_len]
        # 合并两个的输出
        logits = gp_logits + conv_logits
        if labels is None:
            # 提取出logit中的结果，与数据构造的标签对应
            logits = logits.cpu().detach().numpy()
            logits_max = np.max(logits, 1)
            select_labels = np.where(logits_max < 0, 0, 1)
            predictions_logits = (np.argmax(logits, axis=1) + 1) * select_labels
            del logits_max, select_labels
            # predictions_logits = regularization_match(logits, triggers)
            return {'logits': torch.tensor(predictions_logits, dtype=torch.int8), 'output': torch.tensor(logits)}
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
