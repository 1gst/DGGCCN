import math
import torch
import torch.nn as nn
from transformers import BertModel

from utils.module.CBAM import CBAM


class MultilabelCategoricalCrossentropy(nn.Module):
    """多标签分类的交叉熵
    说明：y_true和y_pred的shape一致，y_true的元素非0即1， 1表示对应的类为目标类，0表示对应的类为非目标类。
    警告：请保证y_pred的值域是全体实数，换言之一般情况下y_pred不用加激活函数，尤其是不能加sigmoid或者softmax！预测
         阶段则输出y_pred大于0的类。如有疑问，请仔细阅读并理解本文。
    参考：https://kexue.fm/archives/7359
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, y_pred, y_true):
        """ y_true ([Tensor]): [..., num_classes]
            y_pred ([Tensor]): [..., num_classes]
        """
        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_pos = y_pred - (1 - y_true) * 1e12  # 预测对的值没有变，预测错的-1e12
        y_pred_neg = y_pred - y_true * 1e12  # 预测错的值没有变，预测对的-1e12

        y_pred_pos = torch.cat([y_pred_pos, torch.zeros_like(y_pred_pos[..., :1])], dim=-1)
        y_pred_neg = torch.cat([y_pred_neg, torch.zeros_like(y_pred_neg[..., :1])], dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        return (pos_loss + neg_loss).mean()


class MyLoss(MultilabelCategoricalCrossentropy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, y_pred, y_true):
        """
        计算损失
        Args:
            y_pred: batch_size*class_num*seq_len*seq_len
            y_true: batch_size*class_num*seq_len*seq+len

        Returns:

        """
        y_true = y_true.view(y_true.shape[0] * y_true.shape[1], -1)  # [btz*ner_vocab_size, seq_len*seq_len]
        y_pred = y_pred.view(y_pred.shape[0] * y_pred.shape[1], -1)  # [btz*ner_vocab_size, seq_len*seq_len]
        return super().forward(y_pred, y_true)


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    '''Returns: [seq_len, d_hid]
    '''
    position = torch.arange(0, n_position, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_hid, 2).float() * (-math.log(10000.0) / d_hid))
    embeddings_table = torch.zeros(n_position, d_hid)
    embeddings_table[:, 0::2] = torch.sin(position * div_term)
    embeddings_table[:, 1::2] = torch.cos(position * div_term)
    return embeddings_table


class RoPEPositionEncoding(nn.Module):
    """旋转式位置编码: https://kexue.fm/archives/8265
    """

    def __init__(self, max_position, embedding_size):
        super(RoPEPositionEncoding, self).__init__()
        position_embeddings = get_sinusoid_encoding_table(max_position, embedding_size)  # [seq_len, hdsz]
        cos_position = position_embeddings[:, 1::2].repeat_interleave(2, dim=-1)
        sin_position = position_embeddings[:, ::2].repeat_interleave(2, dim=-1)
        # register_buffer是为了最外层model.to(device)，不用内部指定device
        self.register_buffer('cos_position', cos_position)
        self.register_buffer('sin_position', sin_position)

    def forward(self, qw, seq_dim=-2):
        # 默认最后两个维度为[seq_len, hdsz]
        seq_len = qw.shape[seq_dim]
        qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], dim=-1).reshape_as(qw)
        return qw * self.cos_position[:seq_len] + qw2 * self.sin_position[:seq_len]


class EfficientGlobalPointer(nn.Module):
    """更加参数高效的GlobalPointer
    参考：https://kexue.fm/archives/8877
    """

    def __init__(self, hidden_size, heads, head_size, RoPE=True, max_len=512, use_bias=True, tril_mask=True):
        super().__init__()
        self.heads = heads
        self.head_size = head_size
        self.RoPE = RoPE
        self.tril_mask = tril_mask
        self.RoPE = RoPE

        self.p_dense = nn.Linear(hidden_size, head_size * 2, bias=use_bias)
        self.q_dense = nn.Linear(head_size * 2, heads * 2, bias=use_bias)
        if self.RoPE:
            self.position_embedding = RoPEPositionEncoding(max_len, head_size)

    def forward(self, inputs, event_inputs, mask=None):
        ''' inputs: [..., hdsz]
            mask: [bez, seq_len], padding部分为0
        '''
        sequence_output = self.p_dense(inputs)  # [..., head_size*2]
        qw, kw = sequence_output[..., :self.head_size], sequence_output[..., self.head_size:]  # [..., heads, head_size]

        # ROPE编码
        if self.RoPE:
            qw = self.position_embedding(qw)
            kw = self.position_embedding(kw)

        # 计算内积
        logits = torch.einsum('bmd,bnd->bmn', qw, kw) / self.head_size ** 0.5  # [btz, seq_len, seq_len]
        bias_input = self.q_dense(sequence_output)  # [..., heads*2]
        bias = torch.stack(torch.chunk(bias_input, self.heads, dim=-1), dim=-2).transpose(1,
                                                                                          2)  # [btz, heads, seq_len,2]
        logits = logits.unsqueeze(1) + bias[..., :1] + bias[..., 1:].transpose(2,
                                                                               3)  # [btz, heads, seq_len, seq_len]

        # 排除padding
        if mask is not None:
            attention_mask1 = 1 - mask.unsqueeze(1).unsqueeze(3)  # [btz, 1, seq_len, 1]
            attention_mask2 = 1 - mask.unsqueeze(1).unsqueeze(2)  # [btz, 1, 1, seq_len]
            logits = logits.masked_fill(attention_mask1.bool(), value=-float('inf'))
            logits = logits.masked_fill(attention_mask2.bool(), value=-float('inf'))

        # 排除下三角
        if self.tril_mask:
            logits = logits - torch.tril(torch.ones_like(logits), -1) * 1e12

        return logits


class GlobalPointer(nn.Module):
    """全局指针模块
    将序列的每个(start, end)作为整体来进行判断
    参考：https://kexue.fm/archives/8373
    """

    def __init__(self, hidden_size, event_hidden_size, heads, head_size, RoPE=True, max_len=512, use_bias=True,
                 tril_mask=True):
        super().__init__()
        self.heads = heads
        self.head_size = head_size
        self.RoPE = RoPE
        self.tril_mask = tril_mask
        self.RoPE = RoPE

        self.dense = nn.Linear(hidden_size, 2 * head_size, bias=use_bias)
        self.event_dense = nn.Linear(event_hidden_size, head_size, bias=use_bias)
        self.attention = CBAM(gate_channels=108)
        if self.RoPE:
            self.position_embedding = RoPEPositionEncoding(max_len, head_size)

    def forward(self, inputs, event_inputs, mask=None):
        ''' inputs: [..., hdsz]
            mask: [bez, seq_len], padding部分为0
        '''
        # [batchsize, 512, 64*2]
        sequence_output = self.dense(inputs)  # [...,head_size*2]
        event_output = self.event_dense(event_inputs)  # [num_tag,head_size]
        event_output = event_output.unsqueeze(0).repeat(sequence_output.size()[0], 1,
                                                        1)  # [batch_size,num_tag,head_size]
        # qw:[batchsize, 512, 64], kw:[batchsize, 512, 64]
        qw, kw = sequence_output[..., :self.head_size], sequence_output[..., self.head_size:]

        # ROPE编码
        if self.RoPE:
            qw = self.position_embedding(qw)
            kw = self.position_embedding(kw)

        # 计算内积
        logits = torch.einsum('bmd,bnd,bcd->bcmn', qw, kw, event_output)  # [btz, heads, seq_len, seq_len]
        logits = self.attention(logits) + logits

        # 排除padding
        if mask is not None:
            attention_mask1 = 1 - mask.unsqueeze(1).unsqueeze(3)  # [btz, 1, seq_len, 1]
            attention_mask2 = 1 - mask.unsqueeze(1).unsqueeze(2)  # [btz, 1, 1, seq_len]
            logits = logits.masked_fill(attention_mask1.bool(), value=-float('inf'))
            logits = logits.masked_fill(attention_mask2.bool(), value=-float('inf'))

        # 排除下三角
        if self.tril_mask:
            logits = logits - torch.tril(torch.ones_like(logits), -1) * 1e12

        return logits / self.head_size ** 0.5


class GlobalPointerNer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.bert = BertModel.from_pretrained(args.bert_dir, output_hidden_states=True,
                                              hidden_dropout_prob=args.dropout_prob)
        if args.use_efficient_globalpointer == "True":
            self.global_pointer = EfficientGlobalPointer(hidden_size=768, heads=args.num_tags, head_size=args.head_size)
        else:
            self.global_pointer = GlobalPointer(hidden_size=768, heads=args.num_tags, head_size=args.head_size)
        self.criterion = MyLoss()

    def forward(self, token_ids, attention_masks, token_type_ids, labels=None):
        output = self.bert(token_ids, attention_masks, token_type_ids)  # [btz, seq_len, hdsz]
        sequence_output = output[0]
        logits = self.global_pointer(sequence_output, attention_masks.gt(0).long())
        if labels is None:
            # scale返回
            return logits

        loss = self.criterion(logits, labels)
        return loss, logits
