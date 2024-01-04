import torch
import torch.nn as nn
import torch.nn.functional as F

# 参考https://github.com/aotumanbiu/Multiclass-Focalloss/blob/main/focal_loss.py
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=3, size_average=True):
        super(FocalLoss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha) == num_classes
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1 - alpha)  # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]
        self.gamma = gamma

    def forward(self, preds, labels):
        assert preds.dim() == 2 and labels.dim() == 1

        self.alpha = self.alpha.to(preds.device)
        log_soft = F.log_softmax(preds, dim=1)
        soft = torch.exp(log_soft).gather(1, labels.view(-1, 1))
        nulloss = F.nll_loss(log_soft, labels, weight=self.alpha, reduction="none").view(-1, 1)

        # pt = torch.exp(-nn.CrossEntropyLoss(reduction="none")(preds, labels)).view(-1, 1)
        # log_pt = nn.CrossEntropyLoss(weight=self.alpha, reduction="none")(preds, labels).view(-1, 1)
        # loss = torch.mul((1 - pt) ** self.gamma, log_pt)

        loss = torch.mul((1 - soft) ** self.gamma, nulloss)

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss