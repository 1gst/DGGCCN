import os

import numpy as np
import torch
from torch import nn
from transformers import AutoModel, AutoConfig

from models.net_gp_cnn import CAILNet as NetGPCNN
from models.net_gp import CAILNet as NetGP
from utils.pub_utils import regularization_match


class CAILNet_Fusion(nn.Module):
    def __init__(self, models_name):
        super(CAILNet_Fusion, self).__init__()
        models = models_name.split('+')
        self.models_num = len(models)
        self.models = []
        for i, model in enumerate(models):
            model_path = os.path.join('best_models', model)
            self.config = AutoConfig.from_pretrained(model_path, output_hidden_states=True)
            if 'cnn' in model_path:
                self.config.kernel_size = 5
                self.config.depth = 3
                self.config.head_size = 64
                self.models.append(
                    NetGPCNN.from_pretrained(pretrained_model_name_or_path=model_path, config=self.config).cuda())
            else:
                self.models.append(
                    NetGP.from_pretrained(pretrained_model_name_or_path=model_path, config=self.config).cuda())

    def forward(self, input_ids, attention_mask, token_type_ids=None, gp_masks=None,triggers=None):
        """
        前向计算
        Args:
            input_ids:
            attention_mask:
            token_type_ids:
            labels:
            gp_masks:
        Returns:
        """
        fusion_logits = None
        for i, model in enumerate(self.models):
            output = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                           gp_masks=gp_masks,triggers=triggers)
            logits=torch.tensor(output['output'])
            if i == 0:
                fusion_logits = logits
            else:
                fusion_logits += logits

        # predictions_logits = regularization_match(fusion_logits, triggers)
        # del logits,output
        logits = fusion_logits.cpu().numpy()
        logits_max = np.max(logits, 1)
        select_labels = np.where(logits_max < 0, 0, 1)
        predictions_logits = (np.argmax(logits, axis=1) + 1) * select_labels
        del logits_max, select_labels, logits
        return {'logits': torch.tensor(predictions_logits, dtype=torch.int8)}
