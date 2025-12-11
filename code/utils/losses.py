import torch
import torch.nn
from torch.nn import functional as F
import numpy as np

"""
The different uncertainty methods loss implementation.
Including:
    Ignore, Zeros, Ones, SelfTrained, MultiClass
"""

METHODS = ['U-Ignore', 'U-Zeros', 'U-Ones', 'U-SelfTrained', 'U-MultiClass']
CLASS_NAMES = [ 'ADI', 'BACK', 'LYM', 'STR', 'DEB', 'MUC', 'TUM','MUS','NORM']
CLASS_NUM = [7338,7381,8144,7315,8037,6163,10033,9489,6100]
CLASS_WEIGHT = torch.Tensor([(70000/i) for i in CLASS_NUM]).cuda()

class cross_entropy_loss(object):
    """
    map all uncertainty values to a unique value "2"
    """
    
    def __init__(self):
        self.base_loss = torch.nn.CrossEntropyLoss(weight=CLASS_WEIGHT,reduction='mean') #  
    
    def __call__(self, output, target):
        # target[target == -1] = 2
        output_softmax = F.softmax(output, dim=1)
        target = torch.argmax(target, dim=1)
        return self.base_loss(output_softmax, target.long())


class DBLLoss(torch.nn.Module):
    def __init__(self, major_classes,gamma=0.5, scale=0.4,reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.scale=scale
        self.major_classes=major_classes

    def forward(self, logits, target, maw_raw=None,daw=None):
        log_probs = F.log_softmax(logits, dim=1)
        probs     = log_probs.exp()
        if target.dim() == 2:
            pt     = torch.sum(probs * target, dim=1)
            log_pt = torch.sum(log_probs * target, dim=1)
        else:
            target = target.view(-1, 1)
            pt     = probs.gather(1, target).squeeze(1)
            log_pt = log_probs.gather(1, target).squeeze(1)
        if daw is None:
            daw = (1 - pt) ** self.gamma  
        if maw_raw is not None:
            targets_detach = target.argmax(dim=1).detach() 
            maw = maw_raw[targets_detach] 

            major_classes_tensor = torch.tensor(self.major_classes, device=target.device)
            is_major = (targets_detach.unsqueeze(1) == major_classes_tensor).any(dim=1) 

            min_tensor = torch.minimum(daw, maw)
            max_tensor = torch.maximum(daw, maw)

            final_weights = torch.where(is_major, min_tensor, max_tensor)
        else:
            final_weights=daw

        loss = -final_weights * log_pt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss 

