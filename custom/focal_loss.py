import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss2d(nn.Module):

    def __init__(self, gamma=2, size_average=True, ignore_label=None):
        super(FocalLoss2d, self).__init__()

        self.gamma = gamma
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.contiguous().view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)
            input = input.contiguous().view(-1, input.size(2)).squeeze()
        if target.dim() == 4:
            target = target.contiguous().view(target.size(0), target.size(1), -1)
            target = target.transpose(1, 2)
            target = target.contiguous().view(-1, target.size(2)).squeeze()
        elif target.dim() == 3:
            target = target.view(-1)
        else:
            target = target.view(-1, 1)

        # compute the negative likelyhood
        logpt = -F.cross_entropy(input, target, ignore_index=self.ignore_label)
        pt = torch.exp(logpt)

        # compute the loss
        loss = -((1 - pt) ** self.gamma) * logpt

        # averaging (or not) loss
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
