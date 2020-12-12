import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss2d(nn.Module):

    def __init__(self, gamma=2, size_average=True, ignore_label=255):
        super(FocalLoss2d, self).__init__()

        self.gamma = gamma
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, input, target):
        # compute the negative likelyhood
        logpt = -F.cross_entropy(input, target, ignore_index=self.ignore_label, reduction='none')
        pt = torch.exp(logpt)
        # compute the loss
        loss = -((1 - pt) ** self.gamma) * logpt

        # averaging (or not) loss
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class FocalLoss_Ori(nn.Module):
    """
    https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/focal_loss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, gamma=2, size_average=True, ignore_label=None):
        super(FocalLoss_Ori, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.eps = 1e-6
        self.ignore_label = ignore_label

    def forward(self, logit, target):

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.transpose(1, 2).contiguous()  # [N,C,d1*d2..] -> [N,d1*d2..,C]
            logit = logit.view(-1, logit.size(-1))  # [N,d1*d2..,C]-> [N*d1*d2..,C]
        target = target.view(-1, 1)  # [N,d1,d2,...]->[N*d1*d2*...,1]
        if self.ignore_label:
            pass
        # -----------legacy way------------
        #  idx = target.cpu().long()
        # one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
        # one_hot_key = one_hot_key.scatter_(1, idx, 1)
        # if one_hot_key.device != logit.device:
        #     one_hot_key = one_hot_key.to(logit.device)
        # pt = (one_hot_key * logit).sum(1) + epsilon

        # ----------memory saving way--------
        pt = logit.gather(1, target.long()).view(-1) + self.eps  # avoid apply
        logpt = pt.log()

        loss = -1 * torch.pow(torch.sub(1.0, pt), self.gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


if __name__ == '__main__':
    x = torch.rand([10, 3, 321, 321])
    target = torch.zeros([10, 321, 321]).long()
    target[:, :, 10] = 255
    focal_loss = FocalLoss2d()
    y = focal_loss(x, target)
