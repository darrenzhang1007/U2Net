import torch
import torch.nn as nn
import torch.nn.functional as F


def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
    bce_loss = nn.BCELoss(size_average=True)
    loss0 = bce_loss(d0, labels_v)
    loss1 = bce_loss(d1, labels_v)
    loss2 = bce_loss(d2, labels_v)
    loss3 = bce_loss(d3, labels_v)
    loss4 = bce_loss(d4, labels_v)
    loss5 = bce_loss(d5, labels_v)
    loss6 = bce_loss(d6, labels_v)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n" %( \
        loss0.data.item(), loss1.data.item(), loss2.data.item(), loss3.data.item(), \
        loss4.data.item(), loss5.data.item(), loss6.data.item()))

    return loss0, loss


def weighted_bce(logit, truth):
    logit = logit.view(-1)
    truth = truth.view(-1)
    assert(logit.shape == truth.shape)

    loss = F.binary_cross_entropy_with_logits(logit, truth, reduction='none')
    if 0:
        loss = loss.mean()
    if 1:
        pos = (truth > 0.5).float()
        neg = (truth < 0.5).float()
        pos_weight = pos.sum().item() + 1e-12
        neg_weight = neg.sum().item() + 1e-12
        loss = (0.01*pos*loss/pos_weight + 0.99*neg*loss/neg_weight).sum()
    return loss


def soft_dice_loss(logit, truth, weight=[0.2, 0.8]):
    batch_size = logit.size()[0]
    logit = logit.view(batch_size, -1)
    truth = truth.view(batch_size, -1)
    assert(logit.shape == truth.shape)

    p = torch.sigmoid(logit)
    t = truth
    w = truth.detach()
    w = w*(weight[1]-weight[0])+weight[0]

    p = w*(p*2-1)  # convert to [0,1] --> [-1, 1]
    t = w*(t*2-1)

    intersection = (p * t).sum(-1)
    union = (p * p).sum(-1) + (t * t).sum(-1)
    dice = 1 - 2*intersection/union

    loss = dice.mean()
    return loss


def dice_loss(logit, target):
    logit = torch.sigmoid(logit)
    smooth = 1.

    iflat = logit.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    return 1 - ((2. * intersection + smooth) /
                (iflat.sum() + tflat.sum() + smooth))


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

# class FocalLoss(nn.Module):
#     def __init__(self, gamma=2):
#         super().__init__()
#         self.gamma = gamma

#     def forward(self, input, target):
#         if not (target.size() == input.size()):
#             raise ValueError("Target size ({}) must be the same as input size ({})"
#                              .format(target.size(), input.size()))

#         max_val = (-input).clamp(min=0)
#         loss = input - input * target + max_val + \
#             ((-max_val).exp() + (-input - max_val).exp()).log()

#         invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
#         loss = (invprobs * self.gamma).exp() * loss

#         return loss.sum(dim=1).mean()


def f1_loss(logits, labels):
    __small_value = 1e-6
    beta = 1
    batch_size = logits.size()[0]
    p = F.sigmoid(logits)
    l = labels
    num_pos = torch.sum(p, 1) + __small_value
    num_pos_hat = torch.sum(l, 1) + __small_value
    tp = torch.sum(l * p, 1)
    precise = tp / num_pos
    recall = tp / num_pos_hat
    fs = (1 + beta * beta) * precise * recall / (beta * beta * precise + recall + __small_value)
    loss = fs.sum() / batch_size
    return (1 - loss)
