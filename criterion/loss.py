import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def VideoLoss(element_logits, seq_len, batch_size, labels, device):
    k = np.ceil(seq_len / 8).astype('int32')
    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    instance_logits = torch.zeros(0).to(device)
    for i in range(batch_size):
        tmp, _ = torch.topk(element_logits[i][:seq_len[i]], k=int(k[i]), dim=0)
        instance_logits = torch.cat([instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)
    milloss = torch.mean(torch.sum(-labels * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)
    return milloss


def BackLoss(args, vl_preds, labels, back_click_label, seq_len, device):
    cas_base = vl_preds.permute(0, 2, 1)
    k = np.ceil(seq_len / 8).astype('int32')
    loss_gt_fg_bg_click = []
    for b, v_base in enumerate(cas_base):
        score_base = torch.topk(v_base, int(k[b]), dim=1)[0]
        cls_idxes = torch.where(labels[b] == 1)[0]
        loss_gt_fg_bg_click_b_c = []
        bg_click_idx = torch.where(back_click_label[b] == args.num_class)[0]
        if not bg_click_idx.numel():
            continue
        for cls_idx in cls_idxes:
            bg_click_value = torch.mean(v_base[cls_idx, bg_click_idx])
            fg_click_value = torch.mean(score_base[cls_idx])
            fg_bg_click_value = torch.cat(
                (fg_click_value.unsqueeze(0).unsqueeze(0), bg_click_value.unsqueeze(0).unsqueeze(0)), dim=1)
            label_c = torch.cat((torch.ones((1, 1)), torch.zeros((1, 1))), dim=1).to(device)
            loss_gt_fg_bg_click_b_c.append(
                torch.sum(-label_c * F.log_softmax(fg_bg_click_value, dim=1), dim=1).unsqueeze(0))
        loss_gt_fg_bg_click_b_c = torch.sum(torch.cat(loss_gt_fg_bg_click_b_c, dim=1)).unsqueeze(0).unsqueeze(0)
        loss_gt_fg_bg_click.append(loss_gt_fg_bg_click_b_c)
    loss_gt_fg_bg_click = torch.mean(torch.cat(loss_gt_fg_bg_click, dim=1))
    return loss_gt_fg_bg_click


def ActionLoss(args, vl_preds, action_click_label, device):
    a_label_list,a_logits_list = [], []
    for logits, a_label in zip(vl_preds, action_click_label):
        act_ind = torch.where(a_label > 0)[0]
        if not act_ind.numel():
            continue
        ones = torch.sparse.torch.eye(args.num_class + 1).to(device)
        a_label_list.append(ones.index_select(0, a_label[act_ind]))
        a_logits_list.append(logits[act_ind])
    a_label_list = torch.cat(a_label_list, dim=0)[:, 1:]
    a_logits_list = torch.cat(a_logits_list, dim=0)

    clsloss = -torch.mean(torch.sum(a_label_list * F.log_softmax(a_logits_list, dim=1), dim=1), dim=0)

    return clsloss

class MultiCrossEntropyLoss(nn.Module):
    def __init__(self, size_average=True, ignore_index=-500):
        super(MultiCrossEntropyLoss, self).__init__()

        self.size_average = size_average
        self.ignore_index = ignore_index

    def forward(self, input, target):
        logsoftmax = nn.LogSoftmax(dim=1).to(input.device)

        if self.ignore_index >= 0:
            notice_index = [i for i in range(target.shape[-1]) if i != self.ignore_index]
            output = torch.sum(-target[:, notice_index] * logsoftmax(input[:, notice_index]), 1)
            return torch.mean(output[target[:, self.ignore_index] != 1])
        else:
            output = torch.sum(-target * logsoftmax(input), 1)
            if self.size_average:
                return torch.mean(output)
            else:
                return torch.sum(output)

class FocalLoss(nn.Module):
    def __init__(self, class_num, device, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = torch.ones(class_num, 1).to(device)
        else:
            self.alpha = alpha.to(device)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets, weights=None):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs, dim=-1)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        alpha = self.alpha[ids.data.view(-1)]
        alpha = alpha.view(-1, 1)
        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()
        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        if not weights is None:
            batch_loss = batch_loss * weights.unsqueeze(1)
        if self.size_average:
            loss = batch_loss.sum() / alpha.sum()
        else:
            loss = batch_loss.sum()
        return loss