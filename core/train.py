import torch
import numpy as np
from misc import utils
from misc.getDetectionMAP import getDetectionMAP
from criterion.loss import ActionLoss, BackLoss, VideoLoss, MultiCrossEntropyLoss, FocalLoss


def train(itr, dataset, args, model, optimizer, device, targets, targetsAS):
    features, action_click_label, back_click_label, \
    feat_sts, feat_eds, labels, vname = dataset.load_data(n_similar=args.num_similar)

    seq_len = np.sum(np.max(np.abs(features), axis=2) > 0, axis=1)
    if len(np.where(seq_len == 0)[0]) > 0:
        return

    features = features[:, :np.max(seq_len), :]
    action_click_label = action_click_label[:, :np.max(seq_len)]
    back_click_label = back_click_label[:, :np.max(seq_len)]
    feat_eds = feat_sts + np.max(seq_len)

    features = torch.from_numpy(features).float().to(device)
    labels = torch.from_numpy(labels).long().to(device)

    action_click_label = torch.from_numpy(action_click_label).long().to(device)
    back_click_label = torch.from_numpy(back_click_label).long().to(device)

    scores, a_scores, feats, vl_preds = model(features, device)

    videoloss = VideoLoss(vl_preds, seq_len, args.batch_size, labels, device)
    actionloss = ActionLoss(args, vl_preds, action_click_label, device)
    backloss = BackLoss(args, vl_preds, labels, back_click_label, seq_len, device)

    if targets == None:
        total_loss = backloss + actionloss + videoloss * 0.1
        print('Iteration: %d, total Loss: %.3f, video loss: %.3f, back loss: %.3f action loss: %.3f' % (
            itr, total_loss, videoloss * 0.1, backloss, actionloss))
    else:
        target_proc, targetAS_proc = [], []
        for i, vnm in enumerate(vname):
            st, ed = feat_sts[i], feat_eds[i]
            ed_o = targets[vnm].shape[0]
            target = utils.pad(targets[vnm], ed)
            targetAS = utils.pad(targetsAS[vnm], ed)
            target[ed_o:ed, 0] = 1
            target_proc.append(target[st:ed, :])
            targetAS_proc.append(targetAS[st:ed])

        target_proc = np.array(target_proc)
        targetAS_proc = np.array(targetAS_proc)
        batch_size = target_proc.shape[0]
        tlen = target_proc.shape[1]

        gru_batch = tlen // args.enc_step
        target_gru = np.zeros((batch_size, gru_batch, args.enc_step, target_proc.shape[-1]))
        targetAS_gru = np.zeros((batch_size, gru_batch, args.enc_step, 1))
        for i in range(0, gru_batch):
            target_gru[:, i, :, :] = target_proc[:, i * args.enc_step:(i + 1) * args.enc_step, :]
            targetAS_gru[:, i, :, :] = targetAS_proc[:, i * args.enc_step:(i + 1) * args.enc_step, :]
        target_gru = torch.from_numpy(target_gru).to(device)
        targetAS_gru = torch.from_numpy(targetAS_gru).to(device)
        target_gru = target_gru.view((batch_size * gru_batch, args.enc_step, -1)).transpose(0, 1)
        targetAS_gru = targetAS_gru.view((batch_size * gru_batch, args.enc_step, -1)).transpose(0, 1)

        target_gru = target_gru.contiguous().view((-1, target_gru.shape[-1])).float()
        targetAS_gru = targetAS_gru.contiguous().view((-1, 1)).squeeze(1).long()

        pos_inds = (targetAS_gru.squeeze() != 0).nonzero().squeeze()
        neg_inds = (targetAS_gru.squeeze() == 0).nonzero().squeeze()
        perm = torch.randperm(len(neg_inds))
        if pos_inds.nelement() == 1:
            pos_inds = torch.tensor([pos_inds])
        if pos_inds.nelement() == 0:
            neg_num = 100
        else:
            pos_num = len(pos_inds)
            neg_num = 3 * pos_num
        sample_neg = perm[0:min(neg_num, len(perm))]
        neg_inds = neg_inds[sample_neg]
        train_inds = torch.cat((pos_inds, neg_inds), 0)

        criterion_frame = MultiCrossEntropyLoss().to(device)
        loss_frame = criterion_frame(scores, target_gru)
        criterion_start = FocalLoss(class_num=2, device=device).to(device)
        loss_start = criterion_start(a_scores[train_inds, :], targetAS_gru[train_inds])

        total_loss = loss_frame + loss_start + (backloss + actionloss + videoloss * 0.1)
        print('Iteration: %d, total Loss: %.3f, frame loss: %.3f, start loss: %.3f' % (
            itr, total_loss, loss_frame, loss_start))

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()


def test_train(dataset, model, device):
    done = False
    element_logits_stack = []
    while not done:
        features, click_labels, vname, done = dataset.load_data(is_training=False, is_testing_on_train=True)
        features = torch.from_numpy(features).float().to(device)

        with torch.no_grad():
            _, element_logits = model(features, is_training=False)

        element_logits = element_logits.cpu().data.numpy()

        element_logits_stack.append(element_logits)

    return getDetectionMAP(element_logits_stack, dataset, th=0.5, eval_set='validation', get_det_res=True)
