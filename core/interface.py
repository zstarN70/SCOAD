import os

import torch
import torch.nn as nn
import numpy as np

from misc.convert_label import create_test_target
from misc.eval import frame_level_map_n_cap_thumos, getASfromCAS, compute_PAP_result_thumos14

torch.set_default_tensor_type('torch.cuda.FloatTensor')


def to_device(x, device):
    return x.unsqueeze(0).to(device)


def test_full(dataset, model, device, fps):
    softmax = nn.Softmax(dim=-1).to(device)
    classnum = dataset.num_class + 1
    scores, scores_metrics, target_metrics, videoIds = [], [], [], []
    done = False

    while not done:
        if dataset.currenttestidx % 100 == 0:
            print('Testing test data point %d of %d' % (dataset.currenttestidx, len(dataset.testidx)))
        working_idx = dataset.testidx[dataset.currenttestidx]
        vname = dataset.video_name[working_idx].decode('utf-8')
        features, labels, done = dataset.load_data(is_training=False)

        vid = vname.split('_')[-1]
        videoIds.extend([int(vid) for i in range(features.shape[0])])

        features = torch.from_numpy(features).float().to(device)
        target = create_test_target(dataset, working_idx, features.shape[0], fps)
        enc_hx = to_device(torch.zeros(model.hidden_size), device)
        element_logits = []
        for l in range(0, features.shape[0]):
            with torch.no_grad():
                enc_score, mask_score, enc_hx = \
                    model.decode(to_device(features[l], device), enc_hx)
            element_logits.append(enc_score.cpu().numpy()[0])
            enc_score = softmax(enc_score).cpu().numpy()[0]
            mask_score = softmax(mask_score).cpu().numpy()[0]

            scores_metrics.append(enc_score)  # per frame action score
            target_metrics.append(target[l])  # per frame action label
            start_score = np.zeros_like(enc_score)
            start_score[0] = enc_score[0] * mask_score[0]
            start_score[1:] = enc_score[1:] * mask_score[1]
            scores.append(start_score)  # action start score

    frameScores = np.array(scores)
    dist_ths = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    GTs = np.load(os.path.join(dataset.path_to_annotations, 'TH14_ASlbs_test.npy'), allow_pickle=True).tolist()
    scores, times, videoLen = getASfromCAS(frameScores, videoIds, fps)  # generate action starts
    for dist_th in dist_ths:
        result_point = compute_PAP_result_thumos14(GTs, videoLen, scores, times, videoIds, dist_th, classnum, ignore=[0])
        print('Test point mAP @ dist_th = ' + str(dist_th), result_point['mAP'])

    result = {'probs': np.asarray(scores_metrics).T, 'labels': np.asarray(target_metrics).T}
    map, aps = frame_level_map_n_cap_thumos(result)
    print('Test frame mAP', map)
