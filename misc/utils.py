import torch
import numpy as np
import torch.nn.functional as F


def str2ind(categoryname, classlist):
    return [i for i in range(len(classlist)) if categoryname == classlist[i].decode('utf-8')][0]


def strlist2indlist(strlist, classlist):
    return [str2ind(s, classlist) for s in strlist]


def strlist2multihot(strlist, classlist):
    return np.sum(np.eye(len(classlist))[strlist2indlist(strlist, classlist)], axis=0)


def random_extract(feat, t_max):
    r = np.random.randint(len(feat) - t_max)
    res = feat[r:r + t_max]
    return res, r, r + t_max


def pad(feat, min_len):
    if np.shape(feat)[0] <= min_len:
        return np.pad(feat, ((0, min_len - np.shape(feat)[0]), (0, 0)), mode='constant', constant_values=0)
    else:
        return feat


def process_feat(feat, length):
    if len(feat) > length:
        res, st, ed = random_extract(feat, length)
        return res, st, ed
    else:
        return pad(feat, length), 0, length


def process_action_label(label, length, start, end):
    if len(label) > length:
        return label[start:end]
    else:
        if np.shape(label)[0] <= length:
            return np.pad(label, ((0), (length - np.shape(label)[0])), mode='constant', constant_values=0)
        else:
            return label


def process_back_label(label, length, start, end):
    if len(label) > length:
        return label[start:end]
    else:
        if np.shape(label)[0] <= length:
            return np.pad(label, ((0), (length - np.shape(label)[0])), mode='constant', constant_values=-100)
        else:
            return label




def getPseudoVideoInstances(features, click_label, device):
    t_len = features.shape[0]
    score = torch.zeros((t_len, 21))

    action_index = torch.where(click_label > 0)[0]
    for i, index in enumerate(action_index):
        a, b = index, index
        feat = features[index].unsqueeze(0).unsqueeze(0)
        weight = F.cosine_similarity(features.unsqueeze(0), feat, dim=-1).squeeze()
        th = weight[weight > 0].mean()
        weight = torch.where(weight < th, torch.tensor(0).float().to(device), weight)

        while weight[b] != 0:
            b = b + 1
            if b >= features.shape[0]:
                b = features.shape[0] - 1
                break

        while weight[a] != 0:
            a = a - 1
            if a < 0:
                a = 0
                break
        weight[:a], weight[b:] = 0, 0
        score[a:b, click_label[index]] = weight[a:b]

    return score


def getVideoSimilarityInstances(dataset, device):
    done = False

    predictions_v_score = []
    while not done:
        features, click_labels, vname, done = dataset.load_data(is_training=False, is_testing_on_train=True)
        features = torch.from_numpy(features).float().to(device)
        click_labels = torch.from_numpy(click_labels).long().to(device)

        pe_score = getPseudoVideoInstances(features, click_labels, device)
        predictions_v_score.append(pe_score[:, 1:].cpu().numpy())

    return predictions_v_score
