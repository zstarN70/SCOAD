import random
import torch
import numpy as np
import os


def set_seed(seed, rank=0):
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed + rank)
        torch.cuda.manual_seed_all(seed + rank)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    seed = 100
    set_seed(100)

    root_label = './data/ActivityNet1.2/ActivityNet1.2-Annotations/'
    subpath = os.path.join(root_label, 'subset.npy')
    segpath = os.path.join(root_label, 'segments.npy')
    labpath = os.path.join(root_label, 'labels.npy')
    vnamepath = os.path.join(root_label, 'videoname.npy')
    durpath = os.path.join(root_label, 'duration.npy')
    segments = np.load(segpath, allow_pickle=True)
    subset = np.load(subpath, allow_pickle=True)
    labels = np.load(labpath, allow_pickle=True)
    vname = np.load(vnamepath, allow_pickle=True)
    duration = np.load(durpath, allow_pickle=True)

    subset = [i.decode('utf-8') for i in subset]
    vname = [i.decode('utf-8') for i in vname]

    out_path = os.path.join(root_label, f'seed/random_seed_back/anet_back_click_{seed}.txt')
    with open(out_path, 'w') as f:
        f.write('videoname,time,label')
    for i, flag in enumerate(subset):
        if 'validation' in flag:
            continue
        if len(segments[i]) > 1:
            tmp_list = []
            tmp_list.append([0, segments[i][0][0]])
            for j in range(len(segments[i]) - 1):
                tmp_list.append([segments[i][j][1], segments[i][j + 1][1]])
            tmp_list.append([segments[i][j][1], duration[i]])
            n = random.randint(1, len(segments[i]) + 1)
            for x in range(n):
                cat = random.uniform(tmp_list[x][0], tmp_list[x][1])
                pstr = f'{vname[i]},{cat},Background'
                with open(out_path, 'a+') as f:
                    f.write('\n' + pstr)
        elif len(segments[i]) == 1:
            tmp_list = []
            tmp_list.append([0, segments[i][0][0]])
            tmp_list.append([segments[i][0][1], duration[i]])
            n = random.randint(1, 2)
            for x in range(n):
                cat = random.uniform(tmp_list[x][0], tmp_list[x][1])
                pstr = f'{vname[i]},{cat},Background'
                with open(out_path, 'a+') as f:
                    f.write('\n' + pstr)
        else:
            cat = random.uniform(0, duration[i])
            pstr = f'{vname[i]},{cat},Background'
            with open(out_path, 'a+') as f:
                f.write('\n' + pstr)