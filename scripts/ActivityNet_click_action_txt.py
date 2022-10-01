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

    out_path = os.path.join(root_label, f'seed/random_seed_action/anet_action_click_{seed}.txt')
    for i, flag in enumerate(subset):
        if 'validation' in flag:
            continue
        for seg, action in zip(segments[i], labels[i]):
            segtime = int((((seg[1] - seg[0]) / duration[i]) * 10) // 3)
            if segtime == 0:
                cat = random.uniform(seg[1], seg[0])
                pstr = f'{vname[i]},{cat},{action}'
                with open(out_path, 'a+') as f:
                    f.write('\n' + pstr)
            else:
                for num in range(segtime):
                    start = seg[0] + ((seg[1] - seg[0]) / segtime) * (num)
                    end = seg[0] + ((seg[1] - seg[0]) / segtime) * (num + 1)
                    if start < seg[0]:
                        print(1)
                    if end > seg[1]:
                        end = seg[1]
                    cat = random.uniform(start, end)
                    pstr = f'{vname[i]},{cat},{action}'
                    with open(out_path, 'a+') as f:
                        f.write('\n' + pstr)
