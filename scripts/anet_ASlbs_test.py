import pickle
import numpy as np
import os

if __name__ == '__main__':

    root_label = './data/ActivityNet1.2/ActivityNet1.2-Annotations/'
    subpath = os.path.join(root_label, 'subset.npy')
    segpath = os.path.join(root_label, 'segments.npy')
    labpath = os.path.join(root_label, 'labels.npy')
    vnamepath = os.path.join(root_label, 'videoname.npy')
    classpath = os.path.join(root_label, 'classlist.npy')

    segments = np.load(segpath, allow_pickle=True)
    subset = np.load(subpath, allow_pickle=True)
    labels = np.load(labpath, allow_pickle=True)
    vname = np.load(vnamepath, allow_pickle=True)
    classlist = np.load(classpath, allow_pickle=True)

    classlist = [i.decode('utf-8') for i in classlist]
    subset = [i.decode('utf-8') for i in subset]
    vname = [i.decode('utf-8') for i in vname]

    out_ASLbs = {i: [[] for _ in range(len(classlist) + 1)] for i, s in zip(vname, subset) if 'val' in s}
    for i, flag in enumerate(subset):
        if 'training' in flag:
            continue

        for seg, c in zip(segments[i], labels[i]):
            out_ASLbs[vname[i]][classlist.index(c) + 1].append(seg[0])

    with open(os.path.join(root_label, 'ANet12_ASlbs_test.pkl'), 'wb') as f:
        pickle.dump(out_ASLbs, f)
