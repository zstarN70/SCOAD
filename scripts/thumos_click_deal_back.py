import os
import numpy as np
import pickle


if __name__ == '__main__':

    root_label = './data/Thumos14/Thumos14reduced-Annotations/'
    featpath = './data/Thumos14/feature/Thumos14reduced-I3D-JOINTFeatures.npy'
    txt_path = './data/Thumos14/signal_anno/human/back/THUMOS14_Background-Click-Annotation_A1.txt'

    subpath = os.path.join(root_label, 'subset.npy')
    vnamepath = os.path.join(root_label, 'videoname.npy')
    durpath = os.path.join(root_label, 'duration.npy')

    subset = np.load(subpath, allow_pickle=True)
    vname = np.load(vnamepath, allow_pickle=True)
    duration = np.load(durpath, allow_pickle=True)
    video_feat = np.load(featpath, encoding='bytes', allow_pickle=True)
    subset_index = [i for i, s in enumerate(subset) if s.decode('utf-8') == 'validation']
    vname = [i.decode('utf-8') for i in vname]

    video_feat_num = [int(i.shape[0]) for i in video_feat[subset_index]]

    ret = [[[], []] for i in range(len(subset_index))]
    for lines in open(txt_path, 'r'):
        if lines == '\n' or 'URL' in lines:
            continue
        lines = lines.rstrip('\n')
        lines = lines.rstrip(' ')
        data = lines.split(',')
        index = vname.index(data[0])
        feat_num = int(float(data[4]) / duration[index] * video_feat_num[index])
        ret[index][0].append(feat_num)
        ret[index][1].append('Background')

    target = [[-100 for i in range(x)] for x in video_feat_num]
    for retidnex, i in enumerate(ret):
        for index, j in enumerate(i[0]):
            if target[retidnex][j] == -100:
                target[retidnex][j] = 20
    target = np.asarray([np.asarray(i) for i in target])

    with open(os.path.join(root_label, 'thumos_click_label_back.pkl'), 'wb') as f:
        pickle.dump(target, f)
    with open(os.path.join(root_label, 'thumos_click_txt_back.pkl'), 'wb') as f:
        pickle.dump(ret, f)
