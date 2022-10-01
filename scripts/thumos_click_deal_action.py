import numpy as np
import pickle
import os

if __name__ == '__main__':
    root_label = './data/Thumos14/Thumos14reduced-Annotations/'
    featpath = './data/Thumos14/feature/Thumos14reduced-I3D-JOINTFeatures.npy'
    txt_path = './data/Thumos14/signal_anno/human/action/THUMOS1.txt'


    subpath = os.path.join(root_label, 'subset.npy')
    segpath = os.path.join(root_label, 'segments.npy')
    labpath = os.path.join(root_label, 'labels.npy')
    vnamepath = os.path.join(root_label, 'videoname.npy')
    durpath = os.path.join(root_label, 'duration.npy')
    classpath = os.path.join(root_label, 'classlist.npy')

    segments = np.load(segpath, allow_pickle=True)
    subset = np.load(subpath, allow_pickle=True)
    labels = np.load(labpath, allow_pickle=True)
    vname = np.load(vnamepath, allow_pickle=True)
    duration = np.load(durpath, allow_pickle=True)
    classlist = np.load(classpath, allow_pickle=True)
    video_feat = np.load(featpath, encoding='bytes', allow_pickle=True)
    subset_index = [i for i, s in enumerate(subset) if s.decode('utf-8') == 'validation']
    vname = [i.decode('utf-8') for i in vname]
    classlist = [i.decode('utf-8') for i in classlist]

    video_feat_num = [int(i.shape[0]) for i in video_feat[subset_index]]

    ret = [[[], []] for i in range(len(subset_index))]
    for lines in open(txt_path, 'r'):
        if lines == '\n' or 'label' in lines:
            continue
        lines = lines.rstrip('\n')
        lines = lines.rstrip(' ')
        data = lines.split(',')
        index = vname.index(data[0])
        click_num = int(float(data[1]) / duration[index] * video_feat_num[index])
        if click_num >= video_feat_num[index]:
            continue
        ret[index][0].append(click_num)
        ret[index][1].append(data[2])

    target = [[0 for i in range(x)] for x in video_feat_num]
    for retidnex, i in enumerate(ret):
        for index, j in enumerate(i[0]):
            if target[retidnex][j] == 0:
                target[retidnex][j] = classlist.index(i[1][index]) + 1
    target = np.asarray([np.asarray(i) for i in target])

    with open(os.path.join(root_label, 'thumos_click_label_action.pkl'), 'wb') as f:
        pickle.dump(target, f)
    with open(os.path.join(root_label, 'thumos_click_txt_action.pkl'), 'wb') as f:
        pickle.dump(ret, f)
