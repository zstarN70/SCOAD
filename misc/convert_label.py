import math
import numpy as np

def convert_gt_labels(args,dataset):
    gt_seg = dataset.segments
    gt_seg_label = dataset.seg_labels
    classlist = [dataset.classlist.tolist()[i].decode('utf-8') for i in range(len(dataset.classlist))]

    durations = dataset.video_duration
    fps =  args.fps
    num_class = dataset.num_class
    video_name = dataset.video_name
    trainidx = dataset.trainidx

    target = dict()
    targetAS = dict()
    for i in range(len(trainidx)):
        working_idx = trainidx[i]
        vname = video_name[working_idx].decode('utf-8')
        vlen = math.ceil(durations[working_idx] * fps)
        target[vname] = np.zeros((vlen, num_class + 1))
        targetAS[vname] = np.zeros((vlen, 1))
        target[vname][:, 0] = 1
        targetAS[vname][:, 0] = 0
        for j in range(len(gt_seg[working_idx])):
            st = gt_seg[working_idx][j][0]
            ed = gt_seg[working_idx][j][1]
            clsname = gt_seg_label[working_idx][j]
            cls = classlist.index(clsname) + 1
            st = int(st * fps)
            ed = int(ed * fps)
            if st >= vlen:  ## some annotations is out of video length
                continue
            targetAS[vname][st] = 1
            for ind in range(st, min(ed + 1, vlen)):
                target[vname][ind, 0] = 0
                target[vname][ind, cls] = 1
    return target, targetAS


def convert_labels_target(args, det_res,train_res, dataset, th=0.0):
    classlist = [dataset.classlist.tolist()[i].decode('utf-8') for i in range(len(dataset.classlist))]
    durations = dataset.video_duration
    fps = args.fps
    num_class = dataset.num_class
    video_name = dataset.video_name
    trainidx = dataset.trainidx

    target = dict()
    targetAS = dict()
    for i in range(len(det_res)):
        working_idx = trainidx[i]
        vname = video_name[working_idx].decode('utf-8')
        vlen = math.ceil(durations[working_idx] * fps)
        target[vname] = np.zeros((vlen, num_class + 1))
        target[vname][:, 0] = 1
        targetAS[vname] = np.zeros((vlen, 1))
        targetAS[vname][:, 0] = 0
        for j in range(len(det_res[i])):
            sc = det_res[i][j][-1]
            if sc < th:
                continue
            cls = classlist.index(det_res[i][j][0]) + 1
            st = det_res[i][j][1]
            ed = det_res[i][j][2]
            for ind in range(st, min(ed + 1, vlen)):
                target[vname][ind, 0] = 0
                target[vname][ind, cls] = 1

    for i in range(len(train_res)):
        working_idx = trainidx[i]
        vname = video_name[working_idx].decode('utf-8')
        for j in range(len(train_res[i])):
            sc = train_res[i][j][-1]
            if sc < th:
                continue
            st = train_res[i][j][1]
            targetAS[vname][st] = 1

    return target, targetAS



def mix_annotations(targets_noisy, targetsAS_noisy, targets_gt, targetsAS_gt, sup_inds):
    targets = dict()
    targetsAS = dict()
    vnames = []
    for i, vname in enumerate(targets_noisy):
        vnames.append(vname)
        if i in sup_inds:
            targets[vname] = targets_gt[vname]
            targetsAS[vname] = targetsAS_gt[vname]
        else:
            targets[vname] = targets_noisy[vname]
            targetsAS[vname] = targetsAS_noisy[vname]
    return targets, targetsAS


def create_test_target(dataset, idx, vlen, fps):
    classlist = [dataset.classlist.tolist()[i].decode('utf-8') for i in range(len(dataset.classlist))]
    gt_seg = dataset.segments
    gt_seg_label = dataset.seg_labels
    classnum = dataset.num_class + 1

    target = np.zeros((vlen, classnum))
    for i, seg in enumerate(gt_seg[idx]):
        st = int(fps * seg[0])
        ed = int(fps * seg[1])
        cls = classlist.index(gt_seg_label[idx][i]) + 1
        target[st:ed + 1, cls] = 1
    bg_inds = np.where(np.sum(target, axis=-1) == 0)[0]
    target[bg_inds, 0] = 1
    return target










