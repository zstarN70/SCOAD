import numpy as np


def filter_wClsLabel(det_res, dataset):
    vnum = len(det_res)

    labels = dataset.labels
    trainidx = dataset.trainidx
    click_label=dataset.back_click_segment

    res = [[] for i in range(vnum)]
    for i in range(vnum):
        lb = labels[trainidx[i]]
        clb = click_label[trainidx[i]]
        for j in range(len(det_res[i])):
            if j == 0:  # video name
                continue
            if det_res[i][j][0] in lb:
                # if not include background
                for t in clb[0]:
                    if not (det_res[i][j][1] >= t and det_res[i][j][2] <= t):
                        res[i].append(det_res[i][j])
                        break
    return res


def tiou(anchors_min, anchors_max, len_anchors, box_min, box_max):
    '''
    calculate jaccatd score between a box and an anchor
    '''
    inter_xmin = np.maximum(anchors_min, box_min)
    inter_xmax = np.minimum(anchors_max, box_max)
    inter_len = np.maximum(inter_xmax - inter_xmin, 0.)
    union_len = len_anchors - inter_len + box_max - box_min
    tiou = np.divide(inter_len, union_len)
    return tiou


def get_train_instance(ret, lb):
    if lb == ret[0]:
        return ret
    return None


def get_click_instance(ret, f_click, lb):
    if lb == ret[0]:
        if (ret[1] < f_click and ret[2] > f_click):
            return ret
    return None


def final_result_process(dst, th=0.3):
    if len(dst) == 1:
        return [dst[0]]
    elif len(dst) == 0:
        return []
    else:
        tstart = [i[1] for i in dst]
        tend = [i[2] for i in dst]
        tmp_width = tend[0] - tstart[0]
        iou = tiou(tstart[0], tend[0], tmp_width, np.array(tstart), np.array(tend))
        ret = [dst[i] for i in range(len(dst)) if (i != 0 and iou[i] >= th)]

        if len(ret) == 0:
            return [dst[0]]

        return ret


def filterIOU(init_ret, train_ret, dataset):
    trainidx = dataset.trainidx
    action_click_label = dataset.action_click_segment

    res = [[] for i in range(len(init_ret))]
    for i in range(len(init_ret)):
        clb = action_click_label[trainidx[i]]
        for j, (f_click, lb) in enumerate(zip(clb[0], clb[1])):
            dst = [get_click_instance(init_ret[i][m], f_click, lb) for m in range(len(init_ret[i]))]
            dst = [i for i in dst if i != None][:1]
            dst += [get_train_instance(train_ret[i][m], lb) for m in range(len(train_ret[i]))]
            dst = [i for i in dst if i != None]
            ret = final_result_process(dst)
            if len(ret) != 0:
                res[i].extend(ret)

    return res