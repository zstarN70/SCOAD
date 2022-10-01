import torch
import torch.optim as optim
from model.network import SCOAD
from Dataset.ThumosDataset import Thumos14Dataset
from core.interface import test_full
from core.train import train, test_train
from misc import config
from misc import utils
from misc.getDetectionMAP import getDetectionMAP
from misc.filter import filter_wClsLabel, filterIOU
from misc.convert_label import convert_gt_labels, convert_labels_target, mix_annotations


def main(args):
    config.set_seed(args.seed)
    device = torch.device('cuda:' + str(args.cuda_id))
    model = SCOAD(args).to(device)

    dataset = Thumos14Dataset(args)
    predictions_v_score = utils.getVideoSimilarityInstances(dataset, device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    perm = torch.randperm(len(dataset.trainidx))
    sup_num = int(len(perm) * args.sup_percent)
    sup_inds = perm[0:sup_num]

    init_res, _ = getDetectionMAP(predictions_v_score, dataset, th=0.5, eval_set='validation', get_det_res=True)

    targets, targetsAS = None, None
    targets_gt, targetsAS_gt = convert_gt_labels(args, dataset)

    for itr in range(args.max_iter + 1):
        train(itr, dataset, args, model, optimizer, device, targets, targetsAS)
        if itr % args.eval_intern == 0 and itr > args.warm_start:
            det_res, ap = test_train(dataset, model, device)
            print("detection ap on train set " + str(ap))
            if len(det_res) == 0:
                targets, targetsAS = None, None
                continue

            train_res = filter_wClsLabel(det_res, dataset)
            next_res = filterIOU(init_res, train_res, dataset)
            if len(train_res) == 0:
                targets, targetsAS = None, None
                continue

            targets_noisy, targetsAS_noisy = convert_labels_target(args, next_res, train_res, dataset)
            if args.supervision == 'click':
                targets, targetsAS = targets_noisy, targetsAS_noisy
            elif args.supervision == 'mixframe':
                # use mixed of p-labels and gt labels (semi-supervised)
                targets, targetsAS = mix_annotations(targets_noisy, targetsAS_noisy, targets_gt, targetsAS_gt, sup_inds)
            else:
                print("only video and segment are supported as a supervision")
                raise NotImplementedError

        if itr % args.eval_intern == 0 and itr > 2200:
            test_full(dataset, model, device, args.fps)


if __name__ == '__main__':
    args = config.parse_args()
    main(args)