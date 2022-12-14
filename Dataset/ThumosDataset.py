import numpy as np
from misc import utils
import os.path as osp


class Thumos14Dataset():
    def __init__(self, args):
        self.dataset_name = "Thumos14reduced"
        self.num_class = args.num_class
        self.feature_size = args.feature_size
        self.batch_size = args.batch_size
        self.t_max = args.max_seqlen
        self.path_to_features = osp.join(args.data_root, 'feature/{}-I3D-JOINTFeatures.npy'.format(self.dataset_name))
        self.path_to_annotations = osp.join(args.data_root, '{}-Annotations/'.format(self.dataset_name))
        self.features = np.load(self.path_to_features, encoding='bytes', allow_pickle=True)
        self.segments = np.load(self.path_to_annotations + 'segments.npy', allow_pickle=True)
        self.labels = np.load(self.path_to_annotations + 'labels_all.npy', allow_pickle=True)
        self.classlist = np.load(self.path_to_annotations + 'classlist.npy', allow_pickle=True)
        self.subset = np.load(self.path_to_annotations + 'subset.npy', allow_pickle=True)
        self.video_name = np.load(self.path_to_annotations + 'videoname.npy', allow_pickle=True)
        self.video_duration = np.load(self.path_to_annotations + 'duration.npy', allow_pickle=True)
        self.seg_labels = np.load(self.path_to_annotations + 'labels.npy', allow_pickle=True)

        self.action_click_label = np.load(self.path_to_annotations + 'thumos_click_label_action.pkl', allow_pickle=True)
        self.back_click_label = np.load(self.path_to_annotations + 'thumos_click_label_back.pkl', allow_pickle=True)
        self.back_click_segment = np.load(self.path_to_annotations + 'thumos_click_txt_back.pkl', allow_pickle=True)
        self.action_click_segment = np.load(self.path_to_annotations + 'thumos_click_txt_action.pkl', allow_pickle=True)

        self.trainidx = []
        self.testidx = []
        self.classwiseidx = []
        self.currenttestidx = 0
        self.currenttrainidx = 0
        self.labels_multihot = [utils.strlist2multihot(labs, self.classlist) for labs in self.labels]

        self.train_test_idx()
        self.classwise_feature_mapping()

    def train_test_idx(self):
        train_set = 'validation'
        for i, s in enumerate(self.subset):
            if s.decode('utf-8') == train_set:
                self.trainidx.append(i)
            else:
                self.testidx.append(i)

    def classwise_feature_mapping(self):
        for category in self.classlist:
            idx = []
            for i in self.trainidx:
                for label in self.labels[i]:
                    if label == category.decode('utf-8'):
                        idx.append(i);
                        break;
            self.classwiseidx.append(idx)

    def load_data(self, n_similar=3, is_training=True, is_testing_on_train=False):
        if is_training == True:
            idx = []

            # Load similar pairs
            rand_classid = np.random.choice(len(self.classwiseidx), size=n_similar)
            for rid in rand_classid:
                rand_sampleid = np.random.choice(len(self.classwiseidx[rid]), size=2)
                idx.append(self.classwiseidx[rid][rand_sampleid[0]])
                idx.append(self.classwiseidx[rid][rand_sampleid[1]])

            # Load rest pairs
            rand_sampleid = np.random.choice(len(self.trainidx), size=self.batch_size - 2 * n_similar)
            for r in rand_sampleid:
                idx.append(self.trainidx[r])
            feats, sts, eds, action_click, back_click = [], [], [], [], []
            for i in idx:
                f, s, e = utils.process_feat(self.features[i], self.t_max)
                a_cl = utils.process_action_label(self.action_click_label[i], self.t_max, s, e)
                b_cl = utils.process_back_label(self.back_click_label[i], self.t_max, s, e)
                feats.append(f)
                action_click.append(a_cl)
                back_click.append(b_cl)
                sts.append(s)
                eds.append(e)

            return np.array(feats), np.array(action_click), np.array(back_click), np.array(sts), np.array(eds), \
                   np.array([self.labels_multihot[i] for i in idx]), [self.video_name[i].decode('utf-8') for i in idx]
        else:
            if is_testing_on_train:
                alabs = self.action_click_label[self.trainidx[self.currenttrainidx]]

                feat = self.features[self.trainidx[self.currenttrainidx]]
                vname = self.video_name[self.trainidx[self.currenttrainidx]].decode('utf-8')

                if self.currenttrainidx == len(self.trainidx) - 1:
                    done = True;
                    self.currenttrainidx = 0
                else:
                    done = False;
                    self.currenttrainidx += 1

                return np.array(feat), np.array(alabs), vname, done

            else:
                labs = self.labels_multihot[self.testidx[self.currenttestidx]]
                feat = self.features[self.testidx[self.currenttestidx]]

                if self.currenttestidx == len(self.testidx) - 1:
                    done = True;
                    self.currenttestidx = 0
                else:
                    done = False;
                    self.currenttestidx += 1

                return np.array(feat), np.array(labs), done
