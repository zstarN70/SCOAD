import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_default_tensor_type('torch.cuda.FloatTensor')

class Model(torch.nn.Module):
    def __init__(self, n_feature, n_class):
        super(Model, self).__init__()

        self.fc = nn.Linear(n_feature, n_feature)
        self.classifier = nn.Linear(n_feature, n_class)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, inputs, is_training=True):
        x = F.relu(self.fc(inputs))

        if is_training:
            x = self.dropout(x)

        return x, self.classifier(x)


class SCOAD(nn.Module):
    def __init__(self, args):
        super(SCOAD, self).__init__()

        self.enc_step = args.enc_step
        self.fusion_size = args.feature_size
        self.hidden_size = args.hidden_size
        self.num_classes = args.num_class

        self.GRUCell = nn.GRUCell(self.fusion_size, self.hidden_size)
        self.classifier_frame = nn.Linear(self.hidden_size, self.num_classes + 1)
        self.mask = nn.Linear(self.hidden_size, 2)
        self.model = Model(self.fusion_size, self.num_classes)

    def encoder(self, feat, enc_hx):
        enc_hx = self.GRUCell(feat, enc_hx)

        enc_score = self.classifier_frame(F.relu(enc_hx))
        mask_score = self.mask(F.relu(enc_hx))

        return enc_score, mask_score, enc_hx

    def decode(self, feat, enc_hx):

        features, _ = self.model(feat, is_training=False)
        enc_score, mask_score, enc_hx = \
            self.encoder(features, enc_hx)

        return enc_score, mask_score, enc_hx

    def forward(self, feats, device='cuda', is_training=True):

        feats, vl_preds = self.model(feats, is_training=is_training)
        if is_training:
            batch_size = feats.shape[0]
            tlen = feats.shape[1]
            feat_dim = feats.shape[2]
        else:
            return feats, vl_preds

        grn_batch = tlen // self.enc_step
        feats_grn = torch.zeros((batch_size, grn_batch, self.enc_step, feat_dim)).to(device)
        for i in range(0, grn_batch):
            feats_grn[:, i, :, :] = feats[:, (i * self.enc_step):((i + 1) * self.enc_step), :]
        feats_grn = feats_grn.view((batch_size * grn_batch, self.enc_step, feat_dim)).transpose(0, 1)
        enc_hx = torch.zeros((batch_size * grn_batch, self.hidden_size)).to(device)

        enc_score_stack, mask_score_stack = [], []
        for i in range(self.enc_step):
            enc_score, mask_score, enc_hx = self.encoder(feats_grn[i], enc_hx)
            enc_score_stack.append(enc_score)
            mask_score_stack.append(mask_score)
        enc_scores = torch.stack(enc_score_stack).view(-1, self.num_classes + 1)
        mask_scores = torch.stack(mask_score_stack).view(-1, 2)

        return enc_scores, mask_scores, feats, vl_preds
