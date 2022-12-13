from os.path import join
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F

from dataloader import get_dataloader
from models.C2DSR import C2DSR
from utils.graph import make_graph


class Trainer(object):
    def __init__(self, args, noter):
        print('Loading data..')
        self.trainloader, self.valloader, self.testloader = get_dataloader(args)
        print('Loading graphs..')
        self.adj_share, self.adj_specific = make_graph(args, join(args.path_raw, 'train_new.txt'))
        print('Finish loading data and graphs.')

        self.model = C2DSR(args, self.adj_share, self.adj_specific).to(args.device)
        self.optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, self.model.parameters()), lr=args.lr,
                                          weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

        self.len_train_dl = len(self.trainloader)
        self.len_val_dl = len(self.valloader)
        self.len_test_dl = len(self.testloader)

        self.noter = noter
        self.device = args.device
        self.lambda_loss = args.lambda_loss
        self.n_item_x = args.n_item_x
        self.n_item_y = args.n_item_y

        self.noter.log_brief()

        self.mi_loss = 0

    def run_epoch(self):
        self.model.train()
        self.optimizer.zero_grad()
        self.mi_loss = 0
        loss_tr_epoch = 0

        # training phase
        self.model.convolve_graph()
        self.model.train()
        for batch in tqdm(self.trainloader, desc='  - training', leave=False):
            loss_tr = self.train_batch(batch)
            loss_tr_epoch += loss_tr.item() * len(batch)
        loss_tr_epoch /= self.len_train_dl

        # evaluating phase
        self.model.eval()
        pred_val_x, pred_val_y, pred_test_x, pred_test_y = [], [], [], []
        with torch.no_grad():
            for batch in tqdm(self.valloader, desc='  - validating', leave=False):
                pred_x, pred_y = self.evaluate_batch(batch)
                pred_val_x += pred_x
                pred_val_y += pred_y

            for batch in tqdm(self.testloader, desc='  - testing', leave=False):
                pred_x, pred_y = self.evaluate_batch(batch)
                pred_test_x += pred_x
                pred_test_y += pred_y

        return loss_tr_epoch, pred_val_x, pred_val_y, pred_test_x, pred_test_y

    def cal_mask(self, gt_mask, seq_enc):
        mask_x = gt_mask.float().sum(-1).unsqueeze(-1).repeat(1, gt_mask.size(-1))
        mask_x = gt_mask / mask_x  # for mean
        mask_x = mask_x.unsqueeze(-1).repeat(1, 1, seq_enc.size(-1))
        return mask_x

    def train_batch(self, batch):
        # representation learning
        seq, seq_x, seq_y, pos, pos_x, pos_y, gt, gt_share_x, gt_share_y, gt_x, gt_y, gt_mask, gt_mask_share_x, \
            gt_mask_share_y, gt_mask_x, gt_mask_y, crpt_x, crpt_y = map(lambda x: x.to(self.device), batch)

        seq_enc, seq_enc_x, seq_enc_y = self.model(seq, seq_x, seq_y, pos, pos_x, pos_y)

        crpt_enc_x = self.model.forward_negative(crpt_x, pos)
        crpt_enc_y = self.model.forward_negative(crpt_y, pos)

        #
        mask_x = self.cal_mask(gt_mask_x, seq_enc)
        mask_y = self.cal_mask(gt_mask_y, seq_enc)

        r_enc_x = (seq_enc_x * mask_x).sum(1)
        r_enc_y = (seq_enc_y * mask_y).sum(1)

        real_enc_x = (seq_enc * mask_x).sum(1)
        real_enc_y = (seq_enc * mask_y).sum(1)
        x_false_enc = (crpt_enc_x * mask_x).sum(1)
        y_false_enc = (crpt_enc_y * mask_y).sum(1)

        score_pos_x = self.model.D_X(r_enc_x, real_enc_y)  # the cross-domain infomax
        score_neg_x = self.model.D_X(r_enc_x, y_false_enc)

        score_pos_y = self.model.D_Y(r_enc_y, real_enc_x)
        score_neg_y = self.model.D_Y(r_enc_y, x_false_enc)

        pos_label = torch.ones_like(score_pos_x, device=self.device)
        neg_label = torch.zeros_like(score_neg_x, device=self.device)
        x_mi_real = F.binary_cross_entropy_with_logits(score_pos_x, pos_label)
        x_mi_false = F.binary_cross_entropy_with_logits(score_neg_x, neg_label)
        x_mi_loss = x_mi_real + x_mi_false

        y_mi_real = F.binary_cross_entropy_with_logits(score_pos_y, pos_label)
        y_mi_false = F.binary_cross_entropy_with_logits(score_neg_y, neg_label)
        y_mi_loss = y_mi_real + y_mi_false

        used = 10
        gt = gt[:, -used:]
        gt_mask = gt_mask[:, -used:]
        gt_share_x = gt_share_x[:, -used:]
        gt_mask_share_x = gt_mask_share_x[:, -used:]
        gt_share_y = gt_share_y[:, -used:]
        gt_mask_share_y = gt_mask_share_y[:, -used:]
        gt_x = gt_x[:, -used:]
        gt_mask_x = gt_mask_x[:, -used:]
        gt_y = gt_y[:, -used:]
        gt_mask_y = gt_mask_y[:, -used:]

        share_x_result = self.model.lin_X(seq_enc[:, -used:])  # b * seq * X_num
        share_y_result = self.model.lin_Y(seq_enc[:, -used:])  # b * seq * Y_num
        share_pad_result = self.model.lin_PAD(seq_enc[:, -used:])  # b * seq * 1
        share_trans_x_result = torch.cat((share_x_result, share_pad_result), dim=-1)
        share_trans_y_result = torch.cat((share_y_result, share_pad_result), dim=-1)

        specific_x_result = self.model.lin_X(seq_enc[:, -used:] + seq_enc_x[:, -used:])  # b * seq * X_num
        specific_x_pad_result = self.model.lin_PAD(seq_enc_x[:, -used:])  # b * seq * 1
        specific_x_result = torch.cat((specific_x_result, specific_x_pad_result), dim=-1)

        specific_y_result = self.model.lin_Y(seq_enc[:, -used:] + seq_enc_y[:, -used:])  # b * seq * Y_num
        specific_y_pad_result = self.model.lin_PAD(seq_enc_y[:, -used:])  # b * seq * 1
        specific_y_result = torch.cat((specific_y_result, specific_y_pad_result), dim=-1)

        loss_share_x = F.cross_entropy(share_trans_x_result.reshape(-1, self.n_item_x + 1),
                                       gt_share_x.reshape(-1))  # b * seq
        loss_share_y = F.cross_entropy(share_trans_y_result.reshape(-1, self.n_item_y + 1),
                                       gt_share_y.reshape(-1))  # b * seq

        loss_x = F.cross_entropy(specific_x_result.reshape(-1, self.n_item_x + 1), gt_x.reshape(-1))  # b * seq
        loss_y = F.cross_entropy(specific_y_result.reshape(-1, self.n_item_y + 1), gt_y.reshape(-1))  # b * seq

        loss_share_x = (loss_share_x * (gt_mask_share_x.reshape(-1))).mean()
        loss_share_y = (loss_share_y * (gt_mask_share_y.reshape(-1))).mean()
        loss_x = (loss_x * (gt_mask_x.reshape(-1))).mean()
        loss_y = (loss_y * (gt_mask_y.reshape(-1))).mean()

        loss_batch = self.lambda_loss * (loss_share_x + loss_share_y + loss_x + loss_y) + (1 - self.lambda_loss) * (
                x_mi_loss + y_mi_loss)

        self.mi_loss += (1 - self.lambda_loss) * (x_mi_loss.item() + y_mi_loss.item())

        loss_batch.backward()
        self.optimizer.step()

        return loss_batch

    def evaluate_batch(self, batch):
        seq, seq_x, seq_y, pos, pos_x, pos_y, X_last, Y_last, XorY, gt, neg_list = \
            map(lambda x: x.to(self.device), batch)
        seq_enc, seq_enc_x, seq_enc_y = self.model(seq, seq_x, seq_y, pos, pos_x, pos_y)

        pred_x, pred_y = [], []
        for idx, feat in enumerate(seq_enc):  # b * s * f
            if XorY[idx] == 0:
                share_enc = seq_enc[idx, -1]
                specific_enc = seq_enc_x[idx, X_last[idx]]
                X_score = self.model.lin_X(share_enc + specific_enc).squeeze(0)
                cur = X_score[gt[idx]]
                score_larger = (X_score[neg_list[idx]] > (cur + 0.00001)).data.cpu().numpy()
                true_item_rank = np.sum(score_larger) + 1
                pred_x.append(true_item_rank)

            else:
                share_enc = seq_enc[idx, -1]
                specific_enc = seq_enc_y[idx, Y_last[idx]]
                score_y = self.model.lin_Y(share_enc + specific_enc).squeeze(0)
                cur = score_y[gt[idx]]
                score_larger = (score_y[neg_list[idx]] > (cur + 0.00001)).data.cpu().numpy()
                true_item_rank = np.sum(score_larger) + 1
                pred_y.append(true_item_rank)

        return pred_x, pred_y
