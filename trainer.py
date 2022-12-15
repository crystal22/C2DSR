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
        self.noter = noter

        self.len_train_dl = len(self.trainloader)
        self.len_val_dl = len(self.valloader)
        self.len_test_dl = len(self.testloader)

        self.device = args.device
        self.d_latent = args.d_latent
        self.n_item_x = args.n_item_x
        self.n_item_y = args.n_item_y
        self.len_rec = args.len_rec
        self.lambda_loss = args.lambda_loss

        self.noter.log_brief()

    def run_epoch(self):
        self.model.train()
        self.optimizer.zero_grad()
        loss_tr_epoch, loss_rec_epoch, loss_mi_epoch = 0, 0, 0

        # training phase
        for batch in tqdm(self.trainloader, desc='  - training', leave=False):
            self.model.convolve_graph()
            loss_tr_batch, loss_rec_batch, loss_mi_batch = self.train_batch(batch)
            loss_tr_epoch += loss_tr_batch.item() * len(batch)
            loss_rec_epoch += loss_rec_batch.item() * len(batch)
            loss_mi_epoch += loss_mi_batch.item() * len(batch)
        loss_tr_epoch /= self.len_train_dl
        loss_rec_epoch /= self.len_train_dl
        loss_mi_epoch /= self.len_train_dl

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

        return [loss_tr_epoch, loss_rec_epoch, loss_mi_epoch], pred_val_x, pred_val_y, pred_test_x, pred_test_y

    def cal_mask(self, gt_mask):
        mask_x = gt_mask.float().sum(-1).unsqueeze(-1).repeat(1, gt_mask.size(-1))
        mask_x = gt_mask / mask_x  # for mean
        mask_x = mask_x.unsqueeze(-1).repeat(1, 1, self.d_latent)
        return mask_x

    def train_batch(self, batch):
        # representation learning
        seq, seq_x, seq_y, pos, pos_x, pos_y, gt, gt_share_x, gt_share_y, gt_x, gt_y, gt_mask, gt_mask_x, gt_mask_y, \
            crpt_x, crpt_y = map(lambda x: x.to(self.device), batch)

        seq_enc, seq_enc_x, seq_enc_y = self.model(seq, seq_x, seq_y, pos, pos_x, pos_y)

        crpt_enc_x = self.model.forward_negative(crpt_x, pos)
        crpt_enc_y = self.model.forward_negative(crpt_y, pos)

        # contrastive learning
        mask_x = self.cal_mask(gt_mask_x)
        mask_y = self.cal_mask(gt_mask_y)

        hx_enc_pos = (seq_enc_x * mask_x).sum(1)
        hy_enc_pos = (seq_enc_y * mask_y).sum(1)

        hx_share_enc_pos = (seq_enc * mask_x).sum(1)
        hy_share_enc_pos = (seq_enc * mask_y).sum(1)

        hx_share_enc_neg = (crpt_enc_x * mask_x).sum(1)
        hy_share_enc_neg = (crpt_enc_y * mask_y).sum(1)

        sim_x_pos = self.model.D_X(hx_enc_pos, hy_share_enc_pos)  # the cross-domain infomax
        sim_x_neg = self.model.D_X(hx_enc_pos, hy_share_enc_neg)

        sim_y_pos = self.model.D_Y(hy_enc_pos, hx_share_enc_pos)
        sim_y_neg = self.model.D_Y(hy_enc_pos, hx_share_enc_neg)

        # infomax
        pos_label = torch.ones_like(sim_x_pos, device=self.device)
        neg_label = torch.zeros_like(sim_x_neg, device=self.device)

        bce_x_pos = F.binary_cross_entropy_with_logits(sim_x_pos, pos_label)
        bce_x_neg = F.binary_cross_entropy_with_logits(sim_x_neg, neg_label)
        loss_cts_x = bce_x_pos + bce_x_neg

        bce_y_pos = F.binary_cross_entropy_with_logits(sim_y_pos, pos_label)
        bce_y_neg = F.binary_cross_entropy_with_logits(sim_y_neg, neg_label)
        loss_cts_y = bce_y_pos + bce_y_neg

        loss_mi = loss_cts_x + loss_cts_y

        # recommendation
        gt_share_x = gt_share_x[:, -self.len_rec:]
        gt_share_y = gt_share_y[:, -self.len_rec:]
        gt_x = gt_x[:, -self.len_rec:]
        gt_mask_x = gt_mask_x[:, -self.len_rec:]
        gt_y = gt_y[:, -self.len_rec:]
        gt_mask_y = gt_mask_y[:, -self.len_rec:]

        res_share_x = self.model.lin_X(seq_enc[:, -self.len_rec:])
        res_share_y = self.model.lin_Y(seq_enc[:, -self.len_rec:])
        res_share_pad = self.model.lin_PAD(seq_enc[:, -self.len_rec:])
        res_share_x = torch.cat((res_share_x, res_share_pad), dim=-1)
        res_share_y = torch.cat((res_share_y, res_share_pad), dim=-1)

        res_x = self.model.lin_X(seq_enc[:, -self.len_rec:] + seq_enc_x[:, -self.len_rec:])
        res_pad_x = self.model.lin_PAD(seq_enc_x[:, -self.len_rec:])
        res_x = torch.cat((res_x, res_pad_x), dim=-1)

        res_y = self.model.lin_Y(seq_enc[:, -self.len_rec:] + seq_enc_y[:, -self.len_rec:])
        res_pad_y = self.model.lin_PAD(seq_enc_y[:, -self.len_rec:])
        res_y = torch.cat((res_y, res_pad_y), dim=-1)

        loss_share_x = F.cross_entropy(res_share_x.reshape(-1, self.n_item_x + 1), gt_share_x.reshape(-1))
        loss_share_y = F.cross_entropy(res_share_y.reshape(-1, self.n_item_y + 1), gt_share_y.reshape(-1))

        loss_x = F.cross_entropy(res_x.reshape(-1, self.n_item_x + 1), gt_x.reshape(-1))
        loss_y = F.cross_entropy(res_y.reshape(-1, self.n_item_y + 1), gt_y.reshape(-1))

        loss_share_x *= (gt_share_x != self.n_item_x).sum() / self.len_rec
        loss_share_y *= (gt_share_y != self.n_item_y).sum() / self.len_rec
        loss_x = (loss_x * (gt_mask_x.reshape(-1))).mean()
        loss_y = (loss_y * (gt_mask_y.reshape(-1))).mean()

        loss_rec = loss_share_x + loss_share_y + loss_x + loss_y

        loss_batch = self.lambda_loss * loss_rec + (1 - self.lambda_loss) * loss_mi

        loss_batch.backward()
        self.optimizer.step()

        return loss_batch, loss_rec, loss_mi

    def evaluate_batch(self, batch):
        seq, seq_x, seq_y, pos, pos_x, pos_y, X_last, Y_last, XorY, gt, neg_list = \
            map(lambda x: x.to(self.device), batch)
        seq_enc, seq_enc_x, seq_enc_y = self.model(seq, seq_x, seq_y, pos, pos_x, pos_y)

        pred_x, pred_y = [], []
        for idx, feat in enumerate(seq_enc):
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
