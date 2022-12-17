import time
from os.path import join
from tqdm import tqdm
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
        print('Finished loading data and graphs.')

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
        t_start = time.time()

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

        self.scheduler.step()
        self.noter.log_train(loss_tr_epoch, loss_rec_epoch, loss_mi_epoch, time.time() - t_start)

        # evaluating phase
        self.model.eval()
        pred_val_x, pred_val_y = [], []
        with torch.no_grad():
            self.model.convolve_graph()
            for batch in tqdm(self.valloader, desc='  - validating', leave=False):
                pred_x, pred_y = self.evaluate_batch(batch)
                pred_val_x += pred_x
                pred_val_y += pred_y

        return [loss_tr_epoch, loss_rec_epoch, loss_mi_epoch], pred_val_x, pred_val_y

    def run_test(self):
        self.model.eval()
        pred_test_x, pred_test_y = [], []

        with torch.no_grad():
            for batch in tqdm(self.testloader, desc='  - testing', leave=False):
                pred_x, pred_y = self.evaluate_batch(batch)
                pred_test_x += pred_x
                pred_test_y += pred_y

        return pred_test_x, pred_test_y

    def cal_mask(self, gt_mask):
        mask_x = gt_mask.float().sum(-1).unsqueeze(-1).repeat(1, gt_mask.size(-1))
        mask_x = gt_mask / mask_x  # for mean
        mask_x = mask_x.unsqueeze(-1).repeat(1, 1, self.d_latent)
        return mask_x

    def train_batch(self, batch):
        seq_share, seq_share_x, seq_share_y, pos, pos_x, pos_y, gt_share_x, gt_share_y, gt_x, gt_y, \
            gt_mask_x, gt_mask_y, seq_share_neg_x, seq_share_neg_y = map(lambda x: x.to(self.device), batch)

        # representation learning
        h_share_pos, hx_pos, hy_pos = self.model(seq_share, seq_share_x, seq_share_y, pos, pos_x, pos_y)

        h_share_neg_x = self.model.forward_share(seq_share_neg_x, pos)
        h_share_neg_y = self.model.forward_share(seq_share_neg_y, pos)

        # contrastive learning
        # h_share_neg_x & h_share_neg_y are partially contrastive to hh_share_pos
        mask_x = self.cal_mask(gt_mask_x)
        mask_y = self.cal_mask(gt_mask_y)

        hx_mean_pos = (hx_pos * mask_x).sum(1)
        hy_mean_pos = (hy_pos * mask_y).sum(1)

        hx_share_mean_pos = (h_share_pos * mask_x).sum(1)
        hy_share_mean_pos = (h_share_pos * mask_y).sum(1)

        h_share_mean_neg_x = (h_share_neg_x * mask_x).sum(1)
        h_share_mean_neg_y = (h_share_neg_y * mask_y).sum(1)

        sim_x_pos = self.model.D_x(hx_mean_pos, hy_share_mean_pos)
        sim_x_neg = self.model.D_x(hx_mean_pos, h_share_mean_neg_x)

        sim_y_pos = self.model.D_y(hy_mean_pos, hx_share_mean_pos)
        sim_y_neg = self.model.D_y(hy_mean_pos, h_share_mean_neg_y)

        label_pos = torch.ones(len(seq_share), 1, device=self.device)
        label_neg = torch.zeros(len(seq_share), 1, device=self.device)

        loss_mi_x_pos = F.binary_cross_entropy_with_logits(sim_x_pos, label_pos)
        loss_mi_x_neg = F.binary_cross_entropy_with_logits(sim_x_neg, label_neg)

        loss_mi_y_pos = F.binary_cross_entropy_with_logits(sim_y_pos, label_pos)
        loss_mi_y_neg = F.binary_cross_entropy_with_logits(sim_y_neg, label_neg)

        loss_mi = loss_mi_x_pos + loss_mi_x_neg + loss_mi_y_pos + loss_mi_y_neg
        len_mi = (gt_mask_y + gt_mask_x).sum()

        # recommendation
        h_share_rec = h_share_pos[:, -self.len_rec:, :]
        h_x_rec = hx_pos[:, -self.len_rec:]
        h_y_rec = hy_pos[:, -self.len_rec:]

        gt_share_x = gt_share_x[:, -self.len_rec:]
        gt_share_y = gt_share_y[:, -self.len_rec:]
        gt_mask_x = gt_mask_x[:, -self.len_rec:]
        gt_mask_y = gt_mask_y[:, -self.len_rec:]
        gt_x = gt_x[:, -self.len_rec:]
        gt_y = gt_y[:, -self.len_rec:]

        scores_share_x = self.model.classifier_x(h_share_rec)
        scores_share_y = self.model.classifier_y(h_share_rec)
        scores_share_pad = self.model.classifier_pad(h_share_rec)

        scores_share_x = torch.cat((scores_share_x, scores_share_pad), dim=-1)
        scores_share_y = torch.cat((scores_share_y, scores_share_pad), dim=-1)

        scores_x = self.model.classifier_x(h_share_rec + h_x_rec)
        scores_x_pad = self.model.classifier_pad(h_x_rec)
        scores_x = torch.cat((scores_x, scores_x_pad), dim=-1)

        scores_y = self.model.classifier_y(h_share_rec + h_y_rec)
        scores_y_pad = self.model.classifier_pad(h_y_rec)
        scores_y = torch.cat((scores_y, scores_y_pad), dim=-1)

        loss_share_x = F.cross_entropy(scores_share_x.reshape(-1, self.n_item_x + 1), gt_share_x.reshape(-1))
        loss_share_y = F.cross_entropy(scores_share_y.reshape(-1, self.n_item_y + 1), gt_share_y.reshape(-1))

        loss_x = F.cross_entropy(scores_x.reshape(-1, self.n_item_x + 1), gt_x.reshape(-1))
        loss_y = F.cross_entropy(scores_y.reshape(-1, self.n_item_y + 1), gt_y.reshape(-1))

        loss_share_x *= (gt_share_x != self.n_item_x).sum() / self.len_rec
        loss_share_y *= (gt_share_y != self.n_item_y).sum() / self.len_rec
        loss_x = (loss_x * (gt_mask_x.reshape(-1))).mean()  # why do that
        loss_y = (loss_y * (gt_mask_y.reshape(-1))).mean()

        loss_rec = loss_share_x + loss_share_y + loss_x + loss_y
        len_rec = (gt_mask_y + gt_mask_x).sum()

        loss_batch = self.lambda_loss * loss_rec + (1 - self.lambda_loss) * loss_mi
        loss_batch.backward()
        self.optimizer.step()

        loss_rec /= len_rec
        loss_mi /= len_mi
        return self.lambda_loss * loss_rec + (1 - self.lambda_loss) * loss_mi, loss_rec, loss_mi

    def evaluate_batch(self, batch):
        seq_share, seq_share_x, seq_share_y, pos, pos_x, pos_y, idx_last_x, idx_last_y, xory_last, gt_last, list_neg = \
            map(lambda x: x.to(self.device), batch)
        h_share, hx, hy = self.model(seq_share, seq_share_x, seq_share_y, pos, pos_x, pos_y)

        rank_x, rank_y = [], []
        for i, feat in enumerate(h_share):
            h_share_last = h_share[i, -1]

            if xory_last[i] == 0:  # gt_last belongs to domain x
                hx_last = hx[i, idx_last_x[i]]
                scores_x = self.model.classifier_x(h_share_last + hx_last).squeeze(0)
                rank_x.append((scores_x[list_neg[i]] > scores_x[gt_last[i]]).sum().item() + 1)

            else:  # gt_last belongs to domain y
                hy_last = hy[i, idx_last_y[i]]
                scores_y = self.model.classifier_y(h_share_last + hy_last).squeeze(0)
                rank_y.append((scores_y[list_neg[i]] > scores_y[gt_last[i]]).sum().item() + 1)

        return rank_x, rank_y
