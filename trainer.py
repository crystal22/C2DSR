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
        print('[info]\n\tLoading data..')
        self.trainloader, self.valloader, self.testloader = get_dataloader(args)
        print('\tLoading graphs..')
        self.adj_share, self.adj_specific = make_graph(args, join(args.path_raw, 'train_new.txt'))
        print('\tFinished loading data and graphs.')

        self.model = C2DSR(args, self.adj_share, self.adj_specific).to(args.device)
        self.optimizer = torch.optim.AdamW(filter(lambda x: x.requires_grad, self.model.parameters()), lr=args.lr,
                                           weight_decay=args.l2, amsgrad=True)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
        self.noter = noter

        self.n_tr = len(self.trainloader.dataset)
        self.n_val = len(self.valloader.dataset)
        self.n_te = len(self.testloader.dataset)

        self.device = args.device
        self.d_latent = args.d_latent
        self.n_item_a = args.n_item_a
        self.n_item_b = args.n_item_b
        self.len_rec = args.len_rec
        self.lambda_loss = args.lambda_loss

        self.label_pos = torch.ones(args.batch_size, 1, device=self.device)
        self.label_neg = torch.zeros(args.batch_size, 1, device=self.device)

    def run_epoch(self):
        self.model.train()
        self.optimizer.zero_grad()
        loss_tr_epoch, loss_rec_epoch, loss_mi_epoch = 0., 0., 0.
        t_start = time.time()

        # training phase
        for batch in tqdm(self.trainloader, desc='  - training', leave=False):
            self.model.convolve_graph()
            loss_tr_batch, loss_rec_batch, loss_mi_batch = self.train_batch(batch)

            loss_tr_epoch += (loss_tr_batch.item() * batch[0].shape[0])
            loss_rec_epoch += (loss_rec_batch.item() * batch[0].shape[0])
            loss_mi_epoch += (loss_mi_batch.item() * batch[0].shape[0])

        loss_tr_epoch /= self.n_tr
        loss_rec_epoch /= self.n_tr
        loss_mi_epoch /= self.n_tr

        self.noter.log_train(loss_tr_epoch, loss_rec_epoch, loss_mi_epoch, time.time() - t_start)

        # evaluating phase
        self.model.eval()
        pred_val_a, pred_val_b = [], []
        with torch.no_grad():
            self.model.convolve_graph()
            for batch in tqdm(self.valloader, desc='  - validating', leave=False):
                pred_a, pred_b = self.evaluate_batch(batch)
                pred_val_a += pred_a
                pred_val_b += pred_b

        return pred_val_a, pred_val_b

    def run_test(self):
        self.model.eval()
        pred_test_a, pred_test_b = [], []

        with torch.no_grad():
            for batch in tqdm(self.testloader, desc='  - testing', leave=False):
                pred_a, pred_b = self.evaluate_batch(batch)
                pred_test_a += pred_a
                pred_test_b += pred_b

        return pred_test_a, pred_test_b

    def cal_mask(self, gt_mask):
        mask_a = gt_mask.float().sum(-1).unsqueeze(-1).repeat(1, gt_mask.size(-1))
        mask_a = gt_mask / mask_a  # for mean
        mask_a = mask_a.unsqueeze(-1).repeat(1, 1, self.d_latent)
        return mask_a

    def train_batch(self, batch):
        seq_share, seq_share_a, seq_share_b, pos, pos_a, pos_b, gt_share_a, gt_share_b, gt_a, gt_b, gt_mask_a, gt_mask_b, \
            seq_share_neg_a, seq_share_neg_b = map(lambda x: x.to(self.device), batch)
        n_batch = seq_share.shape[0]

        # contrastive learning, h_share_neg_a & h_share_neg_b are partially contrastive to h_share_pos
        h_share_pos, hx_pos, hy_pos = self.model(seq_share, seq_share_a, seq_share_b, pos, pos_a, pos_b)
        mask_a = self.cal_mask(gt_mask_a)
        mask_b = self.cal_mask(gt_mask_b)

        hx_mean_pos = (hx_pos * mask_a).sum(1)
        hy_mean_pos = (hy_pos * mask_b).sum(1)

        sim_a_pos = self.model.D_a(hx_mean_pos, (h_share_pos * mask_b).sum(1))
        sim_a_neg = self.model.D_a(hx_mean_pos, (self.model.forward_share(seq_share_neg_a, pos) * mask_a).sum(1))

        sim_b_pos = self.model.D_b(hy_mean_pos, (h_share_pos * mask_a).sum(1))
        sim_b_neg = self.model.D_b(hy_mean_pos, (self.model.forward_share(seq_share_neg_b, pos) * mask_b).sum(1))

        label_pos = self.label_pos[:n_batch, :]
        label_neg = self.label_neg[:n_batch, :]

        loss_mi_a_pos = F.binary_cross_entropy_with_logits(sim_a_pos, label_pos)
        loss_mi_a_neg = F.binary_cross_entropy_with_logits(sim_a_neg, label_neg)

        loss_mi_b_pos = F.binary_cross_entropy_with_logits(sim_b_pos, label_pos)
        loss_mi_b_neg = F.binary_cross_entropy_with_logits(sim_b_neg, label_neg)

        loss_mi = loss_mi_a_pos + loss_mi_a_neg + loss_mi_b_pos + loss_mi_b_neg

        # recommendation
        h_share_rec = h_share_pos[:, -self.len_rec:, :]
        h_a_rec = hx_pos[:, -self.len_rec:]
        h_b_rec = hy_pos[:, -self.len_rec:]

        gt_share_a = gt_share_a[:, -self.len_rec:]
        gt_share_b = gt_share_b[:, -self.len_rec:]
        gt_a = gt_a[:, -self.len_rec:]
        gt_b = gt_b[:, -self.len_rec:]

        scores_share_a = torch.cat((self.model.classifier_a(h_share_rec),
                                    self.model.classifier_pad(h_share_rec)), dim=-1)
        scores_share_b = torch.cat((self.model.classifier_b(h_share_rec),
                                    self.model.classifier_pad(h_share_rec)), dim=-1)

        scores_a = torch.cat((self.model.classifier_a(h_share_rec + h_a_rec),
                              self.model.classifier_pad(h_a_rec)), dim=-1)

        scores_b = torch.cat((self.model.classifier_b(h_share_rec + h_b_rec),
                              self.model.classifier_pad(h_b_rec)), dim=-1)

        # compute loss
        loss_share_a = F.cross_entropy(scores_share_a.reshape(-1, self.n_item_a + 1), gt_share_a.reshape(-1),
                                       ignore_index=self.n_item_a)
        loss_share_b = F.cross_entropy(scores_share_b.reshape(-1, self.n_item_b + 1), gt_share_b.reshape(-1),
                                       ignore_index=self.n_item_b)
        loss_share = \
            loss_share_a * (gt_share_a != self.n_item_a).sum() / (self.len_rec * n_batch) + \
            loss_share_b * (gt_share_b != self.n_item_b).sum() / (self.len_rec * n_batch)

        loss_a = F.cross_entropy(scores_a.reshape(-1, self.n_item_a + 1), gt_a.reshape(-1), ignore_index=self.n_item_a)
        loss_b = F.cross_entropy(scores_b.reshape(-1, self.n_item_b + 1), gt_b.reshape(-1), ignore_index=self.n_item_b)

        loss_rec = loss_share + loss_a + loss_b

        loss_batch = self.lambda_loss * loss_rec + (1 - self.lambda_loss) * loss_mi
        loss_batch.backward()
        self.optimizer.step()

        return loss_batch, loss_rec, loss_mi

    def evaluate_batch(self, batch):
        seq_share, seq_share_a, seq_share_b, pos, pos_a, pos_b, idx_last_a, idx_last_b, xory_last, gt_last, list_neg = \
            map(lambda x: x.to(self.device), batch)
        h_share, hx, hy = self.model(seq_share, seq_share_a, seq_share_b, pos, pos_a, pos_b)

        rank_a, rank_b = [], []
        for i, feat in enumerate(h_share):
            h_share_last = h_share[i, -1]

            if xory_last[i] == 0:  # gt_last belongs to domain a
                hx_last = hx[i, idx_last_a[i]]
                scores_a = self.model.classifier_a(h_share_last + hx_last).squeeze(0)
                rank_a.append((scores_a[list_neg[i]] > scores_a[gt_last[i]]).sum().item() + 1)

            else:  # gt_last belongs to domain b
                hy_last = hy[i, idx_last_b[i]]
                scores_b = self.model.classifier_b(h_share_last + hy_last).squeeze(0)
                rank_b.append((scores_b[list_neg[i]] > scores_b[gt_last[i]]).sum().item() + 1)

        return rank_a, rank_b
