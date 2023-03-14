import os
from os.path import join
import time
import numpy as np


class Noter(object):
    """ console printing and saving into files """
    def __init__(self, args):
        self.args = args

        self.cuda = args.cuda
        self.dataset = args.dataset
        self.n_gnn = args.n_gnn
        self.dropout_gnn = args.dropout_gnn
        self.n_attn = args.n_attn
        self.n_head = args.n_head
        self.dropout_attn = args.dropout_attn
        self.lr = args.lr
        self.l2 = args.l2
        self.benchmark = args.benchmark

        self.f_log = join(args.path_log, args.data + time.strftime('-%m-%d-%H:%M-', time.localtime()) + str(args.n_gnn)
                          + '-' + str(args.n_attn) + '-' + str(args.n_head) + '-' + str(args.lr) + '-' + str(args.l2) +
                          '.txt')

        if os.path.exists(self.f_log):
            os.remove(self.f_log)  # remove the existing file if duplicate

        # welcome
        self.log_welcome()

    # write into log file
    def write(self, msg):
        with open(self.f_log, 'a') as out:
            print(msg, file=out)

    # calculate average improvement on all metrics
    def cal_improve(self, res):
        rate = np.zeros(4)  # hr5_a, ndcg5_a, hr5_b, ndcg5_b
        for i, (x, y) in enumerate(zip([res[1], res[5], res[7], res[11]], self.benchmark)):
            rate[i] = x / y - 1
        return np.mean(rate)

    # log any message
    def log_msg(self, msg):
        print(msg)
        self.write(msg)

    # print and save experiment briefs
    def log_settings(self):
        msg = (f'\n[Info] Experiment (dataset:{self.dataset}, cuda:{self.cuda}) '
               f'\n\t| lr {self.lr:.2e} | l2 {self.l2:.2e} |'
               f'\n\t| n_gnn  {self.n_gnn} | dropout {self.dropout_gnn} |'
               f'\n\t| n_attn {self.n_attn} | dropout {self.dropout_attn} | n_head {self.n_head} |\n')
        self.log_msg(msg)

    # print and save experiment briefs
    def log_welcome(self):
        self.log_msg('\n' + '-' * 20 + ' Experiment: C2DSR (CIKM\'22)' + '-' * 20)
        self.log_settings()

    # save args into log file
    def save_args(self):
        info = '-' * 10 + ' Experiment settings ' + '-' * 10 + '\n'
        for k, v in vars(self.args).items():
            info += '\n\t{} : {}'.format(k, str(v))
        self.write(info + '\n')

    # print and save train phase result
    def log_train(self, loss_tr, loss_rec, loss_mi, t_gap):
        msg = f'\t| train |\n\t| loss {loss_tr:.4f} | rec {loss_rec:.4f} | mi {loss_mi:.4f} | time {t_gap:.0f}s |\n'
        self.log_msg(msg)

    # print and save evaluate phase result
    def log_evaluate(self, mode, res):
        msg = (f'\t| {mode:5} |\n\t| Improve | hr5_a  | hr20_a '
               f'| mrr5_a | mrr20_a | ndcg5_a | ndcg20_a | hr5_b  | hr20_b | mrr5_b | mrr20_b '
               f'| ndcg5_b | ndcg20_b |\n'
               f'\t| {res[0]:+.4f} | {res[1]:.4f} | {res[2]:.4f} | {res[3]:.4f} | {res[4]:.4f}  | {res[5]:.4f}  '
               f'| {res[6]:.4f}   | {res[7]:.4f} | {res[8]:.4f} | {res[9]:.4f} | {res[10]:.4f}  | {res[11]:.4f}  '
               f'| {res[12]:.4f}   |')
        msg += '\n' if mode == 'valid' else ''
        self.log_msg(msg)

    # print and save final result
    def log_final_result(self, epoch: int, imp_val_best: float, res):
        self.log_msg('\n' + '-' * 10 + f' C2DSR (CIKM\'22) experiment ends at epoch {epoch} ' + '-' * 10)
        self.log_settings()

        # extra spaces are for a satisfyingly aligned output format
        msg = (f'[ Valid result ]\n\t| Improve |\n\t| {imp_val_best:+.4f} |\n\n'
               f'[ Test result ]\n\t| Improve | hr5_a  | hr20_a | mrr5_a | mrr20_a | ndcg5_a | ndcg20_a | hr5_b  '
               f'| hr20_b | mrr5_b | mrr20_b | ndcg5_b | ndcg20_b |\n'
               f'\t| {res[0]:+.4f} | {res[1]:.4f} | {res[2]:.4f} | {res[3]:.4f} | {res[4]:.4f}  | {res[5]:.4f}  '
               f'| {res[6]:.4f}   | {res[7]:.4f} | {res[8]:.4f} | {res[9]:.4f} | {res[10]:.4f}  | {res[11]:.4f}  '
               f'| {res[12]:.4f}   |\n')
        self.log_msg(msg)
