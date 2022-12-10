import os
from os.path import join
import time


class Noter(object):

    def __init__(self, args):
        self.args = args

        self.cuda = args.cuda
        self.dataset = args.dataset
        self.n_gnn = args.n_gnn
        self.n_attn = args.n_attn
        self.n_head = args.n_head
        self.dropout = args.dropout
        self.weight_decay = args.weight_decay

        self.f_log = join(args.path_log, time.strftime('%m-%d-%H:%M-', time.localtime()) + args.dataset + '-' +
                          str(args.n_gnn) + '-' + str(args.n_attn) + '-' + str(args.n_head) + '.txt')

        if os.path.exists(self.f_log):
            os.remove(self.f_log)  # remove the existing file if duplicate

        self.welcome = ('-' * 20 + ' Experiment starts ' + '-' * 20)
        print('\n' + self.welcome)

    # write into log file
    def write(self, msg):
        with open(self.f_log, 'a') as out:
            print(msg, file=out)

    # log any message
    def log_msg(self, msg):
        print(msg)
        self.write(msg)

    # print and save experiment briefs
    def log_brief(self):
        msg = f'\n[Info] Experiment (dataset:{self.dataset}, cuda:{self.cuda}) ' \
              f'\n\t| n_gnn {self.n_gnn} | n_attn {self.n_attn} | n_head {self.n_head} | dropout {self.dropout} |' \
              f'\n\t| weight_decay {self.weight_decay} |'
        print(msg)
        self.write(self.welcome + '\n' + msg)

    # save args into log file
    def save_args(self):
        info = '-' * 10 + ' Experiment settings ' + '-' * 10 + '\n'
        for k, v in vars(self.args).items():
            info += '\n\t{} : {}'.format(k, str(v))
        self.write(info + '\n')

    # print and save train phase result
    def log_train(self, loss, loss_rec, loss_jump, t_gap):
        msg = (f'\t| train | loss {loss:.4f} | loss_rec {loss_rec:.4f} | loss_jump {loss_jump:.4f} '
               f'| time {t_gap:.1f}s |')
        print(msg)
        self.write(msg)

    # print and save evaluate phase result
    def log_evaluate(self, msg, res):
        msg += f'\n\t| mrr_x {res[0]:.4f} | ndcg_x_5 {res[1]:.4f} | ndcg_x_10 {res[2]:.4f} ' \
               f'| hr_x_1 {res[3]:.4f} | hr_x_5 {res[4]:.4f} | ndcg_x_10 {res[5]:.4f} |' \
               f'\n\t| mrr_y {res[6]:.4f} | ndcg_y_5 {res[7]:.4f} | ndcg_y_10 {res[8]:.4f} |' \
               f'| hr_y_1 {res[9]:.4f} | hr_y_5 {res[10]:.4f} | hr_y_10 {res[11]:.4f} |\n'
        print(msg)
        self.write(msg)

    # print and save final result
    def log_final_result(self, epoch: int, dict_res: dict):
        msg = f'\n[info] Modeling ends at epoch {epoch}' + '\n'

        for mode, res in dict_res.items():
            msg += f'\n[ {mode} ]'
            msg += f'\n\t| mrr_x {res[0]:.4f} | ndcg_x_5 {res[1]:.4f} | ndcg_x_10 {res[2]:.4f} ' \
                   f'| hr_x_1 {res[3]:.4f} | hr_x_5 {res[4]:.4f} | ndcg_x_10 {res[5]:.4f} |' \
                   f'\n\t| mrr_y {res[6]:.4f} | ndcg_y_5 {res[7]:.4f} | ndcg_y_10 {res[8]:.4f} |' \
                   f'| hr_y_1 {res[9]:.4f} | hr_y_5 {res[10]:.4f} | hr_y_10 {res[11]:.4f} |\n'
        msg += '\n'
        print(msg)
        self.write(msg)
