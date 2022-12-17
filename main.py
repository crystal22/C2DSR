import os
from os.path import join
import argparse
import numpy as np
import random
import torch

from trainer import Trainer
from utils.noter import Noter
from utils.constant import MAPPING_DATASET
from utils.metrics import cal_score


def main():
    parser = argparse.ArgumentParser(description='C2DSR')

    # Experiment
    parser.add_argument('--data', type=str, default='ee', help='fk: Food-Kitchen'
                                                               'mb: Movie-Book'
                                                               'ee: Entertainment-Education')
    parser.add_argument('--len_rec', type=int, default=10, help='window length of sequence for recommendation')

    # data
    parser.add_argument('--use_raw', action='store_true', help='use raw data from C2DSR, takes longer time')
    parser.add_argument('--save_processed', action='store_true', help='use raw data from C2DSR, takes longer time')
    parser.add_argument('--n_neg_sample', type=int, default=999, help='# negative samples')

    # Model
    parser.add_argument('--d_latent', type=int, default=256, help='dimension of latent representation')
    parser.add_argument('--disable_embed_l2', action='store_true', help='disable l2 regularization on embedding')
    parser.add_argument('--shared_item_embed', action='store_true',
                        help='shared item embedding for a, b and merged domains')
    parser.add_argument('--d_bias', action='store_true', help='bias of bilinear classifier for contrastive learning')

    # GNN
    parser.add_argument('--n_gnn', type=int, default=1, help='# layer of GNN implemented')
    parser.add_argument('--dropout_gnn', type=float, default=0.2, help='dropout rate for gnn')

    # Transformer
    parser.add_argument('--n_attn', type=int, default=2, help='# layer of TransformerEncoderLayer stack')
    parser.add_argument('--n_head', type=int, default=1, help='# multi-head for self-attention')
    parser.add_argument('--dropout_attn', type=float, default=0.2, help='dropout rate for Transformer')
    parser.add_argument('--norm_first', action='store_true', help='pre norm on Transformer encoder')

    # optimizer
    parser.add_argument('--optim', choices=['sgd', 'adagrad', 'adam', 'adamax'], default='adam',
                        help='Optimizer: sgd, adagrad, adam or adamax.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Applies to sgd and adagrad.')
    parser.add_argument('--lr_decay', type=float, default=0.1, help='Learning rate decay rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--lr_gamma', type=float, default=0.5)
    parser.add_argument('--lr_step', type=int, default=10)
    parser.add_argument('--n_lr_decay', type=int, default=5)

    parser.add_argument('--decay_epoch', type=int, default=5, help='Decay learning rate after this epoch.')
    parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')
    parser.add_argument('--len_max', type=int, default=15)
    parser.add_argument('--lambda_loss', type=float, default=0.7)

    # train part
    parser.add_argument('--cuda', type=str, default='0', help='running device')
    parser.add_argument('--seed', type=int, default=3407, help='random seeding')
    parser.add_argument('--n_epoch', type=int, default=200, help='# epoch maximum')
    parser.add_argument('--batch_size', type=int, default=512, help='size of batch for training')
    parser.add_argument('--batch_size_eval', type=int, default=2048, help='size of batch for evaluation')
    parser.add_argument('--num_workers', type=int, default=8, help='# dataloader worker')
    parser.add_argument('--es_patience', type=int, default=10)

    args = parser.parse_args()

    args.dataset = MAPPING_DATASET[args.data]
    args.path_root = os.getcwd()
    args.path_data = join(args.path_root, 'data', args.dataset)
    args.path_raw = join(args.path_root, 'data', 'raw', args.dataset)
    args.path_ckpt = join(args.path_root, 'checkpoints')
    args.path_log = join(args.path_root, 'log')
    for p in (args.path_ckpt, args.path_log):
        if not os.path.exists(p):
            os.makedirs(p)

    if args.use_raw and not os.path.exists(args.path_raw):
        raise FileNotFoundError(f'Selected raw dataset {args.dataset} does not exist..')
    if not args.use_raw and not os.path.exists(args.path_data):
        raise FileNotFoundError(f'Selected processed dataset {args.dataset} does not exist..')
    if args.save_processed:
        args.use_raw = True

    # device
    if args.cuda == 'cpu':
        args.device = torch.device('cpu')
    else:
        args.device = torch.device('cuda:' + args.cuda)

    # seeding
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # settings
    args.len_max = 30 if args.dataset == 'Entertainment-Education' else 15
    es_counter = 0

    # initialize
    noter = Noter(args)
    trainer = Trainer(args, noter)

    # modeling
    epoch, loss_tr, mrr_val_best_x, mrr_val_best_y = 0, 1e5, 0, 0
    res_best_val_x, res_best_val_y, res_val_epoch = [0] * 12, [0] * 12, [0] * 12
    res_test_x, res_test_y, res_test_epoch = [0] * 12, [0] * 12, [0] * 12
    lr_register = args.lr

    for epoch in range(1, args.n_epoch + 1):
        # training and validation phase
        noter.log_msg(f'\n[Epoch {epoch}]')

        loss_tr, pred_val_x, pred_val_y = trainer.run_epoch()
        res_val_x, res_val_y = cal_score(pred_val_x), cal_score(pred_val_y)
        res_val_epoch = res_val_x + res_val_y

        msg_best_val = ''

        if res_val_x[0] > mrr_val_best_x:
            mrr_val_best_x = res_val_x[0]
            msg_best_val += ' new x |'
            res_best_val_x = res_val_x + res_val_y

        if res_val_y[0] > mrr_val_best_y:
            mrr_val_best_y = res_val_y[0]
            msg_best_val += ' new y |'
            res_best_val_y = res_val_x + res_val_y

        noter.log_msg('\t| valid |' + msg_best_val)
        noter.log_evaluate(res_val_epoch)

        # testing phase
        if 'x' in msg_best_val or 'y' in msg_best_val:
            pred_test_x, pred_test_y = trainer.run_test()
            res_test_epoch = cal_score(pred_test_x) + cal_score(pred_test_y)
            noter.log_msg('\t| test  |' + msg_best_val)
            noter.log_evaluate(res_test_epoch)

            if 'x' in msg_best_val:
                res_test_x = res_test_epoch
            if 'y' in msg_best_val:
                res_test_y = res_test_epoch

        # notice of changing lr
        lr_current = trainer.scheduler.get_last_lr()[0]
        if lr_register != lr_current:
            if trainer.optimizer.param_groups[0]['lr'] == args.lr_min:
                noter.log_msg(f'\t| lr    | reaches btm | {args.lr_min:.2e} |')
            else:
                noter.log_msg(f'\t| lr    | from {lr_register:.2e} | to {lr_current:.2e} |')
                lr_register = lr_current

        # early stop: overfitting
        if not (flag_test_x or flag_test_y):
            es_counter += 1
            noter.log_msg(f'\n\t| es    | {es_counter} / {args.es_patience} |')
        elif es_counter != 0:
            es_counter = 0
            noter.log_msg(f'\t| es    | 0 / {args.es_patience} |')
        if es_counter >= args.es_patience:
            break

    noter.log_final_result(epoch, {
        'Best val x': res_best_val_x,
        'Best val y': res_best_val_y,
        'Best test x': res_test_x,
        'Best test y': res_test_y
    })


if __name__ == '__main__':
    main()
