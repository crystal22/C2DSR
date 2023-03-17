import os
from os.path import join
import argparse
import numpy as np
import random
import torch

from trainer import Trainer
from utils.noter import Noter
from utils.constant import MAPPING_DATASET, BENCHMARKS
from utils.metrics import cal_score


def main():
    parser = argparse.ArgumentParser(description='C2DSR')

    # Experiment
    parser.add_argument('--data', type=str, default='fk', help='fk: Food-Kitchen'
                                                               'mb: Movie-Book'
                                                               'ee: Entertainment-Education')
    parser.add_argument('--len_rec', type=int, default=10, help='window length of sequence for recommendation')

    # data
    parser.add_argument('--use_raw', action='store_true', help='use raw data from C2DSR, takes longer time')
    parser.add_argument('--n_neg_sample', type=int, default=999, help='# negative samples')
    parser.add_argument('--zip_ee', action='store_true', help='zip Ent.-Edu. dataset')

    # Model
    parser.add_argument('--d_latent', type=int, default=128, help='dimension of latent representation')
    parser.add_argument('--disable_embed_l2', action='store_true', help='disable l2 regularization on embedding')
    parser.add_argument('--shared_item_embed', action='store_true',
                        help='shared item embedding for a, b and merged domains')
    parser.add_argument('--d_bias', action='store_true', help='bias of bilinear classifier for contrastive learning')

    # GNN
    parser.add_argument('--n_gnn', type=int, default=1, help='# layer of GNN implemented')
    parser.add_argument('--dropout_gnn', type=float, default=0.2, help='dropout rate for gnn')

    # Transformer
    parser.add_argument('--n_attn', type=int, default=1, help='# layer of TransformerEncoderLayer stack')
    parser.add_argument('--n_head', type=int, default=1, help='# multi-head for self-attention')
    parser.add_argument('--dropout_attn', type=float, default=0.2, help='dropout rate for Transformer')
    parser.add_argument('--norm_first', action='store_true', help='pre norm on Transformer encoder')

    # Optimizer
    parser.add_argument('--lr', type=float, default=1e-3, help='Applies to sgd and adagrad.')
    parser.add_argument('--lr_decay', type=float, default=0.1, help='Learning rate decay rate.')
    parser.add_argument('--l2', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--lr_gamma', type=float, default=0.5)
    parser.add_argument('--lr_step', type=int, default=10)
    parser.add_argument('--n_lr_decay', type=int, default=5)

    parser.add_argument('--decay_epoch', type=int, default=5, help='Decay learning rate after this epoch.')
    parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')
    parser.add_argument('--len_max', type=int, default=15)
    parser.add_argument('--lambda_loss', type=float, default=0.7)

    # Training
    parser.add_argument('--cuda', type=str, default='0', help='running device')
    parser.add_argument('--seed', type=int, default=3407, help='random seeding')
    parser.add_argument('--n_epoch', type=int, default=200, help='# epoch maximum')
    parser.add_argument('--batch_size', type=int, default=512, help='size of batch for training')
    parser.add_argument('--batch_size_eval', type=int, default=2048, help='size of batch for evaluation')
    parser.add_argument('--num_workers', type=int, default=1, help='# dataloader worker')
    parser.add_argument('--es_patience', type=int, default=10)

    args = parser.parse_args()

    args.dataset = MAPPING_DATASET[args.data]
    args.benchmark = BENCHMARKS[args.data]
    args.len_max = 30 if args.dataset == 'Entertainment-Education' else 15
    if args.cuda == 'cpu':
        args.device = torch.device('cpu')
    else:
        args.device = torch.device('cuda:' + args.cuda)

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

    # seeding
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # modeling
    noter = Noter(args)
    trainer = Trainer(args, noter)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer.optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

    epoch, es_counter = 0, 0
    imp_val_best = -1
    res_test_imp = [0] * 13
    lr_register = args.lr

    for epoch in range(1, args.n_epoch + 1):
        # training and validation phase
        noter.log_msg(f'\n[Epoch {epoch}]')
        pred_val_a, pred_val_b = trainer.run_epoch()
        res_val_epoch = cal_score(pred_val_a, pred_val_b, args.benchmark)
        scheduler.step()
        noter.log_evaluate('valid', res_val_epoch)

        # model selection
        imp_val_epoch, flag_imp = res_val_epoch[0], False
        if imp_val_epoch > imp_val_best:
            imp_val_best = imp_val_epoch

            # testing phase
            res_test_epoch = cal_score(*trainer.run_test(), args.benchmark)
            res_test_imp = res_test_epoch

            noter.log_evaluate('test', res_test_epoch)

            # early stop: reset
            es_counter = 0
        else:
            # early stop: overfitting
            es_counter += 1
            noter.log_msg(f'\t| es    | {es_counter} / {args.es_patience} |')

            if es_counter >= args.es_patience:
                break

        # notice of changing lr
        lr_current = trainer.scheduler.get_last_lr()[0]
        if lr_register != lr_current:
            if trainer.optimizer.param_groups[0]['lr'] == args.lr_min:
                noter.log_msg(f'\t| lr    | reaches btm | {args.lr_min:.2e} |')
            else:
                noter.log_msg(f'\t| lr    | from {lr_register:.2e} | to {lr_current:.2e} |')
                lr_register = lr_current

    noter.log_final_result(epoch, imp_val_best, res_test_imp)


if __name__ == '__main__':
    main()
