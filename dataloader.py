from os.path import join
import numpy as np
import random
import codecs
import pickle
import copy
import torch
from torch.utils.data import Dataset, DataLoader


class CDSRDataset(Dataset):
    def __init__(self, args, mode):
        self.mode = mode
        self.batch_size = args.batch_size
        self.dataset = args.dataset
        self.n_item_x = args.n_item_x
        self.n_item_y = args.n_item_y
        self.idx_pad = args.n_item_x + args.n_item_y
        self.len_max = args.len_max
        self.n_neg_sample = args.n_neg_sample

        if args.use_raw:
            if mode == 'train':
                data = self.read_raw(join(args.path_raw, mode + '_new.txt'), is_train=True)
                self.data = self.preprocess_train(data)
            else:
                data = self.read_raw(join(args.path_raw, mode + '_new.txt'), is_train=False)
                self.data = self.preprocess_evaluate(data)
            if args.save_processed:
                with open(join(args.path_data, mode + '.pkl'), 'wb') as f:
                    pickle.dump(self.data, f)
        else:
            with open(join(args.path_data, mode + '.pkl'), 'rb') as f:
                self.data = pickle.load(f)

    def read_raw(self, path_file, is_train=True):
        def takeSecond(elem):
            return elem[1]
        with codecs.open(path_file, 'r', encoding='utf-8') as f:
            data = []
            for line in f:
                res = []
                line = line.strip().split('\t')[2:]
                for w in line:
                    w = w.split('|')
                    res.append((int(w[0]), int(w[1])))
                res.sort(key=takeSecond)
                res_2 = []
                for r in res[:-1]:
                    res_2.append(r[0])

                if not is_train:
                    # denoted the corresponding validation/test entry
                    if res[-1][0] >= self.n_item_x:
                        data.append([res_2, 1, res[-1][0]])

                    else:
                        data.append([res_2, 0, res[-1][0]])
                else:
                    data.append(res_2)
        return data

    def preprocess_train(self, data):
        """ Preprocess the data and convert to ids. """
        processed = []
        for seq in data:
            gt = copy.deepcopy(seq)[1:]
            seq = seq[:-1]
            len_seq = len(seq)

            # gt
            gt_share_x, gt_share_y, gt_mask_share_x, gt_mask_share_y = [], [], [], []
            for w in gt:
                if w < self.n_item_x:
                    gt_share_x.append(w)
                    gt_mask_share_x.append(1)
                    gt_share_y.append(self.n_item_y)
                    gt_mask_share_y.append(0)
                else:
                    gt_share_x.append(self.n_item_x)
                    gt_mask_share_x.append(0)
                    gt_share_y.append(w - self.n_item_x)
                    gt_mask_share_y.append(1)

            # seq
            pos = list(range(1, len(seq)+1))
            gt_mask = [1] * len_seq

            seq_x, x_count, pos_x = [], 1, []
            seq_y, y_count, pos_y = [], 1, []
            crpt_x, crpt_y = [], []

            for w in seq:
                if w < self.n_item_x:
                    crpt_x.append(w)
                    seq_x.append(w)
                    pos_x.append(x_count)
                    x_count += 1
                    crpt_y.append(random.randint(0, self.n_item_x - 1))
                    seq_y.append(self.idx_pad)
                    pos_y.append(0)

                else:
                    crpt_x.append(random.randint(self.n_item_x, self.idx_pad - 1))
                    seq_x.append(self.idx_pad)
                    pos_x.append(0)
                    crpt_y.append(w)
                    seq_y.append(w)
                    pos_y.append(y_count)
                    y_count += 1

            now = -1
            x_gt = [self.n_item_x] * len(seq_x)  # caution!
            gt_mask_x = [0] * len(seq_x)
            for i in range(1, len(seq_x)+1):
                if pos_x[-i]:
                    if now == -1:
                        now = seq_x[-i]
                        if gt[-1] < self.n_item_x:
                            x_gt[-i] = gt[-1]
                            gt_mask_x[-i] = 1
                        else:
                            seq_x[-i] = self.idx_pad
                            pos_x[-i] = 0
                    else:
                        x_gt[-i] = now
                        gt_mask_x[-i] = 1
                        now = seq_x[-i]
            if sum(gt_mask_x) == 0:
                continue

            now = -1
            y_gt = [self.n_item_y] * len(seq_y)  # caution!
            gt_mask_y = [0] * len(seq_y)
            for i in range(1, len(seq_y)+1):
                if pos_y[-i]:
                    if now == -1:
                        now = seq_y[-i] - self.n_item_x
                        if gt[-1] > self.n_item_x:
                            y_gt[-i] = gt[-1] - self.n_item_x
                            gt_mask_y[-i] = 1
                        else:
                            seq_y[-i] = self.idx_pad
                            pos_y[-i] = 0
                    else:
                        y_gt[-i] = now
                        gt_mask_y[-i] = 1
                        now = seq_y[-i] - self.n_item_x
            if sum(gt_mask_y) == 0:
                continue

            if len(seq) < self.len_max:
                pos = [0] * (self.len_max - len(seq)) + pos
                pos_x = [0] * (self.len_max - len(seq)) + pos_x
                pos_y = [0] * (self.len_max - len(seq)) + pos_y

                gt = [self.idx_pad] * (self.len_max - len(seq)) + gt
                gt_share_x = [self.n_item_x] * (self.len_max - len(seq)) + gt_share_x
                gt_share_y = [self.n_item_y] * (self.len_max - len(seq)) + gt_share_y
                x_gt = [self.n_item_x] * (self.len_max - len(seq)) + x_gt
                y_gt = [self.n_item_y] * (self.len_max - len(seq)) + y_gt

                gt_mask = [0] * (self.len_max - len(seq)) + gt_mask
                gt_mask_share_x = [0] * (self.len_max - len(seq)) + gt_mask_share_x
                gt_mask_share_y = [0] * (self.len_max - len(seq)) + gt_mask_share_y
                gt_mask_x = [0] * (self.len_max - len(seq)) + gt_mask_x
                gt_mask_y = [0] * (self.len_max - len(seq)) + gt_mask_y

                crpt_x = [self.idx_pad] * (self.len_max - len(seq)) + crpt_x
                crpt_y = [self.idx_pad] * (self.len_max - len(seq)) + crpt_y
                seq_x = [self.idx_pad] * (self.len_max - len(seq)) + seq_x
                seq_y = [self.idx_pad] * (self.len_max - len(seq)) + seq_y
                seq = [self.idx_pad] * (self.len_max - len(seq)) + seq

            processed.append([seq, seq_x, seq_y, pos, pos_x, pos_y, gt, gt_share_x, gt_share_y, x_gt, y_gt, gt_mask,
                              gt_mask_share_x, gt_mask_share_y, gt_mask_x, gt_mask_y, crpt_x, crpt_y])
        return processed

    def preprocess_evaluate(self, data):
        processed = []
        for [seq, XorY, gt] in data:
            len_seq = len(seq)
            pos = list(range(len_seq+1))[1:]
            seq_x, x_count, pos_x = [], 1, []
            seq_y, y_count, pos_y = [], 1, []

            for w in seq:
                if w < self.n_item_x:
                    seq_x.append(w)
                    pos_x.append(x_count)
                    x_count += 1
                    seq_y.append(self.idx_pad)
                    pos_y.append(0)

                else:
                    seq_x.append(self.idx_pad)
                    pos_x.append(0)
                    seq_y.append(w)
                    pos_y.append(y_count)
                    y_count += 1

            if len_seq < self.len_max:
                pos = [0] * (self.len_max - len_seq) + pos
                pos_x = [0] * (self.len_max - len_seq) + pos_x
                pos_y = [0] * (self.len_max - len_seq) + pos_y

                seq_x = [self.idx_pad] * (self.len_max - len_seq) + seq_x
                seq_y = [self.idx_pad] * (self.len_max - len_seq) + seq_y
                seq = [self.idx_pad]*(self.len_max - len_seq) + seq

            gt_x = -1
            for i in range(1, len(pos_x)+1):
                if pos_x[-i]:
                    gt_x = -i
                    break

            gt_y = -1
            for i in range(1, len(pos_y)+1):
                if pos_y[-i]:
                    gt_y = -i
                    break

            if XorY:  # XorY == 1
                idx_neg = random.sample(list(range(0, gt - self.n_item_x)) +
                                        list(range(gt - self.n_item_x + 1, self.n_item_y - 1)), self.n_neg_sample)
                processed.append([seq, seq_x, seq_y, pos, pos_x, pos_y, gt_x, gt_y, XorY, gt - self.n_item_x, idx_neg])

            else:  # XorY == 0
                idx_neg = random.sample(list(range(0, gt)) +
                                        list(range(gt + 1, self.n_item_x - 1)), self.n_neg_sample)
                processed.append([seq, seq_x, seq_y, pos, pos_x, pos_y, gt_x, gt_y, XorY, gt, idx_neg])

        return processed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        """ Get a batch with index. """
        data = self.data[key]
        if self.mode != 'train':
            return torch.LongTensor(data[0]), torch.LongTensor(data[1]), torch.LongTensor(data[2]),\
                torch.LongTensor(data[3]), torch.LongTensor(data[4]), torch.LongTensor(data[5]),\
                torch.LongTensor(data[6]), torch.LongTensor(data[7]), torch.LongTensor(data[8]),\
                torch.LongTensor(data[9]), torch.LongTensor(data[10])
        else:
            return torch.LongTensor(data[0]), torch.LongTensor(data[1]), torch.LongTensor(data[2]),\
                torch.LongTensor(data[3]), torch.LongTensor(data[4]), torch.LongTensor(data[5]),\
                torch.LongTensor(data[6]), torch.LongTensor(data[7]), torch.LongTensor(data[8]),\
                torch.LongTensor(data[9]), torch.LongTensor(data[10]), torch.LongTensor(data[11]),\
                torch.LongTensor(data[12]), torch.LongTensor(data[13]), torch.LongTensor(data[14]),\
                torch.LongTensor(data[15]), torch.LongTensor(data[16]), torch.LongTensor(data[17])


def count_item(path):
    count = 0
    with codecs.open(path, 'r', encoding='utf-8') as f:
        for _ in f:
            count += 1
    return count


def get_dataloader(args):

    p = args.path_raw if args.use_raw else args.path_data

    args.n_item_x = count_item(join(p, 'items_x.txt'))
    args.n_item_y = count_item(join(p, 'items_y.txt'))
    args.n_item = args.n_item_x + args.n_item_y + 1  # with padding
    args.idx_pad = args.n_item - 1

    trainloader = DataLoader(CDSRDataset(args, mode='train'), batch_size=args.batch_size, shuffle=True,
                             num_workers=args.num_workers)
    valloader = DataLoader(CDSRDataset(args, mode='val'), batch_size=args.batch_size_eval, shuffle=False, num_workers=args.num_workers)
    testloader = DataLoader(CDSRDataset(args, mode='test'), batch_size=args.batch_size_eval, shuffle=False, num_workers=args.num_workers)

    return trainloader, valloader, testloader
