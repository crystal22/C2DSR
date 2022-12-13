from os.path import join
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
        self.len_max = args.len_max

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
        for d in data:  # the pad is needed! but to be careful.
            gt = copy.deepcopy(d)[1:]
            share_x_gt, share_y_gt, share_x_gt_mask, share_y_gt_mask = [], [], [], []
            for w in gt:
                if w < self.n_item_x:
                    share_x_gt.append(w)
                    share_x_gt_mask.append(1)
                    share_y_gt.append(self.n_item_y)
                    share_y_gt_mask.append(0)
                else:
                    share_x_gt.append(self.n_item_x)
                    share_x_gt_mask.append(0)
                    share_y_gt.append(w - self.n_item_x)
                    share_y_gt_mask.append(1)

            d = d[:-1]  # delete the ground truth
            pos = list(range(len(d)+1))[1:]
            gt_mask = [1] * len(d)

            xd, x_count, pos_x = [], 1, []
            yd, y_count, pos_y = [], 1, []
            x_corrupt, y_corrupt = [], []

            for w in d:
                if w < self.n_item_x:
                    x_corrupt.append(w)
                    xd.append(w)
                    pos_x.append(x_count)
                    x_count += 1
                    y_corrupt.append(random.randint(0, self.n_item_x - 1))
                    yd.append(self.n_item_x + self.n_item_y)
                    pos_y.append(0)

                else:
                    x_corrupt.append(random.randint(self.n_item_x, self.n_item_x + self.n_item_y - 1))
                    xd.append(self.n_item_x + self.n_item_y)
                    pos_x.append(0)
                    y_corrupt.append(w)
                    yd.append(w)
                    pos_y.append(y_count)
                    y_count += 1

            now = -1
            x_gt = [self.n_item_x] * len(xd)  # caution!
            x_gt_mask = [0] * len(xd)
            for i in range(1, len(xd)+1):
                if pos_x[-i]:
                    if now == -1:
                        now = xd[-i]
                        if gt[-1] < self.n_item_x:
                            x_gt[-i] = gt[-1]
                            x_gt_mask[-i] = 1
                        else:
                            xd[-i] = self.n_item_x + self.n_item_y
                            pos_x[-i] = 0
                    else:
                        x_gt[-i] = now
                        x_gt_mask[-i] = 1
                        now = xd[-i]
            if sum(x_gt_mask) == 0:
                continue

            now = -1
            y_gt = [self.n_item_y] * len(yd)  # caution!
            y_gt_mask = [0] * len(yd)
            for i in range(1, len(yd)+1):
                if pos_y[-i]:
                    if now == -1:
                        now = yd[-i] - self.n_item_x
                        if gt[-1] > self.n_item_x:
                            y_gt[-i] = gt[-1] - self.n_item_x
                            y_gt_mask[-i] = 1
                        else:
                            yd[-i] = self.n_item_x + self.n_item_y
                            pos_y[-i] = 0
                    else:
                        y_gt[-i] = now
                        y_gt_mask[-i] = 1
                        now = yd[-i] - self.n_item_x
            if sum(y_gt_mask) == 0:
                continue

            if len(d) < self.len_max:
                pos = [0] * (self.len_max - len(d)) + pos
                pos_x = [0] * (self.len_max - len(d)) + pos_x
                pos_y = [0] * (self.len_max - len(d)) + pos_y

                gt = [self.n_item_x + self.n_item_y] * (self.len_max - len(d)) + gt
                share_x_gt = [self.n_item_x] * (self.len_max - len(d)) + share_x_gt
                share_y_gt = [self.n_item_y] * (self.len_max - len(d)) + share_y_gt
                x_gt = [self.n_item_x] * (self.len_max - len(d)) + x_gt
                y_gt = [self.n_item_y] * (self.len_max - len(d)) + y_gt

                gt_mask = [0] * (self.len_max - len(d)) + gt_mask
                share_x_gt_mask = [0] * (self.len_max - len(d)) + share_x_gt_mask
                share_y_gt_mask = [0] * (self.len_max - len(d)) + share_y_gt_mask
                x_gt_mask = [0] * (self.len_max - len(d)) + x_gt_mask
                y_gt_mask = [0] * (self.len_max - len(d)) + y_gt_mask

                x_corrupt = [self.n_item_x + self.n_item_y] * (self.len_max - len(d)) + x_corrupt
                y_corrupt = [self.n_item_x + self.n_item_y] * (self.len_max - len(d)) + y_corrupt
                xd = [self.n_item_x + self.n_item_y] * (self.len_max - len(d)) + xd
                yd = [self.n_item_x + self.n_item_y] * (self.len_max - len(d)) + yd
                d = [self.n_item_x + self.n_item_y] * (self.len_max - len(d)) + d

            processed.append([d, xd, yd, pos, pos_x, pos_y, gt, share_x_gt, share_y_gt, x_gt, y_gt, gt_mask,
                              share_x_gt_mask, share_y_gt_mask, x_gt_mask, y_gt_mask, x_corrupt, y_corrupt])
        return processed

    def preprocess_evaluate(self, data):
        processed = []
        for d in data:  # the pad is needed! but to be careful.
            pos = list(range(len(d[0])+1))[1:]
            xd, x_count, pos_x = [], 1, []
            yd, y_count, pos_y = [], 1, []

            for w in d[0]:
                if w < self.n_item_x:
                    xd.append(w)
                    pos_x.append(x_count)
                    x_count += 1
                    yd.append(self.n_item_x + self.n_item_y)
                    pos_y.append(0)

                else:
                    xd.append(self.n_item_x + self.n_item_y)
                    pos_x.append(0)
                    yd.append(w)
                    pos_y.append(y_count)
                    y_count += 1

            if len(d[0]) < self.len_max:
                pos = [0] * (self.len_max - len(d[0])) + pos
                pos_x = [0] * (self.len_max - len(d[0])) + pos_x
                pos_y = [0] * (self.len_max - len(d[0])) + pos_y

                xd = [self.n_item_x + self.n_item_y] * (self.len_max - len(d[0])) + xd
                yd = [self.n_item_x + self.n_item_y] * (self.len_max - len(d[0])) + yd
                seq = [self.n_item_x + self.n_item_y]*(self.len_max - len(d[0])) + d[0]

            x_last = -1
            for i in range(1, len(pos_x)+1):
                if pos_x[-i]:
                    x_last = -i
                    break

            y_last = -1
            for i in range(1, len(pos_y)+1):
                if pos_y[-i]:
                    y_last = -i
                    break

            neg_samples = []
            for i in range(999):  # need re-format
                while True:
                    if d[1]:  # in Y domain, the validation/test negative samples
                        sample = random.randint(0, self.n_item_y - 1)
                        if sample != d[2] - self.n_item_x:
                            neg_samples.append(sample)
                            break
                    else:  # in X domain, the validation/test negative samples
                        sample = random.randint(0, self.n_item_x - 1)
                        if sample != d[2]:
                            neg_samples.append(sample)
                            break

            if d[1]:
                processed.append([seq, xd, yd, pos, pos_x, pos_y, x_last, y_last, d[1], d[2]-self.n_item_x, neg_samples])
            else:
                processed.append([seq, xd, yd, pos, pos_x, pos_y, x_last, y_last, d[1], d[2], neg_samples])
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

    trainloader = DataLoader(CDSRDataset(args, mode='train'), batch_size=args.batch_size, shuffle=True, num_workers=3)
    valloader = DataLoader(CDSRDataset(args, mode='val'), batch_size=64, shuffle=False, num_workers=3)
    testloader = DataLoader(CDSRDataset(args, mode='test'), batch_size=64, shuffle=False, num_workers=3)

    return trainloader, valloader, testloader
