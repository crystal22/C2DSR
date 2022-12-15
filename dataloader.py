from os.path import join
import random
import codecs
import pickle
import torch
from torch.utils.data import Dataset, DataLoader


class CDSRDataset(Dataset):
    def __init__(self, args, mode):
        self.mode = mode
        self.path_raw = args.path_raw
        self.device = args.device
        self.batch_size = args.batch_size
        self.dataset = args.dataset
        self.n_item_x = args.n_item_x
        self.n_item_y = args.n_item_y
        self.idx_pad = args.idx_pad
        self.len_max = args.len_max
        self.n_neg_sample = args.n_neg_sample

        if args.use_raw:
            if mode == 'train':
                self.data = self.preprocess_train()
            else:
                self.data = self.preprocess_evaluate()
            if args.save_processed:
                with open(join(args.path_data, mode + '.pkl'), 'wb') as f:
                    pickle.dump(self.data, f)
        else:
            with open(join(args.path_data, mode + '.pkl'), 'rb') as f:
                self.data = pickle.load(f)

        self.length = len(self.data)

    def read_raw(self):

        def takeSecond(elem):
            return elem[1]

        with codecs.open(join(self.path_raw, self.mode + '_new.txt'), 'r', encoding='utf-8') as f:
            data = []
            for line in f:
                seq_ui = []
                line = line.strip().split('\t')[2:]
                for ui in line:
                    ui = ui.split('|')
                    seq_ui.append((int(ui[0]), int(ui[1])))
                seq_ui.sort(key=takeSecond)
                seq_i = []
                for r in seq_ui:
                    seq_i.append(r[0])

                if self.mode == 'train':
                    data.append(seq_i)
                else:
                    if seq_ui[-1][0] >= self.n_item_x:
                        data.append([seq_i, 1, seq_ui[-1][0]])
                    else:
                        data.append([seq_i, 0, seq_ui[-1][0]])
        return data

    def preprocess_train(self):
        """ Preprocess the raw sequence data for training. """
        data = self.read_raw()
        processed = []
        for u_behavior in data:
            # extract from sequences
            gt = u_behavior[1:]
            seq_in = u_behavior[:-1]
            len_seq = len(u_behavior)

            pos = list(range(1, len_seq))

            x_count, seq_in_x, pos_x, seq_crpt_x = 1, [], [], []
            y_count, seq_in_y, pos_y, seq_crpt_y = 1, [], [], []

            for i, idx in enumerate(seq_in):
                if idx < self.n_item_x:
                    seq_crpt_x.append(idx)
                    seq_in_x.append(idx)
                    pos_x.append(x_count)
                    x_count += 1
                    seq_crpt_y.append(random.randint(0, self.n_item_x - 1))
                    seq_in_y.append(self.idx_pad)
                    pos_y.append(0)

                else:
                    seq_crpt_x.append(random.randint(self.n_item_x, self.idx_pad - 1))
                    seq_in_x.append(self.idx_pad)
                    pos_x.append(0)
                    seq_crpt_y.append(idx)
                    seq_in_y.append(idx)
                    pos_y.append(y_count)
                    y_count += 1

            # formulate domain-specific sequences, pad for x and y are different with its own length of item set
            gt_x = [self.n_item_x] * len(seq_in_x)
            gt_mask_x = [0] * len(seq_in_x)
            cur = -1
            for i in range(1, len_seq - 1):
                if pos_x[-i]:  # find last non-pad item
                    if cur == -1:  # fetch ground truth for domain-specific seq and pad it in inputs
                        cur = seq_in_x[-i]
                        if gt[-1] < self.n_item_x:
                            gt_x[-i] = gt[-1]
                            gt_mask_x[-i] = 1
                        else:
                            seq_in_x[-i] = self.idx_pad
                            pos_x[-i] = 0
                    else:
                        gt_x[-i] = cur
                        gt_mask_x[-i] = 1
                        cur = seq_in_x[-i]
            if sum(gt_mask_x) == 0:
                continue

            cur = -1
            gt_y = [self.n_item_y] * len(seq_in_y)  # caution!
            gt_mask_y = [0] * len(seq_in_y)
            for i in range(1, len(seq_in_y)):
                if pos_y[-i]:
                    if cur == -1:
                        cur = seq_in_y[-i] - self.n_item_x
                        if gt[-1] > self.n_item_x:
                            gt_y[-i] = gt[-1] - self.n_item_x
                            gt_mask_y[-i] = 1
                        else:
                            seq_in_y[-i] = self.idx_pad
                            pos_y[-i] = 0
                    else:
                        gt_y[-i] = cur
                        gt_mask_y[-i] = 1
                        cur = seq_in_y[-i] - self.n_item_x
            if sum(gt_mask_y) == 0:
                continue

            # pad sequence
            len_pad = self.len_max - len_seq + 1

            seq_in = [self.idx_pad] * len_pad + seq_in
            seq_in_x = [self.idx_pad] * len_pad + seq_in_x
            seq_in_y = [self.idx_pad] * len_pad + seq_in_y
            seq_crpt_x = [self.idx_pad] * len_pad + seq_crpt_x
            seq_crpt_y = [self.idx_pad] * len_pad + seq_crpt_y

            pos = [0] * len_pad + pos
            pos_x = [0] * len_pad + pos_x
            pos_y = [0] * len_pad + pos_y

            # generate ground truth
            gt = [self.idx_pad] * len_pad + gt
            gt_share_x = [idx if idx < self.n_item_x else self.n_item_x for idx in gt]
            gt_share_y = [idx - self.n_item_x if idx >= self.n_item_x else self.n_item_y for idx in gt]
            gt_x = [self.n_item_x] * len_pad + gt_x
            gt_y = [self.n_item_y] * len_pad + gt_y

            gt_mask = [1] * (len_seq - 1)
            gt_mask_x = [0] * len_pad + gt_mask_x
            gt_mask_y = [0] * len_pad + gt_mask_y

            processed.append([seq_in, seq_in_x, seq_in_y, pos, pos_x, pos_y, gt, gt_share_x, gt_share_y, gt_x, gt_y,
                              gt_mask, gt_mask_x, gt_mask_y, seq_crpt_x, seq_crpt_y])
        return processed

    def preprocess_evaluate(self):
        """ Preprocess the raw sequence data for evaluation including validation and testing. """
        data = self.read_raw()
        processed = []
        for [seq_in, XorY, gt] in data:
            len_in = len(seq_in)
            pos = list(range(len_in+1))[1:]
            seq_in_x, x_count, pos_x = [], 1, []
            seq_in_y, y_count, pos_y = [], 1, []

            for w in seq_in:
                if w < self.n_item_x:
                    seq_in_x.append(w)
                    pos_x.append(x_count)
                    x_count += 1
                    seq_in_y.append(self.idx_pad)
                    pos_y.append(0)

                else:
                    seq_in_x.append(self.idx_pad)
                    pos_x.append(0)
                    seq_in_y.append(w)
                    pos_y.append(y_count)
                    y_count += 1

            if len_in < self.len_max:
                pos = [0] * (self.len_max - len_in) + pos
                pos_x = [0] * (self.len_max - len_in) + pos_x
                pos_y = [0] * (self.len_max - len_in) + pos_y

                seq_in_x = [self.idx_pad] * (self.len_max - len_in) + seq_in_x
                seq_in_y = [self.idx_pad] * (self.len_max - len_in) + seq_in_y
                seq_in = [self.idx_pad]*(self.len_max - len_in) + seq_in

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
                processed.append([seq_in, seq_in_x, seq_in_y, pos, pos_x, pos_y, gt_x, gt_y, XorY, gt - self.n_item_x, idx_neg])

            else:  # XorY == 0
                idx_neg = random.sample(list(range(0, gt)) +
                                        list(range(gt + 1, self.n_item_x - 1)), self.n_neg_sample)
                processed.append([seq_in, seq_in_x, seq_in_y, pos, pos_x, pos_y, gt_x, gt_y, XorY, gt, idx_neg])

        return processed

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        """ Get a batch with index. """
        data = self.data[index]
        return tuple((map(lambda x: torch.LongTensor(x).to(self.device), data)))


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
    valloader = DataLoader(CDSRDataset(args, mode='val'), batch_size=args.batch_size_eval, shuffle=False,
                           num_workers=args.num_workers)
    testloader = DataLoader(CDSRDataset(args, mode='test'), batch_size=args.batch_size_eval, shuffle=False,
                            num_workers=args.num_workers)

    return trainloader, valloader, testloader
