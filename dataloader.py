from os.path import join
import random
import pickle
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader


class CDSRDataset(Dataset):
    def __init__(self, args, mode):
        self.mode = mode
        self.path_raw = args.path_raw
        self.device = args.device
        self.batch_size = args.batch_size
        self.dataset = args.dataset

        self.idx_pad = args.idx_pad
        self.n_item_a = args.n_item_a
        self.n_item_b = args.n_item_b
        self.n_neg_sample = args.n_neg_sample

        self.len_max = args.len_max

        if args.use_raw:
            if mode == 'train':
                self.data = self.preprocess_train()
            else:
                self.data = self.preprocess_evaluate()
            with open(join(args.path_data, mode + '.pkl'), 'wb') as f:
                pickle.dump(self.data, f)
            print('Processed ' + mode + ' data saved.')

        else:
            with open(join(args.path_data, mode + '.pkl'), 'rb') as f:
                self.data = pickle.load(f)

        self.length = len(self.data)

    def read_raw(self):

        def takeSecond(elem):
            return elem[1]

        with open(join(self.path_raw, self.mode + '_new.txt'), 'r', encoding='utf-8') as f:
            data = []
            for line in f:
                seq_ui = []
                line = line.strip().split('\t')[2:]
                for ui in line:
                    ui = ui.split('|')
                    seq_ui.append((int(ui[0]), int(ui[1])))
                seq_ui.sort(key=takeSecond)
                seq_i = []
                for ui in seq_ui:
                    seq_i.append(ui[0])

                data.append(seq_i)
        return data

    def preprocess_train(self):
        """ Preprocess the raw sequence data for training. """
        data = self.read_raw()
        processed = []
        for u_behavior in tqdm(data):
            gt = u_behavior[1:]
            seq_share = u_behavior[:-1]
            len_seq = len(u_behavior)

            pos = list(range(1, len_seq))

            # split shared seq into domain-specific sequences, which still use indices from the shared-domain.
            x_count, seq_share_a, pos_a, seq_share_neg_a = 1, [], [], []
            y_count, seq_share_b, pos_b, seq_share_neg_b = 1, [], [], []
            for i, idx in enumerate(seq_share):
                if idx < self.n_item_a:
                    seq_share_neg_a.append(idx)
                    seq_share_a.append(idx)
                    pos_a.append(x_count)
                    x_count += 1
                    seq_share_neg_b.append(random.randint(0, self.n_item_a - 1))
                    seq_share_b.append(self.idx_pad)
                    pos_b.append(0)

                else:
                    seq_share_neg_a.append(random.randint(self.n_item_a, self.idx_pad - 1))
                    seq_share_a.append(self.idx_pad)
                    pos_a.append(0)
                    seq_share_neg_b.append(idx)
                    seq_share_b.append(idx)
                    pos_b.append(y_count)
                    y_count += 1

            # formulate domain-specific sequences, pad for x and y are different with its own length of item set
            gt_a, gt_b = [self.n_item_a] * len(seq_share_a), [self.n_item_b] * len(seq_share_b)
            gt_mask_a, gt_mask_b = [0] * len(seq_share_a), [0] * len(seq_share_b)

            cur = -1
            for i in range(1, len_seq):
                # find last non-pad item in x-domain sequence
                if pos_a[-i]:
                    if cur == -1:
                        cur = seq_share_a[-i]
                        # fetch ground truth for domain-specific seq and pad it in inputs
                        if gt[-1] < self.n_item_a:
                            gt_a[-i] = gt[-1]
                            gt_mask_a[-i] = 1
                        else:
                            seq_share_a[-i] = self.idx_pad
                            pos_a[-i] = 0
                    else:
                        # save the fetched 'cur' as the step ground truth
                        gt_a[-i] = cur
                        gt_mask_a[-i] = 1
                        cur = seq_share_a[-i]
            if sum(gt_mask_a) == 0:
                continue

            cur = -1
            for i in range(1, len_seq):
                if pos_b[-i]:
                    if cur == -1:
                        cur = seq_share_b[-i] - self.n_item_a
                        if gt[-1] > self.n_item_a:
                            gt_b[-i] = gt[-1] - self.n_item_a
                            gt_mask_b[-i] = 1
                        else:
                            seq_share_b[-i] = self.idx_pad
                            pos_b[-i] = 0
                    else:
                        gt_b[-i] = cur
                        gt_mask_b[-i] = 1
                        cur = seq_share_b[-i] - self.n_item_a
            if sum(gt_mask_b) == 0:
                continue

            # pad sequence
            len_pad = self.len_max - len_seq + 1

            seq_share = [self.idx_pad] * len_pad + seq_share
            seq_share_a = [self.idx_pad] * len_pad + seq_share_a
            seq_share_b = [self.idx_pad] * len_pad + seq_share_b
            seq_share_neg_a = [self.idx_pad] * len_pad + seq_share_neg_a
            seq_share_neg_b = [self.idx_pad] * len_pad + seq_share_neg_b

            pos = [0] * len_pad + pos
            pos_a = [0] * len_pad + pos_a
            pos_b = [0] * len_pad + pos_b

            # generate ground truth
            gt = [self.idx_pad] * len_pad + gt
            gt_mask_a = [0] * len_pad + gt_mask_a
            gt_mask_b = [0] * len_pad + gt_mask_b

            gt_share_a = [idx if idx < self.n_item_a else self.n_item_a for idx in gt]
            gt_share_b = [idx - self.n_item_a if idx >= self.n_item_a else self.n_item_b for idx in gt]
            gt_a = [self.n_item_a] * len_pad + gt_a
            gt_b = [self.n_item_b] * len_pad + gt_b

            processed.append([seq_share, seq_share_a, seq_share_b, pos, pos_a, pos_b, gt_share_a, gt_share_b,
                              gt_a, gt_b, gt_mask_a, gt_mask_b, seq_share_neg_a, seq_share_neg_b])
        return processed

    def preprocess_evaluate(self):
        """ Preprocess the raw sequence data for evaluation including validation and testing. """
        data = self.read_raw()
        processed = []
        for u_behavior in data:
            gt = u_behavior[1:]
            gt_last = gt[-1]
            seq_share = u_behavior[:-1]
            len_seq = len(u_behavior)

            pos = list(range(1, len_seq))

            x_count, seq_share_a, pos_a = 1, [], []
            y_count, seq_share_b, pos_b = 1, [], []

            for i, idx in enumerate(seq_share):
                if idx < self.n_item_a:
                    seq_share_a.append(idx)
                    pos_a.append(x_count)
                    x_count += 1
                    seq_share_b.append(self.idx_pad)
                    pos_b.append(0)

                else:
                    seq_share_a.append(self.idx_pad)
                    pos_a.append(0)
                    seq_share_b.append(idx)
                    pos_b.append(y_count)
                    y_count += 1

            # pad sequence
            len_pad = self.len_max - len_seq + 1
            pos = [0] * len_pad + pos
            pos_a = [0] * len_pad + pos_a
            pos_b = [0] * len_pad + pos_b

            seq_share = [self.idx_pad] * len_pad + seq_share
            seq_share_a = [self.idx_pad] * len_pad + seq_share_a
            seq_share_b = [self.idx_pad] * len_pad + seq_share_b

            # find last available item as ground truth
            idx_last_a, idx_last_b = -1, -1
            for i in range(1, self.len_max+1):
                if pos_a[-i]:
                    idx_last_a = self.len_max - i
                    break

            for i in range(1, self.len_max+1):
                if pos_b[-i]:
                    idx_last_b = self.len_max - i
                    break

            if gt_last < self.n_item_a:
                list_neg = random.sample(list(range(gt_last)) + list(range(gt_last + 1, self.n_item_a)),
                                         self.n_neg_sample)
                processed.append([seq_share, seq_share_a, seq_share_b, pos, pos_a, pos_b, [idx_last_a], [idx_last_b],
                                  [0], [gt_last], list_neg])

            else:
                list_neg = random.sample(list(range(gt_last - self.n_item_a)) +
                                         list(range(gt_last - self.n_item_a + 1, self.n_item_b - self.n_item_a)),
                                         self.n_neg_sample)
                processed.append([seq_share, seq_share_a, seq_share_b, pos, pos_a, pos_b, [idx_last_a], [idx_last_b],
                                  [1], [gt_last - self.n_item_a], list_neg])

        return processed

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return tuple(torch.LongTensor(x) for x in self.data[index])


def count_item(path):
    count = 0
    with open(path, 'r', encoding='utf-8') as f:
        for _ in f:
            count += 1
    return count


def get_dataloader(args):

    p = args.path_raw if args.use_raw else args.path_data

    args.n_item_a = count_item(join(p, 'items_a.txt'))
    args.n_item_b = count_item(join(p, 'items_b.txt'))
    args.n_item = args.n_item_a + args.n_item_b + 1  # with padding
    args.idx_pad = args.n_item - 1

    trainloader = DataLoader(CDSRDataset(args, mode='train'), batch_size=args.batch_size, shuffle=True,
                             num_workers=args.num_workers)
    valloader = DataLoader(CDSRDataset(args, mode='val'), batch_size=args.batch_size_eval, shuffle=False,
                           num_workers=args.num_workers)
    testloader = DataLoader(CDSRDataset(args, mode='test'), batch_size=args.batch_size_eval, shuffle=False,
                            num_workers=args.num_workers)

    # zip dataset that exceeds maximum file size for GitHub pushing
    if args.dataset in ['Entertainment-Education'] and args.zip_ee:
        import os
        import zipfile
        with zipfile.ZipFile(join(args.path_data, f'{args.dataset}.zip'), 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(args.path_data):
                for file in files:
                    zipf.write(join(root, file),
                               os.path.relpath(join(root, file), join(args.path_data, '..')))
        print(f'Zipped {args.dataset} folder.')

    return trainloader, valloader, testloader
