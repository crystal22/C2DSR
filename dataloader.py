from os.path import join
import random
import codecs
import pickle
import copy
import torch


class DataLoader(object):
    """
    Load data from json files, preprocess and prepare batches.
    """
    def __init__(self, args, mode):
        self.mode = mode
        self.batch_size = args.batch_size
        self.dataset = args.dataset
        self.n_item_x = args.n_item_x
        self.n_item_y = args.n_item_y
        self.len_max = args.len_max

        if args.raw_data:
            if mode == 'train':
                self.data = self.read_raw(join(args.path_raw, mode + '_new.txt'), is_train=True)
                data = self.preprocess_train()
            else:
                self.data = self.read_raw(join(args.path_raw, mode + '_new.txt'), is_train=False)
                data = self.preprocess_evaluate()
            if args.save_processed:
                with open(join(args.path_data, mode + '.pkl'), 'wb') as f:
                    pickle.dump(data, f)
        else:
            with open(join(args.path_data, mode + '.pkl'), 'rb') as f:
                data = pickle.load(f)

        if self.mode == 'train':
            data = shuffle(data, self.batch_size)
        else:
            assert self.mode in ['val', 'test']
            self.batch_size = 2048

        self.data = [data[i:i + self.batch_size] for i in range(0, len(data), self.batch_size)]

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

    def preprocess_train(self):
        """ Preprocess the data and convert to ids. """
        processed = []
        for d in self.data:  # the pad is needed! but to be careful.

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
            for id in range(len(xd)):
                id+=1
                if pos_x[-id]:
                    if now == -1:
                        now = xd[-id]
                        if gt[-1] < self.n_item_x:
                            x_gt[-id] = gt[-1]
                            x_gt_mask[-id] = 1
                        else:
                            xd[-id] = self.n_item_x + self.n_item_y
                            pos_x[-id] = 0
                    else:
                        x_gt[-id] = now
                        x_gt_mask[-id] = 1
                        now = xd[-id]
            if sum(x_gt_mask) == 0:
                print('pass sequence x')
                continue

            now = -1
            y_gt = [self.n_item_y] * len(yd) # caution!
            y_gt_mask = [0] * len(yd)
            for id in range(len(yd)):
                id+=1
                if pos_y[-id]:
                    if now == -1:
                        now = yd[-id] - self.n_item_x
                        if gt[-1] > self.n_item_x:
                            y_gt[-id] = gt[-1] - self.n_item_x
                            y_gt_mask[-id] = 1
                        else:
                            yd[-id] = self.n_item_x + self.n_item_y
                            pos_y[-id] = 0
                    else:
                        y_gt[-id] = now
                        y_gt_mask[-id] = 1
                        now = yd[-id] - self.n_item_x
            if sum(y_gt_mask) == 0:
                print('pass sequence y')
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
            else:
                print('pass')

            processed.append([d, xd, yd, pos, pos_x, pos_y, gt, share_x_gt, share_y_gt, x_gt, y_gt, gt_mask, share_x_gt_mask, share_y_gt_mask, x_gt_mask, y_gt_mask, x_corrupt, y_corrupt])
        return processed

    def preprocess_evaluate(self):
        processed = []
        for d in self.data:  # the pad is needed! but to be careful.
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
            for id in range(len(pos_x)):
                id += 1
                if pos_x[-id]:
                    x_last = -id
                    break

            y_last = -1
            for id in range(len(pos_y)):
                id += 1
                if pos_y[-id]:
                    y_last = -id
                    break

            negative_sample = []
            for i in range(999):
                while True:
                    if d[1]:  # in Y domain, the validation/test negative samples
                        sample = random.randint(0, self.n_item_y - 1)
                        if sample != d[2] - self.n_item_x:
                            negative_sample.append(sample)
                            break
                    else : # in X domain, the validation/test negative samples
                        sample = random.randint(0, self.n_item_x - 1)
                        if sample != d[2]:
                            negative_sample.append(sample)
                            break

            if d[1]:
                processed.append([seq, xd, yd, pos, pos_x, pos_y, x_last, y_last, d[1], d[2]-self.n_item_x, negative_sample])
            else:
                processed.append([seq, xd, yd, pos, pos_x, pos_y, x_last, y_last, d[1],
                                  d[2], negative_sample])
        return processed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
        batch_size = len(batch)

        if self.mode != 'train':
            batch = list(zip(*batch))
            return (torch.LongTensor(batch[0]), torch.LongTensor(batch[1]), torch.LongTensor(batch[2]),
                    torch.LongTensor(batch[3]), torch.LongTensor(batch[4]), torch.LongTensor(batch[5]),
                    torch.LongTensor(batch[6]), torch.LongTensor(batch[7]), torch.LongTensor(batch[8]),
                    torch.LongTensor(batch[9]), torch.LongTensor(batch[10]))
        else:
            batch = list(zip(*batch))

            return (torch.LongTensor(batch[0]), torch.LongTensor(batch[1]), torch.LongTensor(batch[2]),
                    torch.LongTensor(batch[3]), torch.LongTensor(batch[4]), torch.LongTensor(batch[5]),
                    torch.LongTensor(batch[6]), torch.LongTensor(batch[7]), torch.LongTensor(batch[8]),
                    torch.LongTensor(batch[9]), torch.LongTensor(batch[10]), torch.LongTensor(batch[11]),
                    torch.LongTensor(batch[12]), torch.LongTensor(batch[13]), torch.LongTensor(batch[14]),
                    torch.LongTensor(batch[15]), torch.LongTensor(batch[16]), torch.LongTensor(batch[17]))

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)


def count_item(path):
    count = 0
    with codecs.open(path, 'r', encoding='utf-8') as f:
        for _ in f:
            count += 1
    return count


def shuffle(data, batch_size):
    indices = list(range(len(data)))
    random.shuffle(indices)
    data = [data[i] for i in indices]

    if batch_size > len(data):
        batch_size = len(data)
        batch_size = batch_size

    if len(data) % batch_size != 0:
        data += data[:batch_size]

    return data[: (len(data) // batch_size) * batch_size]


def get_dataloader(args):

    args.n_item_x = count_item(join(args.path_raw, 'Alist.txt'))
    args.n_item_y = count_item(join(args.path_raw, 'Blist.txt'))
    args.n_item = args.n_item_x + args.n_item_y + 1  # with padding
    args.idx_pad = args.n_item - 1

    trainloader = DataLoader(args, mode='train')
    valloader = DataLoader(args, mode='val')
    testloader = DataLoader(args, mode='test')

    return trainloader, valloader, testloader
