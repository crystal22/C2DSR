from os.path import join
import codecs
import pickle
import numpy as np
import scipy.sparse as sp

import torch


def normalize(mx):
    """ Row-normalize sparse matrix """
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """ Convert a scipy sparse matrix to a torch sparse tensor. """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def take_second(elem):
    return elem[1]


def preprocess_graph(args, filename):
    train_data = []

    with codecs.open(filename, 'r', encoding='utf-8') as infile:
        for line in infile:
            res = []
            line = line.strip().split('\t')[2:]
            for w in line:
                w = w.split('|')
                res.append((int(w[0]), int(w[1])))
            res.sort(key=take_second)
            res_2 = []
            for r in res:
                res_2.append(r[0])
            train_data.append(res_2)

    VV_edges = []
    VV_edges_single = []

    real_adj_share = {}

    for seq in train_data:
        source = -1
        target = -1
        pre = -1
        for d in seq:
            if d not in real_adj_share:
                real_adj_share[d] = set()
            if d < args.n_item_a:
                if source != -1:
                    if d in real_adj_share[source]:
                        continue
                    else:
                        VV_edges_single.append([source, d])
                source = d

            else :
                if target != -1:
                    if d in real_adj_share[target]:
                        continue
                    else:
                        VV_edges_single.append([target, d])
                target = d

            if pre != -1:
                if d in real_adj_share[pre]:
                    continue
                VV_edges.append([pre, d])
            pre=d

    VV_edges = np.array(VV_edges)
    VV_edges_single = np.array(VV_edges_single)

    adj_share = sp.coo_matrix((np.ones(VV_edges.shape[0]), (VV_edges[:, 0], VV_edges[:, 1])),
                              shape=(args.n_item, args.n_item), dtype=np.float32)
    adj_specific = sp.coo_matrix((np.ones(VV_edges_single.shape[0]), (VV_edges_single[:, 0], VV_edges_single[:, 1])),
                                 shape=(args.n_item, args.n_item), dtype=np.float32)

    adj_share = normalize(adj_share)
    adj_specific = normalize(adj_specific)
    adj_share = sparse_mx_to_torch_sparse_tensor(adj_share)
    adj_specific = sparse_mx_to_torch_sparse_tensor(adj_specific)

    return adj_share, adj_specific


def make_graph(args, filename):
    if args.use_raw:
        adj_share, adj_specific = preprocess_graph(args, filename)
        if args.save_processed:
            with open(join(args.path_data, 'graph.pkl'), 'wb') as f:
                pickle.dump((adj_share, adj_specific), f)
    else:
        with open(join(args.path_data, 'graph.pkl'), 'rb') as f:
            (adj_share, adj_specific) = pickle.load(f)

    return adj_share.to(args.device), adj_specific.to(args.device)
