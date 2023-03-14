import numpy as np


def cal_metrics(ranks):
    """ calculate metrics hr, mrr, ndcg at k = 5, 20 """
    N = len(ranks)
    hr5, hr20, mrr5, mrr20, ndcg5, ndcg20 = 0., 0., 0., 0., 0., 0.

    for rank in ranks:
        if rank <= 20:
            hr20 += 1
            mrr20 += 1 / rank
            ndcg20 += 1 / np.log2(rank + 1)

            if rank <= 5:
                hr5 += 1
                mrr5 += 1 / rank
                ndcg5 += 1 / np.log2(rank + 1)
    return [x / N for x in (hr5, hr20, mrr5, mrr20, ndcg5, ndcg20)]


def cal_score(ranks_a, ranks_b, benchmark):
    """ calculate improvement rate and metrics for both domains """
    res = cal_metrics(ranks_a) + cal_metrics(ranks_b)

    imp = np.zeros(12)  # total 12 metrics
    for i, (x, y) in enumerate(zip(res, benchmark)):
        imp[i] = x / y - 1

    return [np.mean(imp)] + res
