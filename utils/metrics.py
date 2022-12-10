import numpy as np


def cal_score(predictions):
    MRR = 0.0
    HR_1 = 0.0
    HR_5 = 0.0
    HR_10 = 0.0
    NDCG_5 = 0.0
    NDCG_10 = 0.0
    valid_entity = 0.0
    for pred in predictions:
        valid_entity += 1
        MRR += 1 / pred
        if pred <= 1:
            HR_1 += 1
        if pred <= 5:
            NDCG_5 += 1 / np.log2(pred + 1)
            HR_5 += 1
        if pred <= 10:
            NDCG_10 += 1 / np.log2(pred + 1)
            HR_10 += 1
        if valid_entity % 100 == 0:
            print('.', end='')
    return [MRR / valid_entity, NDCG_5 / valid_entity, NDCG_10 / valid_entity,
            HR_1 / valid_entity, HR_5 / valid_entity, HR_10 / valid_entity]
