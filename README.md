# C2DSR-pytorch

This is an ***unofficial*** Pytorch implementation of CIKM'22 paper [Contrastive Cross-Domain Sequential Recommendation](https://dl.acm.org/doi/abs/10.1145/3511808.3557262).

The ***official*** PyTorch implementation by paper's authors is [here](https://github.com/cjx96/C2DSR).


## Dataset

Paper's authors find the information leak issue from existed datasets which released by previous works  (pi-net and MIFN), and their corrected versions are provided [here](https://drive.google.com/drive/folders/1xpnp6tH56xz8PF_xuTi9exEptmcvlAVU?usp=sharing).

Three processed datasets are provided: Amazon - Food & Kitchen, Amazon - Movie & Book and HVIDEO - Entertainment & Education.

Zipped datasets are also provided in this repo, see more in `./data/raw.zip`. Please upzip before use.


## Differences

- Official repo calculate metrics based on *pre-rank* standard in industry procedures, only sampling 999 negative items instead of using the entire itemset. On the contrary, this repo follow the common research standard for a fair comparison, which is called *rank* in industry procedures. FYI, common industry procedures are *recall* -> *pre-rank* -> *rank* (-> *re-rank*).

- I replaced some original metrics with new ones. Now this code will compute Recall, MRR and NDCG under K = {5, 20}.

## Results

Model:
- C2DSR: proposed results in the paper
- w/ 999: use the official code and its default hyperparameters, and use only 999 out of all candidate items to compute metrics
- w/o 999: use the official code and its default hyperparameters, and use all candidate items to compute metrics
- ours: use our code and re-tune hyperparameters, and use all candidate items to compute metrics

***To be noticed***: 
- The best results for w/ or w/o 999 on both domains are chosen separately, which means they may not occur in the same epoch.
- Our model selected the best result on both domains from same epoch outputs. Since our model uses different metrics, we only pick some of them for illustration.

<!--
|  Model  |  Data   |   A    |        |         |         |        |        |        |        |   B    |        |         |         |        |        |        |        |
|:-------:|:-------:|:------:|:------:|:-------:|:-------:|:------:|:------:|:------:|:------:|:------:|:------:|:-------:|:-------:|:------:|:------:|:------:|:------:|
|         |         |  mrr   | ndcg@5 | ndcg@10 | ndcg@20 |  hr@1  |  hr@5  | hr@10  | hr@20  |  mrr   | ndcg@5 | ndcg@10 | ndcg@20 |  hr@1  |  hr@5  | hr@10  | hr@20  |    
|  C2DSR  | Foo-Kit | 0.0891 | 0.0865 | 0.0971  |         | 0.0584 | 0.1124 | 0.1454 |        | 0.0465 | 0.0416 | 0.0494  |         | 0.0251 | 0.0574 | 0.0818 |        |
| w/ 999  | Foo-Kit | 0.0931 | 0.0911 | 0.0996  |         | 0.0622 | 0.0996 | 0.1427 |        | 0.0407 | 0.0357 | 0.0436  |         | 0.0199 | 0.0514 | 0.0761 |        |
| w/o 999 | Foo-Kit | 0.0254 | 0.0243 | 0.0282  |         | 0.0160 | 0.0323 | 0.0444 |        | 0.0090 | 0.0083 | 0.0093  |         | 0.0061 | 0.0104 | 0.0136 |        |
|  ours   | Foo-Kit |        | 0.0655 |         | 0.0818  |        | 0.0841 |        | 0.1420 |        | 0.0146 | 0.0202  |         |        | 0.0198 |        | 0.0202 |
-->

![](img\data.png)


## Usage

```shell
python main.py 
```
