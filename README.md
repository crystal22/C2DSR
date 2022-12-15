# C2DSR-pytorch

This is an unofficial Pytorch implementation of CIKM'22 paper [Contrastive Cross-Domain Sequential Recommendation](https://dl.acm.org/doi/abs/10.1145/3511808.3557262):

The official PyTorch implementation by paper author is [here](https://github.com/cjx96/C2DSR).


## Dataset

Paper author find the information leak issue from existed datasets which released by previous works  (pi-net and MIFN), and their corrected versions are provided [here](https://drive.google.com/drive/folders/1xpnp6tH56xz8PF_xuTi9exEptmcvlAVU?usp=sharing).

Three processed datasets are provided: Amazon - Food & Kitchen, Amazon - Movie & Book and HVIDEO - Entertainment & Education.

I also zip dataset exceed github file size limit, see more in `./data/`. Please upzip before use.


## ***Known issue***

I have re-written the entire project under my framework **without changing logics and operations** but some improving on efficiency and grammar.
During doing so, some issues are found:

- The result metrics are calculated based on samples negative list in `dataloader.py`. This list samples 999 negative items instead of using the whole set.
In other words, the result presented by this code will be better than the reality.

## Usage

```shell
python main.py 
```
