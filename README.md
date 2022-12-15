# C2DSR-pytorch

## Current construction

This is an unofficial Pytorch implementation of CIKM'22 paper [Contrastive Cross-Domain Sequential Recommendation](https://dl.acm.org/doi/abs/10.1145/3511808.3557262):

Current under construction.

The original PyTorch implementation is [here](https://github.com/cjx96/C2DSR).


## Dataset

Paper author find the information leak issue from existed datasets which released by previous works  (pi-net and MIFN), and their corrected versions are provided [here](https://drive.google.com/drive/folders/1xpnp6tH56xz8PF_xuTi9exEptmcvlAVU?usp=sharing).

Three processed datasets are provided: Amazon - Food & Kitchen, Amazon - Movie & Book and HVIDEO - Entertainment & Education.

I also zip all processed data, see more in `./dataloader/dataset.zip`. Please upzip it at its current folder before use.


## Experiments



## Usage

```shell
python main.py
```
