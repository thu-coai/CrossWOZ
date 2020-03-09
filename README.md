# CrossWOZ

Data and codes for the paper ["CrossWOZ: A Large-Scale Chinese Cross-Domain Task-Oriented Dialogue Dataset"](https://arxiv.org/abs/2002.11893) (accepted by TACL)

please install ConvLab-2 first (note that this is a minimum version for CrossWOZ and ConvLab-2 will be officially released soon):

```
pip install -e .
```

Data: `data/crosswoz`

Code:

- BERTNLU: `convlab2/nlu/jointBERT/crosswoz`
- RuleDST: `convlab2/dst/rule/crosswoz`
- TRADE: `convlab2/dst/trade/crosswoz`
- SL policy: `convlab2/policy/mle/crosswoz`
- SCLSTM: `convlab2/nlg/sclstm/crosswoz`
- TemplateNLG: `convlab2/nlg/template/crosswoz`
- Simulator: `convlab2/policy/rule/crosswoz`

## Citing
Please kindly cite our paper if this paper and the dataset are helpful.
```
@article{zhu2020crosswoz,
  author = {Qi Zhu and Kaili Huang and Zheng Zhang and Xiaoyan Zhu and Minlie Huang},
  title = {Cross{WOZ}: A Large-Scale Chinese Cross-Domain Task-Oriented Dialogue Dataset},
  journal = {Transactions of the Association for Computational Linguistics},
  year = {2020}
}
```
