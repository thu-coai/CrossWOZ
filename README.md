# CrossWOZ

Data and codes for the paper "CrossWOZ: A Large-Scale Chinese Cross-Domain Task-Oriented Dialogue Dataset"

please install ConvLab-2 first:

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
- TemplateNLG: `convlab/nlg/template/crosswoz`
- Simulator: `convlab/policy/rule/crosswoz`