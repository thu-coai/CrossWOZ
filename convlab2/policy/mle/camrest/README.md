# Imitation on camrest

Vanilla MLE Policy employs a multi-class classification via Imitation Learning with a set of compositional actions where a compositional action consists of a set of dialog act items.

## Train

```
python train.py
```

You can modify *config.json* to change the setting.

## Data

data/camrest/[train/val/test].json

## Performance

|Dialog act accuracy|
|-|
|0.7459|

