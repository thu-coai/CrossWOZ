# MDRG word policy for Multiwoz

MDRG word policy for Multiwoz dataset, can be trained or tested on Multiwoz dataset.

## Data preparation

Before training or testing, you need to download training data and database files. Please run:
```python
python auto_download.py
```

After you've got `/data` `/db` `/model` folders, you can start training your MDRG policy by:
```python
python train.py
```

After you've trained your own policy, you can use it in your own dialog system:

```python
from tatk.policy.mdrg.multiwoz.policy import MDRGWordPolicy
policy = MDRGWordPolicy(num)
```
You can choose different model parameter by changing num, default set to `num=1`, which uses parameter from `model/model/translate.ckpt-1`

MDRG policy takes dialog state as input and generates an utterance.
