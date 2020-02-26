# SVMNLU on multiwoz

SVMNLU build a classifier for each semantic tuple (intent-slot-value) based on n-gram features. It's first proposed by [Mairesse et al. (2009)](http://mairesse.s3.amazonaws.com/research/papers/icassp09-final.pdf). We adapt the implementation from [pydial](https://bitbucket.org/dialoguesystems/pydial/src/master/semi/CNetTrain/).

- For each semantic tuple (intent-slot-value) that has limited value, such as `(Hotel-Request, Addr, ?)` and `(Hotel-Inform, Internet, [yes|no])`, an SVM classifier is applied directly.
- If the semantic tuple (intent-slot-value) may have many value, such as Hotel-Name and Hotel-Addr, we use an SVM classifier `(INTENT, SLOT, GENERAL)` for all tuples that have same intent and slot but different value. When a new sentence come, we identify these value using the ontology.

## Usage

Determine which data you want to use: if **mode**='usr', use user utterances to train; if **mode**='sys', use system utterances to train; if **mode**='all', use both user and system utterances to train.

#### Preprocess data

On `svm/multiwoz` dir:

```sh
$ python preprocess.py [mode]
```

output processed data on `data/[mode]_data` dir.

#### Train a model

On `svm` dir:

```sh
$ PYTHONPATH=../../.. python train.py multiwoz/configs/multiwoz_[mode].cfg
```

Please refer to `svm/config.md` for how to write config file (`*.cfg`)

The model will be saved on `model/svm_multiwoz_[mode].pickle`. Also, it will be zipped as `model/svm_multiwoz_[mode].zip`. 

Trained models can be download on: 

- Trained on all data: [mode=all](https://tatk-data.s3-ap-northeast-1.amazonaws.com/svm_multiwoz_all.zip)
- Trained on user utterances only: [mode=usr](https://tatk-data.s3-ap-northeast-1.amazonaws.com/svm_multiwoz_usr.zip)
- Trained on system utterances only: [mode=sys](https://tatk-data.s3-ap-northeast-1.amazonaws.com/svm_multiwoz_usr.zip)

#### Evaluate

On `svm/multiwoz` dir:

```sh
$ PYTHONPATH=../../../.. python evaluate.py [mode]
```

#### Predict

In `nlu.py` , the `SVMNLU` class inherits the NLU interface and adapts to multiwoz dataset. Example usage:

```python
from tatk.nlu.svm.multiwoz import SVMNLU

model = SVMNLU(mode, model_file=PATH_TO_ZIPPED_MODEL)
dialog_act = model.predict(utterance)
```

You can refer to `evaluate.py` for specific usage.

## Data

We use the multiwoz data (`data/multiwoz/[train|val|test].json.zip`).

## Performance

`mode` determines the data we use: if mode=`usr`, use user utterances to train; if mode=`sys`, use system utterances to train; if mode=`all`, use both user and system utterances to train.

We evaluate the precision/recall/f1 of predicted dialog act.

| mode | Precision | Recall | F1    |
| ---- | --------- | ------ | ----- |
| usr  | 71.88     | 61.86  | 66.49 |
| sys  | 68.14     | 40.88  | 51.10 |
| all  | 68.16     | 46.98  | 55.62 |



## References

```
@inproceedings{mairesse2009spoken,
  title={Spoken language understanding from unaligned data using discriminative classification models},
  author={Mairesse, Fran{\c{c}}ois and Gasic, Milica and Jurcicek, Filip and Keizer, Simon and Thomson, Blaise and Yu, Kai and Young, Steve},
  booktitle={2009 IEEE International Conference on Acoustics, Speech and Signal Processing},
  pages={4749--4752},
  year={2009},
  organization={IEEE}
}

@InProceedings{ultes2017pydial,
  author    = {Ultes, Stefan  and  Rojas Barahona, Lina M.  and  Su, Pei-Hao  and  Vandyke, David  and  Kim, Dongho  and  Casanueva, I\~{n}igo  and  Budzianowski, Pawe{\l}  and  Mrk\v{s}i\'{c}, Nikola  and  Wen, Tsung-Hsien  and  Gasic, Milica  and  Young, Steve},
  title     = {{PyDial: A Multi-domain Statistical Dialogue System Toolkit}},
  booktitle = {Proceedings of ACL 2017, System Demonstrations},
  month     = {July},
  year      = {2017},
  address   = {Vancouver, Canada},
  publisher = {Association for Computational Linguistics},
  pages     = {73--78},
  url       = {http://aclweb.org/anthology/P17-4013}
}
```

