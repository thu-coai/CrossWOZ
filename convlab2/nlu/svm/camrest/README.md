# SVMNLU on camrest

SVMNLU build a classifier for each semantic tuple (intent-slot-value) based on n-gram features. It's first proposed by [Mairesse et al. (2009)](http://mairesse.s3.amazonaws.com/research/papers/icassp09-final.pdf). We adapt the implementation from [pydial](https://bitbucket.org/dialoguesystems/pydial/src/master/semi/CNetTrain/).

- For each semantic tuple (intent-slot-value) that has limited value, such as `(Hotel-Request, Addr, ?)` and `(Hotel-Inform, Internet, [yes|no])`, an SVM classifier is applied directly.
- If the semantic tuple (intent-slot-value) may have many value, such as Hotel-Name and Hotel-Addr, we use an SVM classifier `(INTENT, SLOT, GENERAL)` for all tuples that have same intent and slot but different value. When a new sentence come, we identify these value using the ontology.

## Usage

Determine which data you want to use: if **mode**='usr', use user utterances to train; if **mode**='sys', use system utterances to train; if **mode**='all', use both user and system utterances to train.

#### Preprocess data

On `svm/camrest` dir:

```sh
$ python preprocess.py [mode]
```

output processed data on `data/[mode]_data` dir.

#### Train a model

On `svm` dir:

```sh
$ PYTHONPATH=../../.. python train.py camrest/configs/camrest_[mode].cfg
```

Please refer to `svm/config.md` for how to write config file (`*.cfg`)

The model will be saved on `model/svm_camrest_[mode].pickle`. Also, it will be zipped as `model/svm_camrest_[mode].zip`. 

Trained models can be download on: 

- Trained on all data: [mode=all](https://tatk-data.s3-ap-northeast-1.amazonaws.com/svm_camrest_all.zip)
- Trained on user utterances only: [mode=usr](https://tatk-data.s3-ap-northeast-1.amazonaws.com/svm_camrest_usr.zip)
- Trained on system utterances only: [mode=sys](https://tatk-data.s3-ap-northeast-1.amazonaws.com/svm_multiwoz_usr.zip)

#### Evaluate

On `svm/camrest` dir:

```sh
$ PYTHONPATH=../../../.. python evaluate.py [mode]
```

#### Predict

In `nlu.py` , the `SVMNLU` class inherits the NLU interface and adapts to camrest dataset. Example usage:

```python
from tatk.nlu.svm.camrest import SVMNLU

model = SVMNLU(mode, model_file=PATH_TO_ZIPPED_MODEL)
dialog_act = model.predict(utterance)
```

You can refer to `evaluate.py` for specific usage.

## Data

We use the multiwoz data (`data/camrest/[train|val|test].json.zip`).

## Performance

`mode` determines the data we use: if mode=`usr`, use user utterances to train; if mode=`sys`, use system utterances to train; if mode=`all`, use both user and system utterances to train.

We evaluate the precision/recall/f1 of predicted dialog act.

| mode | Precision | Recall | F1    |
| ---- | --------- | ------ | ----- |
| usr  | 64.53     | 74.52  | 69.16 |
| sys  | 47.20     | 45.57  | 46.37 |
| all  | 55.95     | 59.03  | 57.45 |

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

@article{wenN2N16,
       Author = {Wen, Tsung-Hsien and Vandyke, David and Mrk{\v{s}}i\'c, Nikola and Ga{\v{s}}i\'c, Milica and M. Rojas-Barahona, Lina and Su, Pei-Hao and Ultes, Stefan and Young, Steve},
       title={A Network-based End-to-End Trainable Task-oriented Dialogue System},
       journal={arXiv preprint: 1604.04562},
       year={2016},
       month={April}
}

@article{wencond16,
       Author = {Wen, Tsung-Hsien and Ga{\v{s}}i\'c, Milica and Mrk{\v{s}}i\'c, Nikola and M. Rojas-Barahona, Lina and Su, Pei-Hao and Ultes, Stefan and Vandyke, David and Young, Steve},
       title={Conditional Generation and Snapshot Learning in Neural Dialogue Systems},
       journal={arXiv preprint: 1606.03352},
       year={2016},
       month={June}
}
```

