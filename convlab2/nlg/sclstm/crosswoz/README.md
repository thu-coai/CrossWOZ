# SCLSTM NLG on Multiwoz_zh

Semantically-conditioned LSTM (SC-LSTM) is an NLG model that generates natural linguistically varied responses based on a deep, semantically controlled LSTM architecture. 

- *Sentence planning* maps input semantic symbols (e.g. dialog acts) into an intermediary form representing the utterance.
- *Surface realization* converts the intermediate structure into the final text.

The code derives from [github](https://github.com/andy194673/nlg-sclstm-multiwoz). We modify it to support user NLG. The original paper can be found at [ACL Anthology](https://aclweb.org/anthology/papers/D/D15/D15-1199/)

## Usage

### Prepare the data

```bash
$ python generate_resources.py
```

This will generate two folders that contain training data: ./resource/\*, ./resource_usr/\*.

### Train

```bash
$ python train.py  --mode=train --model_path=sclstm.pt --n_layer=1 --lr=0.005 > sclstm.log
```

Set *user* to use user NLG，e.g.

```bash
$ python train.py  --mode=train --model_path=sclstm_usr.pt --n_layer=1 --lr=0.005 --user True > sclstm_usr.log
```

### Test

```bash
$ python train.py --mode=test --model_path=sclstm.pt --n_layer=1 --beam_size=10 > sclstm.res
```

### Evaluate

```bash
$ python evaluate.py [usr|sys]
```

## Data

We use CrossWOZ data (`tatk/data/crosswoz`).

## Performance on CrossWOZ

`mode` determines the data we use: if mode=`usr`, use user utterances to train; if mode=`sys`, use system utterances to train.

We evaluate the BLEU4 of delexicalized utterance. The references of a generated sentence are all the golden sentences that have the same dialog act.

| mode  | usr    | sys    |
| ----- | ------ | ------ |
| BLEU4 | 0.7858 | 0.8595 |

## Reference

```
@inproceedings{wen2015semantically,
  title={Semantically Conditioned LSTM-based Natural Language Generation for Spoken Dialogue Systems},
  author={Wen, Tsung-Hsien and Gasic, Milica and Mrk{\v{s}}i{\'c}, Nikola and Su, Pei-Hao and Vandyke, David and Young, Steve},
  booktitle={Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing},
  pages={1711--1721},
  year={2015}
}
```

