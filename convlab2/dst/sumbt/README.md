# SUMBT on Multiwoz

SUMBT (Slot-Utterance Matching Belief Tracker) is a belief tracking model that
utilizes semantic similarity between dialogue utterances and slot-values
, which is proposed by [Hwaran Lee et al., 2019](https://www.aclweb.org/anthology/P19-1546.pdf).

The code derives from [github](https://github.com/SKTBrain/SUMBT). We modify it to support user DST. 

## Usage


### Train

from tatk root directory
```bash
$ cd /tatk/dst/sumbt/multiwoz
$ python sumbt.py --train
```


### Test

```bash
$ cd /tatk/dst/sumbt/multiwoz
$ python sumbt.py --test
```

### Evaluate

```bash
$ cd /tatk/dst/sumbt/multiwoz
$ python sumbt.py --dev
```

## Data

We use the multiwoz data (./resource/\*, ./resource_usr/\*).

## Performance on Multiwoz

`mode` determines the data we use: if mode=`usr`, use user utterances to train; if mode=`sys`, use system utterances to train.

We evaluate the Joint accuracy and Slot accuracy on Multiwoz 2.0 validation and test set. 
The accuracy on validation set are slightly higher than the results reported in the paper,
because in the evaluation code all undefined values in ontology are set `none` but predictions 
will always be wrong for all undefined domain-slots.  

|   | Joint acc  | Slot acc    | Joint acc (Restaurant)  |  Slot acc (Restaurant)|
| ----- | ----- | ------ | ------ | ----    |
| dev     | 0.48 | 0.97 | 0.83 | 0.96  |
| test     | 0.49 | 0.97 | 0.82 | 0.96  |

## Model Structure

SUMBT considers a domain-slot type (e.g., 'restaurant-food') as a query and finds the corresponding 
slot-value in a pair of system-user utterances, under the assumption that the answer appear in the utterances.

The model encodes domain-slot with a fixed BERT model and encodes utterances with another BERT 
of which parameters are fine-tuned during training. A MultiHead attention layer is
employed to capture slot-specific information, and the attention context vector is fed
into an RNN to model the flow of dialogues.


## Reference

```
@inproceedings{lee2019sumbt,
  title={SUMBT: Slot-Utterance Matching for Universal and Scalable Belief Tracking},
  author={Lee, Hwaran and Lee, Jinsik and Kim, Tae-Yoon},
  booktitle={Proceedings of the 57th Conference of the Association for Computational Linguistics},
  pages={5478--5483},
  year={2019}
}
```

