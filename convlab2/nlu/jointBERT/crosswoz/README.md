# BERTNLU on CrossWOZ

Based on pre-trained bert, BERTNLU use a linear layer for slot tagging and another linear layer for intent classification. Dialog acts are split into two groups, depending on whether the value is in the utterance. 

- For those dialog acts that the value appears in the utterance, they are translated to BIO tags. For example, `"Find me a cheap hotel"`, its dialog act is `{"Hotel-Inform":[["Price", "cheap"]]}`, and translated tag sequence is `["O", "O", "O", "B-Hotel-Inform+Price", "O"]`. An MLP takes bert word embeddings as input and classify the tag label. If you set `context=true` in config file, utterances of last three turn will be concatenated and provide context information with embedding of `[CLS]` for classification.  
- For each of the other dialog acts, such as `(Hotel-Request, Address, ?)`, another MLP takes embeddings of `[CLS]` of current utterance as input and do the binary classification. If you set `context=true` in config file, utterances of last three turn will be concatenated and provide context information with embedding of `[CLS]` for classification.  

We fine-tune BERT parameters on crosswoz.

## Usage

Determine which data you want to use: if **mode**='usr', use user utterances to train; if **mode**='sys', use system utterances to train; if **mode**='all', use both user and system utterances to train.

#### Preprocess data

On `jointBERT/crosswoz` dir:

```sh
$ python preprocess.py [mode]
```

output processed data on `data/[mode]_data/` dir.

#### Train a model

On `jointBERT` dir:

```sh
$ python train.py --config_path crosswoz/configs/[config_file]
```

The model will be saved under `output_dir` of config_file. Also, it will be zipped as `zipped_model_path` in config_file. 

#### Test a model

On `jointBERT` dir:

```sh
$ python test.py --config_path crosswoz/configs/[config_file]
```

The result (`output.json`) will be saved under `output_dir` of config_file. 

#### Predict

See `nlu.py` for usage

#### Trained model

We have trained two models: one use context information (last 3 utterances)(`configs/crosswoz_all_context.json`) and the other doesn't (`configs/crosswoz_all.json`) on **all** utterances of crosswoz dataset (`data/crosswoz/[train|val|test].json.zip`). Performance:

|                 | F1    |
| --------------- | ----- |
| without context | 91.85 |
| with context    | 95.53 |

Models can be download form:

Without context: https://convlab.blob.core.windows.net/convlab-2/bert_crosswoz_all.zip

With context: https://convlab.blob.core.windows.net/convlab-2/bert_crosswoz_all_context.zip



## Data

We use the crosswoz data (`data/crosswoz/[train|val|test].json.zip`).

## References

```
@inproceedings{devlin2019bert,
  title={BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding},
  author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  booktitle={Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)},
  pages={4171--4186},
  year={2019}
}
```