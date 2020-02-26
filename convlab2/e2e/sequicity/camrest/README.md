# Sequicity on camrest

Sequicity is an end-to-end task-oriented dialog system based on a single sequence-to-sequence model that uses *belief span* to track dialog believes. 

- It formulates the DST task into a generation problem (rather than the classification problem)
- It tracks the dialog history using *two-state copynet* mechanism, one for *belief span*, the other for response generation.

We adapt the code from [github](https://github.com/WING-NUS/sequicity) to work in camrest corpus. The original paper can be found at [ACL Anthology](https://aclweb.org/anthology/papers/P/P18/P18-1133).

## Usage

### Prepare data

Download [data](https://tatk-data.s3-ap-northeast-1.amazonaws.com/sequicity_camrest_data.zip) and unzip here.

### Training with default parameters

On `sequicity` dir:

```bash
$ python model.py -mode train -model camrest -cfg camrest/configs/camrest.json
$ python model.py -mode adjust -model camrest -cfg camrest/configs/camrest.json
```

### Testing

```bash
$ python model.py -mode test -model camrest -cfg camrest/configs/camrest.json
```

### Reinforcement fine-tuning

```bash
$ python model.py -mode rl -model camrest -cfg camrest/configs/camrest.json
```

### Trained model

Trained model can be download on [here](https://tatk-data.s3-ap-northeast-1.amazonaws.com/sequicity_camrest.pkl). Place it under `output` dir.

### Predict

```python
from tatk.e2e.sequicity.camrest import Sequicity

s = Sequicity(model_file=MODEL_PATH_OR_URL)
s.response("I want to find a cheap restaurant")
```

## Data

[Camrest](https://www.repository.cam.ac.uk/handle/1810/260970)

## Performance

- **BLEU4**
- **Match rate** : determines if a system can generate all correct constraints (belief span) to search the indicated entities of the user
- **Success F1**: F1 score of requested slots answered in the current dialogue

In terms of `success F1`,  Sequicity by order shows the (F1, Precision, Recall) score.

| BLEU | Match | Success (F1, Prec., Rec.) |
| - | - | - |
| 0.2160 | 0.9273 |(0.8365, 0.8707, 0.8049)|

## Reference

   ```
@inproceedings{lei2018sequicity,
	title={Sequicity: Simplifying Task-oriented Dialogue Systems with Single Sequence-to-Sequence Architectures},
	author={Lei, Wenqiang and Jin, Xisen and Ren, Zhaochun and He, Xiangnan and Kan, Min-Yen and Yin, Dawei},
	booktitle={ACL},
	year={2018}
}
   ```