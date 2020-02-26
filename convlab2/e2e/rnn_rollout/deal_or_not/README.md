# RNN Rollout model for Object Division Negotiation Dialog (deal or not)

## The RNN Rollout model is a RNN-based multi-turn seq2seq dialog model
proposed by [Lewis et al., 2017](https://www.aclweb.org/anthology/D17-1259).
We adopted the [original code](https://github.com/facebookresearch/end-to-end-negotiator)
to make it more flexible to be used as a module in our Tatk pipeline
framework.

Note tha you have to follow ```Attribution-NonCommercial 4.0 International```
license when using the code and datasets.


## Data preparation
To use this model, you have to first download the pre-trained models
from [here](s3://tatk-data/rnnrollout_dealornot.zip), and put the *.th
files under ```tatk/e2e/rnn_rullout/deal_or_not/configs```.

## Run the Model
To run the model, you can run this command:
```
python test_deal_or_not.py
```
under ```tests/e2e/rnn_rollout``` directory.

You can also import RNN Rollout model in a pipeline dialog system and
run the entire model to test its performance.

# Performance
The reward of our pretrained model against seq2seq model is 7.2 vs. 6.4.
You can train the model by your self for better performance.