
# Deal or Not
Deal or not is a bilateral object-division negotiation dialog dataset ([Lewis et al., 2017](https://www.aclweb.org/anthology/D17-1259)). In this dataset, the two players are supposed to interact with each other using natural language utterances to divide some object. Note that each type of objects may worth different values for each agent. The target of each agent is to get as many values as possible.

To make the code more flexible to be used as a module in our Tatk framework, we adopted the [original code](https://github.com/facebookresearch/end-to-end-negotiator).

## Getting Started
In this section, will demonstrate how we implemented the code and how to use it.
### Interface
The model of Deal or Not (```DealornotAgent```) is defined in ```tatk/e2e/rnn_rollout/deal_or_not/model.py``` by inheriting the ```RNNRolloutAgent``` defined in ```tatk/e2e/rnn_rollout/rnnrollout.py```. in the ```__init__()``` method, the data and pretrained model of Deal or Not dataset is loaded and feed into ```RNNRolloutAgent```.


```python
class DealornotAgent(RNNRolloutAgent):
    """The Rnn Rollout model for DealorNot dataset."""
    def __init__(self, name, args, sel_args, train=False, diverse=False, max_total_len=100):
        self.config_path = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'configs')
        self.data_path = os.path.join(get_root_path(), args.data)
        domain = get_domain(args.domain)
        corpus = RnnModel.corpus_ty(domain, self.data_path, freq_cutoff=args.unk_threshold, verbose=True,
                                    sep_sel=args.sep_sel)

        model = RnnModel(corpus.word_dict, corpus.item_dict_old,
                         corpus.context_dict, corpus.count_dict, args)
        state_dict = utils.load_model(os.path.join(self.config_path, args.model_file))  # RnnModel
        model.load_state_dict(state_dict)

        sel_model = SelectionModel(corpus.word_dict, corpus.item_dict_old,
                                   corpus.context_dict, corpus.count_dict, sel_args)
        sel_state_dict = utils.load_model(os.path.join(self.config_path, sel_args.selection_model_file))
        sel_model.load_state_dict(sel_state_dict)

        super(DealornotAgent, self).__init__(model, sel_model, args, name, train, diverse, max_total_len)
        self.vis = args.visual
```

### Method
The core method of ```DealornotAgent``` is ```response()```, which takes as input a user utterance (str) and returns the natural utterance response of system.

## Run the Code
If you run the code of ```DealornotAgent```, you can run ```tests/e2e/rnn_rollout/test_deal_or_not.py``` under its path. In this script, we used the framework of Tatk, including ```Session``` and ```Agent```. The example model and experimental parameters are also defined. The responses utterance of both agents will be displayed.
