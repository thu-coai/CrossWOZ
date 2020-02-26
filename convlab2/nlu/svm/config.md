Confusion Network Config Options                 {#CNetConfig}
================================

## [grammar]
* `acts` - a JSON list of all the possible act-types
* `nonempty_acts` - a JSON list of those acts that require a slot-value pair
* `ontology` - a JSON file with the ontology for the domain, this is found in a belief tracking corpus
* `slots_enumerated` - a JSON list of slots for which we do not tag values with <generic_value>


## [classifier]
* `type` - the type of classifier used. {*svm*}
* `features` - a JSON list of the features extracted from a turn. {"cnet", "valueIdentifying","nbest","lastSys","nbestLengths","nbestScores"}. These refer to classes in `Features.py`.
* `max_ngram_length` - the maximum length of ngrams to extract. Default is 3.
* `max_ngrams` - the maximum number of ngrams to extract per turn. Default is 200.
* `skip_ngrams` - Whether to use skip ngrams, for nbest features {"True","False"}
* `skip_ngram_decay` - Factor to discount skips by. Default is 0.9.
* `min_examples` - the minimum number of positive examples of a tuple we require to train a classifier for it. Default is 10.


## [train]
* `output` - the pickle file where the learnt classifier is saved
* `dataListFile` - the file which contains all session name, one name each line
* `dataroot` - the directory where the data is found
* `log_input_key` - the key to use of `log_turn["input"]`. Default is 'batch'.

## [decode]
* `output` - the JSON file that decoder output is saved to. 
* `dataListFile` - the file which contains all session name, one name each line
* `dataroot` - the directory where the data is found
