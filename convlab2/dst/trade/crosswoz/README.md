## TRADE Multi-Domain and Unseen-Domain Dialogue State Tracking
<img src="plot/pytorch-logo-dark.png" width="10%"> [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

<img align="right" src="plot/einstein-scroll.png" width="8%">
<img align="right" src="plot/salesforce-research.jpg" width="18%">
<img align="right" src="plot/HKUST.jpg" width="12%">

This is the PyTorch implementation of the paper:
**Transferable Multi-Domain State Generator for Task-Oriented Dialogue Systems**. [**Chien-Sheng Wu**](https://jasonwu0731.github.io/), Andrea Madotto, Ehsan Hosseini-Asl, Caiming Xiong, Richard Socher and Pascale Fung. ***ACL 2019***. 
[[PDF]](https://arxiv.org/abs/1905.08743)

This code has been written using PyTorch >= 1.0. If you use any source codes or datasets included in this toolkit in your work, please cite the following paper. The bibtex is listed below:
<pre>
@InProceedings{WuTradeDST2019,
  	author = "Wu, Chien-Sheng and Madotto, Andrea and Hosseini-Asl, Ehsan and Xiong, Caiming and Socher, Richard and Fung, Pascale",
  	title = 	"Transferable Multi-Domain State Generator for Task-Oriented Dialogue Systems",
  	booktitle = 	"Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
  	year = 	"2019",
  	publisher = "Association for Computational Linguistics"
}
</pre>

## Abstract
Over-dependence on domain ontology and lack of knowledge sharing across domains are two practical and yet less studied problems of dialogue state tracking. Existing approaches generally fall short in tracking unknown slot values during inference and often have difficulties in adapting to new domains. In this paper, we propose a Transferable Dialogue State Generator (TRADE) that generates dialogue states from utterances using a copy mechanism, facilitating knowledge transfer when predicting (domain, slot, value) triplets not encountered during training. Our model is composed of an utterance encoder, a slot gate, and a state generator, which are shared across domains. Empirical results demonstrate that TRADE achieves state-of-the-art joint goal accuracy of 48.62% for the five domains of MultiWOZ, a human-human dialogue dataset. In addition, we show its transferring ability by simulating zero-shot and few-shot dialogue state tracking for unseen domains. TRADE achieves 60.58% joint goal accuracy in one of the zero-shot domains, and is able to adapt to few-shot cases without forgetting already trained domains.

## Model Architecture

<p align="center">
<img src="plot/model.png" width="75%" />
</p>
The architecture of the proposed TRADE model, which includes (a) an utterance encoder, (b) a state generator, and (c) a slot gate, all of which are shared among domains. The state generator will decode J times independently for all the possible (domain, slot) pairs. At the first decoding step, state generator will take the j-th (domain, slot) embeddings as input to generate its corresponding slot values and slot gate. The slot gate predicts whether the j-th (domain, slot) pair is triggered by the dialogue.


## Data

<p align="center">
<img src="plot/dataset.png" width="50%" />
</p>

Download the MultiWOZ dataset and the processed dst version.
```console
❱❱❱ python3 create_data.py
```
<p align="center">
<img src="plot/example.png" width="50%" />
</p>

An example of multi-domain dialogue state tracking in a conversation. The solid arrows on the left are the single-turn mapping, and the dot arrows on the right are multi-turn mapping. The state tracker needs to track slot values mentioned by the user for all the slots in all the domains.

## Dependency
Check the packages needed or simply run the command
```console
❱❱❱ pip install -r requirements.txt
```
If you run into an error related to Cython, try to upgrade it first.
```console
❱❱❱ pip install --upgrade cython
```


## Multi-Domain DST
Training
```console
❱❱❱ python3 myTrain.py -dec=TRADE -bsz=32 -dr=0.2 -lr=0.001 -le=1
```
Testing
```console
❱❱❱ python3 myTest.py -path=${save_path}
```
* -bsz: batch size
* -dr: drop out ratio
* -lr: learning rate
* -le: loading pretrained embeddings
* -path: model saved path

> [2019.08 Update] Now the decoder can generate all the (domain, slot) pairs in one batch at the same time to speedup decoding process. You can set flag "--parallel_decode=1" to decode all (domain, slot) pairs in one batch.


## Unseen Domain DST

#### Zero-Shot DST
Training
```console
❱❱❱ python3 myTrain.py -dec=TRADE -bsz=32 -dr=0.2 -lr=0.001 -le=1 -exceptd=${domain}
```
Testing
```console
❱❱❱ python3 myTest.py -path=${save_path} -exceptd=${domain}
```
* -exceptd: except domain selection, choose one from {hotel, train, attraction, restaurant, taxi}.

#### Few-Shot DST with CL
Training
Naive 
```console
❱❱❱ python3 fine_tune.py -bsz=8 -dr=0.2 -lr=0.001 -path=${save_path_except_domain} -exceptd=${except_domain}
```
EWC
```console
❱❱❱ python3 EWC_train.py -bsz=8 -dr=0.2 -lr=0.001 -path=${save_path_except_domain} -exceptd=${except_domain} -fisher_sample=10000 -l_ewc=${lambda}
```
GEM
```console
❱❱❱ python3 GEM_train.py -bsz=8 -dr=0.2 -lr=0.001 -path={save_path_except_domain} -exceptd=${except_domain}
```
* -l_ewc: lambda value in EWC training

## Other Notes
- We found that there might be some variances in different runs, especially for the few-shot setting. For our own experiments, we only use one random seed (seed=10) to do the experiments reported in the paper. Please check the results for average three runs in our [ACL presentation](https://jasonwu0731.github.io/files/TRADE-DST-ACL-2019.pdf). 

## Bug Report
Feel free to create an issue or send email to jason.wu@connect.ust.hk
