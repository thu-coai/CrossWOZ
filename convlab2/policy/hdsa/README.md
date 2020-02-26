### HDSA-Dialog Policy

#### Requirements
* Python3.6
* Pytorch 1.0
* Pytorch-pretrained-BERT

#### File
* data: all the needed training/evaluation/testing data
* transformer: all the baseline and proposed models, which include the hierarchical disentangled self-attention (class TableSemanticDecoder)
* preprocessing: the code for pre-processing the database and original downloaded data
* 

1. Dialog Act Predictor
2. Response Generator
Train
CUDA_VISIBLE_DEVICES=0 python3.5 train_generator.py --option train --model BERT_dim128_w_domain_exp --batch_size 512 --max_seq_length 50 --field