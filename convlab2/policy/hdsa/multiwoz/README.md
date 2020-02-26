# HDSA

HDSA constructs a dialog act **graph** to represent the semantic space in task-oriented dialog for language generation. We only migrate the inference part of the code here. Please refer to [github](https://github.com/wenhuchen/HDSA-Dialog) for training and other aspects. The original paper can be found at [ACL anthology](https://aclweb.org/anthology/papers/P/P19/P19-1360/)

The architecture consists of two components:
- Dialog act predictor (Fine-tuned BERT model)
- Response generator (Hierarchical Disentangled Self-Attention Network)

## Requirements
- Python 3.5
- [Pytorch 1.0](https://pytorch.org/)
- [Pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT)

## Folder

- data: all the needed data for *inference*
- transformer: the proposed models: hierarchical disentangled self-attention (class TableSemanticDecoder)
- checkpoints: trained model files of predictor and generator

## Reference

```
@inproceedings{chen2019semantically,
	title={Semantically Conditioned Dialog Response Generation via Hierarchical Disentangled Self-Attention},
	author={Chen, Wenhu and Chen, Jianshu and Qin, Pengda and Yan, Xifeng and Wang, William Yang},
	booktitle={ACL},
	year={2019},
	pages={3696--3709}
}
```