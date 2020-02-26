# MILU
MILU is a joint neural model that allows you to simultaneously predict multiple dialog act items (a dialog act item takes a form of domain-intent(slot, value). Since it is common that, in a multi-domain setting, an utterance has multiple dialog act items, MILU is likely to yield higher performance than conventional single-intent models.


## Example usage
We based our implementation on the [AllenNLP library](https://github.com/allenai/allennlp). For an introduction to this library, you should check [these tutorials](https://allennlp.org/tutorials).

```bash
$ PYTHONPATH=../../.. python train.py multiwoz/configs/[base|context3].jsonnet -s serialization_dir
$ PYTHONPATH=../../.. python evaluate.py serialization_dir/model.tar.gz {test_file} --cuda-device {CUDA_DEVICE}
```

If you want to perform end-to-end evaluation, you can include the trained model by adding the model path (serialization_dir/model.tar.gz) to your ConvLab spec file.

## Data
We use the multiwoz data (data/multiwoz/[train|val|test].json.zip).

## References
```
@inproceedings{lee2019convlab,
  title={ConvLab: Multi-Domain End-to-End Dialog System Platform},
  author={Lee, Sungjin and Zhu, Qi and Takanobu, Ryuichi and Li, Xiang and Zhang, Yaoqin and Zhang, Zheng and Li, Jinchao and Peng, Baolin and Li, Xiujun and Huang, Minlie and Gao, Jianfeng},
  booktitle={Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},
  year={2019}
}
```
