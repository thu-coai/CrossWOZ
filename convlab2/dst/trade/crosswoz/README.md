# TRADE
This is the implementation of TRADE (Transferable Dialogue State Generator) adopted from [jasonwu0731/trade-dst](https://github.com/jasonwu0731/trade-dst)
on the CrossWOZ dataset.


## Example usage
To run an example, you can jump into convlab2/dst/trade/crosswoz, and run the following command:
```bash
$ python demo.py
```
The path in the example is our proposed pre-trained model of TRADE, which will
be downloaded automatically at runtime.
The data required for model running will also be downloaded at runtime.
You can also run you own model by specifying the path parameter.

## Train
To train a model from scratch, jump into convlab/dst/trade/crosswoz, and run the following command:
```bash
$ python train.py
```
Note that the training data will be download automatically.

## Evaluation
To evaluate the model on the test set of CrossWOZ, you can jump into convlab/dst/trade/crosswoz, and then run the following command:
```bash
$ python evaluate.py
```
The evaluation results, including Joint Accuracy, Turn Accuracy, and Joing F1 on the test set will be shown.

## References
```
@InProceedings{WuTradeDST2019,
  	author = "Wu, Chien-Sheng and Madotto, Andrea and Hosseini-Asl, Ehsan and Xiong, Caiming and Socher, Richard and Fung, Pascale",
  	title = 	"Transferable Multi-Domain State Generator for Task-Oriented Dialogue Systems",
  	booktitle = 	"Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
  	year = 	"2019",
  	publisher = "Association for Computational Linguistics"
}
```
