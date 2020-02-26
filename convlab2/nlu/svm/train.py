import configparser
import os
import pprint
import sys
import zipfile

from convlab2.nlu.svm import Classifier
from convlab2.nlu.svm.dataset_walker import dataset_walker


def train(config):
    c = Classifier.classifier(config)
    pprint.pprint(c.tuples.all_tuples)
    print('All tuples:',len(c.tuples.all_tuples))
    model_path = config.get("train", "output")
    model_dir = os.path.dirname(model_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    print('output to {}'.format(model_path))
    zip_name = ''.join(os.path.basename(model_path).split('.')[:-1])+'.zip'
    zip_path = os.path.join(model_dir, zip_name)
    print('zip to {}'.format(zip_path))
    dataListFile = config.get("train", "dataListFile")
    dataroot = config.get("train", "dataroot")
    dw = dataset_walker(dataListFile=dataListFile, dataroot=dataroot, labels=True)
    c = Classifier.classifier(config)
    c.cacheFeature(dw)
    c.train(dw)
    c.save(model_path)

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(model_path)


def usage():
    print("usage:")
    print("\t python train.py multiwoz/config/multiwoz_all.cfg")


if __name__ == '__main__':
    if len(sys.argv) != 2 :
        usage()
        sys.exit()
        
    config = configparser.ConfigParser()
    try :
        config.read(sys.argv[1])
    except Exception as e:
        print("Failed to parse file")
        print(e)

    train(config)
