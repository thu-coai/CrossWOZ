import os
import zipfile
import json
import torch

from convlab2.util.file_util import cached_path
from convlab2.nlu.nlu import NLU
from convlab2.nlu.jointBERT.dataloader import Dataloader
from convlab2.nlu.jointBERT.jointBERT import JointBERT
from convlab2.nlu.jointBERT.crosswoz.postprocess import recover_intent
from convlab2.nlu.jointBERT.crosswoz.preprocess import preprocess


class BERTNLU(NLU):
    def __init__(self, mode='all', config_file='crosswoz_all_context.json',
                 model_file='https://convlab.blob.core.windows.net/convlab-2/bert_crosswoz_all_context.zip'):
        assert mode == 'usr' or mode == 'sys' or mode == 'all'
        config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs/{}'.format(config_file))
        config = json.load(open(config_file))
        DEVICE = config['DEVICE']
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(root_dir, config['data_dir'])
        output_dir = os.path.join(root_dir, config['output_dir'])

        if not os.path.exists(os.path.join(data_dir, 'intent_vocab.json')):
            preprocess(mode)

        intent_vocab = json.load(open(os.path.join(data_dir, 'intent_vocab.json')))
        tag_vocab = json.load(open(os.path.join(data_dir, 'tag_vocab.json')))
        dataloader = Dataloader(intent_vocab=intent_vocab, tag_vocab=tag_vocab,
                                pretrained_weights=config['model']['pretrained_weights'])

        print('intent num:', len(intent_vocab))
        print('tag num:', len(tag_vocab))

        best_model_path = os.path.join(output_dir, 'pytorch_model.bin')
        if not os.path.exists(best_model_path):
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            print('Load from model_file param')
            archive_file = cached_path(model_file)
            archive = zipfile.ZipFile(archive_file, 'r')
            archive.extractall(root_dir)
            archive.close()
        print('Load from', best_model_path)
        model = JointBERT(config['model'], DEVICE, dataloader.tag_dim, dataloader.intent_dim)
        model.load_state_dict(torch.load(os.path.join(output_dir, 'pytorch_model.bin'), DEVICE))
        model.to(DEVICE)
        model.eval()

        self.model = model
        self.dataloader = dataloader
        print("BERTNLU loaded")

    def predict(self, utterance, context=list()):
        ori_word_seq = self.dataloader.tokenizer.tokenize(utterance)
        ori_tag_seq = ['O'] * len(ori_word_seq)
        context_seq = self.dataloader.tokenizer.encode('[CLS] ' + ' [SEP] '.join(context[-3:]))
        intents = []
        da = {}

        word_seq, tag_seq, new2ori = ori_word_seq, ori_tag_seq, None
        batch_data = [[ori_word_seq, ori_tag_seq, intents, da, context_seq,
                       new2ori, word_seq, self.dataloader.seq_tag2id(tag_seq), self.dataloader.seq_intent2id(intents)]]

        pad_batch = self.dataloader.pad_batch(batch_data)
        pad_batch = tuple(t.to(self.model.device) for t in pad_batch)
        word_seq_tensor, tag_seq_tensor, intent_tensor, word_mask_tensor, tag_mask_tensor, context_seq_tensor, context_mask_tensor = pad_batch
        slot_logits, intent_logits = self.model.forward(word_seq_tensor, word_mask_tensor,
                                                        context_seq_tensor=context_seq_tensor,
                                                        context_mask_tensor=context_mask_tensor)
        intent = recover_intent(self.dataloader, intent_logits[0], slot_logits[0], tag_mask_tensor[0],
                                batch_data[0][0], batch_data[0][-4])
        return intent


if __name__ == '__main__':
    nlu = BERTNLU(mode='all', config_file='crosswoz_all_context.json',
                  model_file='https://convlab.blob.core.windows.net/convlab-2/bert_crosswoz_all_context.zip')
    print(nlu.predict("北京布提克精品酒店酒店是什么类型，有健身房吗？", ['你好，给我推荐一个评分是5分，价格在100-200元的酒店。', '推荐您去北京布提克精品酒店。']))
