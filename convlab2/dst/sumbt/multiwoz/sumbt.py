import logging
import random
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from pytorch_pretrained_bert.optimization import BertAdam

from convlab2.dst.sumbt.config.config import *
from convlab2.dst.sumbt.multiwoz.convert_to_glue_format import convert_to_glue_format
from convlab2.dst.sumbt.sumbt import BeliefTracker, Processor, BertTokenizer, convert_examples_to_features, logger, InputExample
from convlab2.util.file_util import cached_path
from convlab2.util.multiwoz.state import default_state
from convlab2.dst.dst import DST
from convlab2.dst.sumbt.config.config import *

import zipfile

from tensorboardX import SummaryWriter


def get_label_embedding(labels, max_seq_length, tokenizer, device):
    features = []
    for label in labels:
        label_tokens = ["[CLS]"] + tokenizer.tokenize(label) + ["[SEP]"]
        label_token_ids = tokenizer.convert_tokens_to_ids(label_tokens)
        label_len = len(label_token_ids)

        label_padding = [0] * (max_seq_length - len(label_token_ids))
        label_token_ids += label_padding
        assert len(label_token_ids) == max_seq_length

        features.append((label_token_ids, label_len))

    all_label_token_ids = torch.tensor([f[0] for f in features], dtype=torch.long).to(device)
    all_label_len = torch.tensor([f[1] for f in features], dtype=torch.long).to(device)

    return all_label_token_ids, all_label_len


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x


class MultiWozSUMBT(DST):
    def __init__(self):
        super(MultiWozSUMBT, self).__init__()
        convert_to_glue_format()

        self.belief_tracker = BeliefTracker()
        self.batch = None  # generated with dataloader
        self.current_turn = 0
        self.idx2slot = {}
        self.idx2value = {}  # slot value for each slot, use processor.get_labels()

        if DEVICE == 'cuda':
            if not torch.cuda.is_available():
                raise ValueError('cuda not available')
            n_gpu = torch.cuda.device_count()
            if n_gpu < N_GPU:
                raise ValueError('gpu not enough')

        print("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(DEVICE, n_gpu,
                                                                                            bool(N_GPU > 1), FP16))

        # Get Processor
        self.processor = Processor()
        self.label_list = self.processor.get_labels()
        self.num_labels = [len(labels) for labels in self.label_list]  # number of slot-values in each slot-type
        self.belief_tracker.init_session(self.num_labels)
        if N_GPU > 1:
                self.belief_tracker = torch.nn.DataParallel(self.belief_tracker)

        # tokenizer
        vocab_dir = os.path.join(BERT_DIR, 'vocab.txt')
        if not os.path.exists(vocab_dir):
            raise ValueError("Can't find %s " % vocab_dir)
        self.tokenizer = BertTokenizer.from_pretrained(vocab_dir, do_lower_case=DO_LOWER_CASE)

        self.num_train_steps = None
        self.accumulation = False

        logger.info('dataset processed')
        if os.path.exists(OUTPUT_DIR) and os.listdir(OUTPUT_DIR):
            print("output dir {} not empty".format(OUTPUT_DIR))
        else:
            os.mkdir(OUTPUT_DIR)

        fileHandler = logging.FileHandler(os.path.join(OUTPUT_DIR, "log.txt"))
        logger.addHandler(fileHandler)

        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        if N_GPU > 0:
            torch.cuda.manual_seed_all(SEED)

        self.state = default_state()
        # state = {'history': [[speaker, utt]], 'belief_state': {...}}

    def load_weights(self):
        DEFAULT_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        DEFAULT_ARCHIVE_FILE = os.path.join(DEFAULT_DIRECTORY, "sumbt_multiwoz.zip")
        archive_file = DEFAULT_ARCHIVE_FILE
        model_file = 'https://tatk-data.s3-ap-northeast-1.amazonaws.com/sumbt_multiwoz.zip'
        if not os.path.isfile(archive_file):
            if not model_file:
                raise Exception("No model for SC-LSTM is specified!")
            archive_file = cached_path(model_file)
        model_dir = os.path.dirname(os.path.abspath(__file__))
        if not os.path.exists(os.path.join(model_dir, 'resource')):
            archive = zipfile.ZipFile(archive_file, 'r')
            archive.extractall(model_dir)

        model_ckpt = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pytorch_model.bin')
        # model_ckpt = os.path.join(OUTPUT_DIR, "pytorch_model.bin")
        model = self.belief_tracker
        # in the case that slot and values are different between the training and evaluation
        ptr_model = torch.load(model_ckpt)
        print('loading pretrained weights')

        if N_GPU == 1:
            state = model.state_dict()
            state.update(ptr_model)
            model.load_state_dict(state)
        else:
            # print("Evaluate using only one device!")
            model.module.load_state_dict(ptr_model)

        model.to(DEVICE)

    def update(self, user_act=None):
        self.current_turn = 0
        # dialog_history: [{'sys': ..., 'usr': ...}, ...]
        dialog_history = []
        for i, (utter, utt) in enumerate(self.state['history']):
            if i == 0:
                dialog_history.append({'sys': '', 'usr': utt})
            elif i % 2 == 0:
                dialog_history.append({'sys': self.state['history'][i-1][1], 'usr': utt})

        examples = []

        for turn_idx, turn in enumerate(dialog_history):
            guid = "%s-%s-%s" % ('track', 1, turn_idx)  # line[0]: dialogue index, line[1]: turn index
            if self.accumulation:
                if turn_idx == 0:
                    assert turn['sys'] == ''
                    text_a = turn['usr']
                    text_b = ''
                else:
                    # The symbol '#' will be replaced with '[SEP]' after tokenization.
                    text_a = turn['usr'] + " # " + text_a
                    text_b = turn['sys'] + " # " + text_b
            else:
                text_a = turn['usr']  # line[2]: user utterance
                text_b = turn['sys']  # line[3]: system response

            # label = ['none' for idx in self.target_slot]
            label = None

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        all_input_ids, all_input_len, all_label_ids = convert_examples_to_features(
            examples, self.label_list, MAX_SEQ_LENGTH, self.tokenizer, MAX_TURN_LENGTH)
        all_input_ids, all_input_len = all_input_ids.to(DEVICE), all_input_len.to(
            DEVICE)

        pred_output = self.belief_tracker(all_input_ids, all_input_len, None, N_GPU)  # [num_slot, ds, num_turn, num_slot_value]
        pred_output = [torch.argmax(pred_slot_res, 2) for pred_slot_res in pred_output]  # [num_slot, ds, num_turn]

        self.pred_slot = []  # [[[(slot, value), slot_value2, ...], turn2, ...], dialog2, ...]
        print(len(pred_output))
        for slot_idx, slot_pred_res in enumerate(pred_output):
            slot_str = self.processor.target_slot[slot_idx]
            ds, num_turn = slot_pred_res.shape
            if self.pred_slot == []:
                for d in range(ds):
                    self.pred_slot.append([])
                for t in range(num_turn):
                    for d in range(ds):
                        self.pred_slot[d].append([])
                for d in range(ds):
                    for t in range(num_turn):
                        self.pred_slot[d][t] = {}

            for d in range(ds):
                for t in range(num_turn):
                    slot_value_idx = slot_pred_res[d, t]
                    pred_slot_value = self.processor.ontology[slot_str][slot_value_idx]
                    self.pred_slot[d][t][slot_str] = pred_slot_value

        for key in self.pred_slot[0][-1]:
            domain, slot_key = key.split('-')
            slot_list = slot_key.split()
            if domain not in self.state['belief_state']:
                print('{} not in self.state'.format(domain))
                continue

            if len(slot_list) == 1:
                self.state['belief_state'][domain]['semi'][slot_list[0]] = self.pred_slot[0][-1][key]
            else:
                self.state['belief_state'][domain]['book'][slot_list[1]] = self.pred_slot[0][-1][key]
        # print(self.state)
        return self.state

    def train(self, load_model=False, start_epoch=0, start_step=0):
        if load_model:
            self.load_weights()

        train_examples = self.processor.get_train_examples(TMP_DATA_DIR, accumulation=self.accumulation)
        dev_examples = self.processor.get_dev_examples(TMP_DATA_DIR, accumulation=self.accumulation)

        ## Training utterances
        all_input_ids, all_input_len, all_label_ids = convert_examples_to_features(
            train_examples, self.label_list, MAX_SEQ_LENGTH, self.tokenizer, MAX_TURN_LENGTH)

        num_train_batches = all_input_ids.size(0)
        num_train_steps = int(
            num_train_batches / BATCH_SIZE / GRADIENT_ACCUM_STEPS * EPOCHS)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", BATCH_SIZE)
        logger.info("  Num steps = %d", num_train_steps)

        all_input_ids, all_input_len, all_label_ids = all_input_ids.to(DEVICE), all_input_len.to(
            DEVICE), all_label_ids.to(DEVICE)

        train_data = TensorDataset(all_input_ids, all_input_len, all_label_ids)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

        all_input_ids_dev, all_input_len_dev, all_label_ids_dev = convert_examples_to_features(
            dev_examples, self.label_list, MAX_SEQ_LENGTH, self.tokenizer, MAX_TURN_LENGTH)

        logger.info("***** Running validation *****")
        logger.info("  Num examples = %d", len(dev_examples))
        logger.info("  Batch size = %d", BATCH_SIZE)

        all_input_ids_dev, all_input_len_dev, all_label_ids_dev = \
            all_input_ids_dev.to(DEVICE), all_input_len_dev.to(DEVICE), all_label_ids_dev.to(DEVICE)

        dev_data = TensorDataset(all_input_ids_dev, all_input_len_dev, all_label_ids_dev)
        dev_sampler = SequentialSampler(dev_data)
        dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=BATCH_SIZE)

        logger.info("Loaded data!")

        if FP16:
            self.belief_tracker.half()
        self.belief_tracker.to(DEVICE)

        ## Get domain-slot-type embeddings
        slot_token_ids, slot_len = \
            get_label_embedding(self.processor.target_slot, MAX_LABEL_LENGTH, self.tokenizer, DEVICE)

        # for slot_idx, slot_str in zip(slot_token_ids, self.processor.target_slot):
        #     self.idx2slot[slot_idx] = slot_str

        ## Get slot-value embeddings
        label_token_ids, label_len = [], []
        for slot_idx, labels in zip(slot_token_ids, self.label_list):
            # self.idx2value[slot_idx] = {}
            token_ids, lens = get_label_embedding(labels, MAX_LABEL_LENGTH, self.tokenizer, DEVICE)
            label_token_ids.append(token_ids)
            label_len.append(lens)
            # for label, token_id in zip(labels, token_ids):
            #     self.idx2value[slot_idx][token_id] = label

        logger.info('embeddings prepared')

        if N_GPU > 1:
            self.belief_tracker.module.initialize_slot_value_lookup(label_token_ids, slot_token_ids)
        else:
            self.belief_tracker.initialize_slot_value_lookup(label_token_ids, slot_token_ids)

        def get_optimizer_grouped_parameters(model):
            param_optimizer = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01,
                 'lr': LEARNING_RATE},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
                 'lr': LEARNING_RATE},
            ]
            return optimizer_grouped_parameters

        if N_GPU == 1:
            optimizer_grouped_parameters = get_optimizer_grouped_parameters(self.belief_tracker)
        else:
            optimizer_grouped_parameters = get_optimizer_grouped_parameters(self.belief_tracker.module)

        t_total = num_train_steps

        if FP16:
            try:
                from apex.optimizers import FP16_Optimizer
                from apex.optimizers import FusedAdam
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            optimizer = FusedAdam(optimizer_grouped_parameters,
                                  lr=LEARNING_RATE,
                                  bias_correction=False,
                                  max_grad_norm=1.0)
            if FP16_LOSS_SCALE == 0:
                optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
            else:
                optimizer = FP16_Optimizer(optimizer, static_loss_scale=FP16_LOSS_SCALE)

        else:
            optimizer = BertAdam(optimizer_grouped_parameters,
                                 lr=LEARNING_RATE,
                                 warmup=WARM_UP_PROPORTION,
                                 t_total=t_total)
        logger.info(optimizer)

        # Training code
        ###############################################################################

        logger.info("Training...")

        global_step = start_step
        last_update = None
        best_loss = None
        model = self.belief_tracker
        if not TENSORBOARD:
            summary_writer = None
        else:
            summary_writer = SummaryWriter("./tensorboard_summary/logs_1214/" )

        for epoch in trange(start_epoch, int(EPOCHS), desc="Epoch"):
            # Train
            model.train()
            tr_loss = 0
            nb_tr_examples = 0
            nb_tr_steps = 0

            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(DEVICE) for t in batch)
                input_ids, input_len, label_ids = batch

                # Forward
                if N_GPU == 1:
                    loss, loss_slot, acc, acc_slot, _ = model(input_ids, input_len, label_ids, N_GPU)
                else:
                    loss, _, acc, acc_slot, _ = model(input_ids, input_len, label_ids, N_GPU)

                    # average to multi-gpus
                    loss = loss.mean()
                    acc = acc.mean()
                    acc_slot = acc_slot.mean(0)

                if GRADIENT_ACCUM_STEPS > 1:
                    loss = loss / GRADIENT_ACCUM_STEPS

                # Backward
                if FP16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                # tensrboard logging
                if summary_writer is not None:
                    summary_writer.add_scalar("Epoch", epoch, global_step)
                    summary_writer.add_scalar("Train/Loss", loss, global_step)
                    summary_writer.add_scalar("Train/JointAcc", acc, global_step)
                    if N_GPU == 1:
                        for i, slot in enumerate(self.processor.target_slot):
                            summary_writer.add_scalar("Train/Loss_%s" % slot.replace(' ', '_'), loss_slot[i],
                                                      global_step)
                            summary_writer.add_scalar("Train/Acc_%s" % slot.replace(' ', '_'), acc_slot[i], global_step)

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % GRADIENT_ACCUM_STEPS == 0:
                    # modify lealrning rate with special warm up BERT uses
                    lr_this_step = LEARNING_RATE * warmup_linear(global_step / t_total, WARM_UP_PROPORTION)
                    if summary_writer is not None:
                        summary_writer.add_scalar("Train/LearningRate", lr_this_step, global_step)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

            # Perform evaluation on validation dataset
            model.eval()
            dev_loss = 0
            dev_acc = 0
            dev_loss_slot, dev_acc_slot = None, None
            nb_dev_examples, nb_dev_steps = 0, 0

            for step, batch in enumerate(tqdm(dev_dataloader, desc="Validation")):
                batch = tuple(t.to(DEVICE) for t in batch)
                input_ids, input_len, label_ids = batch
                if input_ids.dim() == 2:
                    input_ids = input_ids.unsqueeze(0)
                    input_len = input_len.unsqueeze(0)
                    label_ids = label_ids.unsuqeeze(0)

                with torch.no_grad():
                    if N_GPU == 1:
                        loss, loss_slot, acc, acc_slot, _ = model(input_ids, input_len, label_ids, N_GPU)
                    else:
                        loss, _, acc, acc_slot, _ = model(input_ids, input_len, label_ids, N_GPU)

                        # average to multi-gpus
                        loss = loss.mean()
                        acc = acc.mean()
                        acc_slot = acc_slot.mean(0)

                num_valid_turn = torch.sum(label_ids[:, :, 0].view(-1) > -1, 0).item()
                dev_loss += loss.item() * num_valid_turn
                dev_acc += acc.item() * num_valid_turn

                if N_GPU == 1:
                    if dev_loss_slot is None:
                        dev_loss_slot = [l * num_valid_turn for l in loss_slot]
                        dev_acc_slot = acc_slot * num_valid_turn
                    else:
                        for i, l in enumerate(loss_slot):
                            dev_loss_slot[i] = dev_loss_slot[i] + l * num_valid_turn
                        dev_acc_slot += acc_slot * num_valid_turn

                nb_dev_examples += num_valid_turn

            dev_loss = dev_loss / nb_dev_examples
            dev_acc = dev_acc / nb_dev_examples

            if N_GPU == 1:
                dev_acc_slot = dev_acc_slot / nb_dev_examples

            # tensorboard logging
            if summary_writer is not None:
                summary_writer.add_scalar("Validate/Loss", dev_loss, global_step)
                summary_writer.add_scalar("Validate/Acc", dev_acc, global_step)
                if N_GPU == 1:
                    for i, slot in enumerate(self.processor.target_slot):
                        summary_writer.add_scalar("Validate/Loss_%s" % slot.replace(' ', '_'),
                                                  dev_loss_slot[i] / nb_dev_examples, global_step)
                        summary_writer.add_scalar("Validate/Acc_%s" % slot.replace(' ', '_'), dev_acc_slot[i],
                                                  global_step)

            dev_loss = round(dev_loss, 6)

            output_model_file = os.path.join(OUTPUT_DIR, "pytorch_model.bin")

            if last_update is None or dev_loss < best_loss:

                if N_GPU == 1:
                    torch.save(model.state_dict(), output_model_file)
                else:
                    torch.save(model.module.state_dict(), output_model_file)

                last_update = epoch
                best_loss = dev_loss
                best_acc = dev_acc

                logger.info("*** Model Updated: Epoch=%d, Validation Loss=%.6f, Validation Acc=%.6f, global_step=%d ***" % (
                last_update, best_loss, best_acc, global_step))
            else:
                logger.info("*** Model NOT Updated: Epoch=%d, Validation Loss=%.6f, Validation Acc=%.6f, global_step=%d  ***" % (
                epoch, dev_loss, dev_acc, global_step))

            if last_update + PATIENCE <= epoch:
                break

    def test(self, mode='dev'):
        # Evaluation
        self.load_weights()

        if mode == 'test':
            eval_examples = self.processor.get_test_examples(TMP_DATA_DIR, accumulation=self.accumulation)
        elif mode == 'dev':
            eval_examples = self.processor.get_dev_examples(TMP_DATA_DIR, accumulation=self.accumulation)

        all_input_ids, all_input_len, all_label_ids = convert_examples_to_features(
            eval_examples, self.label_list, MAX_SEQ_LENGTH, self.tokenizer, MAX_TURN_LENGTH)
        all_input_ids, all_input_len, all_label_ids = all_input_ids.to(DEVICE), all_input_len.to(
            DEVICE), all_label_ids.to(DEVICE)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", BATCH_SIZE)

        eval_data = TensorDataset(all_input_ids, all_input_len, all_label_ids)

        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=BATCH_SIZE)

        model = self.belief_tracker
        eval_loss, eval_accuracy = 0, 0
        eval_loss_slot, eval_acc_slot = None, None
        nb_eval_steps, nb_eval_examples = 0, 0

        accuracies = {'joint7': 0, 'slot7': 0, 'joint5': 0, 'slot5': 0, 'joint_rest': 0, 'slot_rest': 0,
                      'num_turn': 0, 'num_slot7': 0, 'num_slot5': 0, 'num_slot_rest': 0}

        for input_ids, input_len, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
            if input_ids.dim() == 2:
                input_ids = input_ids.unsqueeze(0)
                input_len = input_len.unsqueeze(0)
                label_ids = label_ids.unsuqeeze(0)

            with torch.no_grad():
                if N_GPU == 1:
                    loss, loss_slot, acc, acc_slot, pred_slot = model(input_ids, input_len, label_ids, N_GPU)
                else:
                    loss, _, acc, acc_slot, pred_slot = model(input_ids, input_len, label_ids, N_GPU)
                    nbatch = label_ids.size(0)
                    nslot = pred_slot.size(3)
                    pred_slot = pred_slot.view(nbatch, -1, nslot)

            accuracies = eval_all_accs(pred_slot, label_ids, accuracies)

            nb_eval_ex = (label_ids[:, :, 0].view(-1) != -1).sum().item()
            nb_eval_examples += nb_eval_ex
            nb_eval_steps += 1

            if N_GPU == 1:
                eval_loss += loss.item() * nb_eval_ex
                eval_accuracy += acc.item() * nb_eval_ex
                if eval_loss_slot is None:
                    eval_loss_slot = [l * nb_eval_ex for l in loss_slot]
                    eval_acc_slot = acc_slot * nb_eval_ex
                else:
                    for i, l in enumerate(loss_slot):
                        eval_loss_slot[i] = eval_loss_slot[i] + l * nb_eval_ex
                    eval_acc_slot += acc_slot * nb_eval_ex
            else:
                eval_loss += sum(loss) * nb_eval_ex
                eval_accuracy += sum(acc) * nb_eval_ex

        eval_loss = eval_loss / nb_eval_examples
        eval_accuracy = eval_accuracy / nb_eval_examples
        if N_GPU == 1:
            eval_acc_slot = eval_acc_slot / nb_eval_examples

        loss = None

        if N_GPU == 1:
            result = {'eval_loss': eval_loss,
                      'eval_accuracy': eval_accuracy,
                      'loss': loss,
                      'eval_loss_slot': '\t'.join([str(val / nb_eval_examples) for val in eval_loss_slot]),
                      'eval_acc_slot': '\t'.join([str((val).item()) for val in eval_acc_slot])
                      }
        else:
            result = {'eval_loss': eval_loss,
                      'eval_accuracy': eval_accuracy,
                      'loss': loss
                      }

        out_file_name = 'eval_results'
        if TARGET_SLOT == 'all':
            out_file_name += '_all'
        output_eval_file = os.path.join(OUTPUT_DIR, "%s.txt" % out_file_name)

        if N_GPU == 1:
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

        out_file_name = 'eval_all_accuracies'
        with open(os.path.join(OUTPUT_DIR, "%s.txt" % out_file_name), 'w') as f:
            f.write(
                'joint acc (7 domain) : slot acc (7 domain) : joint acc (5 domain): slot acc (5 domain): joint restaurant : slot acc restaurant \n')
            f.write('%.5f : %.5f : %.5f : %.5f : %.5f : %.5f \n' % (
                (accuracies['joint7'] / accuracies['num_turn']).item(),
                (accuracies['slot7'] / accuracies['num_slot7']).item(),
                (accuracies['joint5'] / accuracies['num_turn']).item(),
                (accuracies['slot5'] / accuracies['num_slot5']).item(),
                (accuracies['joint_rest'] / accuracies['num_turn']).item(),
                (accuracies['slot_rest'] / accuracies['num_slot_rest']).item()
            ))


def eval_all_accs(pred_slot, labels, accuracies):

    def _eval_acc(_pred_slot, _labels):
        slot_dim = _labels.size(-1)
        accuracy = (_pred_slot == _labels).view(-1, slot_dim)
        num_turn = torch.sum(_labels[:, :, 0].view(-1) > -1, 0).float()
        num_data = torch.sum(_labels > -1).float()
        # joint accuracy
        joint_acc = sum(torch.sum(accuracy, 1) / slot_dim).float()
        # slot accuracy
        slot_acc = torch.sum(accuracy).float()
        return joint_acc, slot_acc, num_turn, num_data

    # 7 domains
    joint_acc, slot_acc, num_turn, num_data = _eval_acc(pred_slot, labels)
    accuracies['joint7'] += joint_acc
    accuracies['slot7'] += slot_acc
    accuracies['num_turn'] += num_turn
    accuracies['num_slot7'] += num_data

    # restaurant domain
    joint_acc, slot_acc, num_turn, num_data = _eval_acc(pred_slot[:,:,18:25], labels[:,:,18:25])
    accuracies['joint_rest'] += joint_acc
    accuracies['slot_rest'] += slot_acc
    accuracies['num_slot_rest'] += num_data

    pred_slot5 = torch.cat((pred_slot[:,:,0:3], pred_slot[:,:,8:]), 2)
    label_slot5 = torch.cat((labels[:,:,0:3], labels[:,:,8:]), 2)

    # 5 domains (excluding bus and hotel domain)
    joint_acc, slot_acc, num_turn, num_data = _eval_acc(pred_slot5, label_slot5)
    accuracies['joint5'] += joint_acc
    accuracies['slot5'] += slot_acc
    accuracies['num_slot5'] += num_data

    return accuracies


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true')
parser.add_argument('--dev', action='store_true')
parser.add_argument('--test', action='store_true')


if __name__ == "__main__":

    # sumbt = MultiWozSUMBT()
    # sumbt.train(load_model=False, start_epoch=0, start_step=0)
    args = parser.parse_args()
    sumbt = MultiWozSUMBT()
    if args.train:
        sumbt.train()
    elif args.dev:
        sumbt.load_weights()
        sumbt.test('dev')
    elif args.test:
        sumbt.load_weights()
        sumbt.test('test')
    else:
        sumbt.load_weights()
        sumbt.state = {'belief_state': {
                            "police": {
                                "book": {
                                    "booked": []
                                },
                                "semi": {}
                            },
                            "hotel": {
                                "book": {
                                    "booked": [],
                                    "people": "",
                                    "day": "",
                                    "stay": ""
                                },
                                "semi": {
                                    "name": "",
                                    "area": "",
                                    "parking": "",
                                    "pricerange": "",
                                    "stars": "",
                                    "internet": "",
                                    "type": ""
                                }
                            },
                            "attraction": {
                                "book": {
                                    "booked": []
                                },
                                "semi": {
                                    "type": "",
                                    "name": "",
                                    "area": ""
                                }
                            },
                            "restaurant": {
                                "book": {
                                    "booked": [],
                                    "people": "",
                                    "day": "",
                                    "time": ""
                                },
                                "semi": {
                                    "food": "",
                                    "pricerange": "",
                                    "name": "",
                                    "area": "",
                                }
                            },
                            "hospital": {
                                "book": {
                                    "booked": []
                                },
                                "semi": {
                                    "department": ""
                                }
                            },
                            "taxi": {
                                "book": {
                                    "booked": []
                                },
                                "semi": {
                                    "leaveAt": "",
                                    "destination": "",
                                    "departure": "",
                                    "arriveBy": ""
                                }
                            },
                            "train": {
                                "book": {
                                    "booked": [],
                                    "people": ""
                                },
                                "semi": {
                                    "leaveAt": "",
                                    "destination": "",
                                    "day": "",
                                    "arriveBy": "",
                                    "departure": ""
                                }
                            }
                        },
                       'history': [['a', 'I need to book a hotel in the east that has 4 stars.'],
                                   ['b', 'I can help you with that . What is your price range?'],
                                   ['a', 'That doesn\'t matter as long as it has free wifi and parking.'],
                                   ['b', 'If you\'d like something cheap , I recommend the Allenbell . For something moderately priced , I would recommend the Warkworth House .'],
                                   ['a', 'Could you book the Wartworth for one night , 1 person ?'],
                                   ['b', 'What day will you be staying ?'],
                                   ['a', 'Friday and Can you book it for me and get a reference number?'],
                                   ['b', 'Booking was successful. Reference number is : BMUKPTG6. Can I help you with anything else today ? '],
                                   ['a', 'I am looking to book a train that is leaving from Cambridge to Bishops Stortford on Friday.'],
                                   ['b', 'There are a number of trains leaving throughout the day.  What time would you like to travel?'],
                                   ['a', 'I want to get there by 19:45 at the latest.'],
                                   ['b', 'Okay! The latest train you can take leaves at 17:29, and arrives by 18:07. Would you like for me to book that for you?'],
                                   ['a', 'Yes please. I also need the travel time, departure time, and price.'],
                                   ['b', 'Reference number is: UIFV8FAS . The price is 10.1 GBP and the trip will take about 38 minutes. May I be of any other assistance?'],
                                   ['a', 'Yes. Sorry, but suddenly my plans changed. Can you change the Wartworth booking to Monday for 3 people and 4 nights?']]}
        sumbt.update()


