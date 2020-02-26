import numpy as np
from convlab2.policy.larl.multiwoz.latent_dialog.enc2dec.decoders import GEN, DecoderRNN
from convlab2.policy.larl.multiwoz.latent_dialog.main import get_sent
from collections import defaultdict


def task_generate(model, data, config, evaluator, num_batch, dest_f=None, verbose=True):
    def write(msg):
        if msg is None or msg == '':
            return
        if dest_f is None:
            print(msg)
        else:
            dest_f.write(msg + '\n')

    model.eval()
    de_tknize = lambda x: ' '.join(x)
    data.epoch_init(config, shuffle=num_batch is not None, verbose=False, fix_batch=config.fix_batch)
    evaluator.initialize()
    print('Generation: {} batches'.format(data.num_batch
                                          if num_batch is None
                                          else num_batch))
    batch_cnt = 0
    generated_dialogs = defaultdict(list)
    while True:
        batch_cnt += 1
        batch = data.next_batch()
        if batch is None or (num_batch is not None and data.ptr > num_batch):
            break
        outputs, labels = model(batch, mode=GEN, gen_type=config.gen_type)

        # move from GPU to CPU
        labels = labels.cpu()
        pred_labels = [t.cpu().data.numpy() for t in outputs[DecoderRNN.KEY_SEQUENCE]]
        pred_labels = np.array(pred_labels, dtype=int).squeeze(-1).swapaxes(0, 1)  # (batch_size, max_dec_len)
        true_labels = labels.data.numpy()  # (batch_size, output_seq_len)

        # get context
        ctx = batch.get('contexts')  # (batch_size, max_ctx_len, max_utt_len)
        ctx_len = batch.get('context_lens')  # (batch_size, )
        keys = batch['keys']

        for b_id in range(pred_labels.shape[0]):
            # TODO attn
            pred_str = get_sent(model.vocab, de_tknize, pred_labels, b_id)
            true_str = get_sent(model.vocab, de_tknize, true_labels, b_id)
            prev_ctx = ''
            if ctx is not None:
                ctx_str = []
                for t_id in range(ctx_len[b_id]):
                    temp_str = get_sent(model.vocab, de_tknize, ctx[:, t_id, :], b_id, stop_eos=False)
                    # print('temp_str = %s' % (temp_str, ))
                    # print('ctx[:, t_id, :] = %s' % (ctx[:, t_id, :], ))
                    ctx_str.append(temp_str)
                ctx_str = '|'.join(ctx_str)[-200::]
                prev_ctx = 'Source context: {}'.format(ctx_str)

            generated_dialogs[keys[b_id]].append(pred_str)
            evaluator.add_example(true_str, pred_str)

            if verbose and (dest_f is not None or batch_cnt < 2):
                write('%s-prev_ctx = %s' % (keys[b_id], prev_ctx,))
                write('True: {}'.format(true_str, ))
                write('Pred: {}'.format(pred_str, ))
                write('-' * 40)

    task_report, success, match = evaluator.evaluateModel(generated_dialogs, mode=data.name)
    resp_report, bleu, prec, rec, f1 = evaluator.get_report()
    write(task_report)
    write(resp_report)
    write('Generation Done')
    return success, match, bleu, f1


def dump_latent(model, data, config):
    latent_results = defaultdict(list)
    model.eval()
    batch_cnt = 0
    de_tknize = lambda x: ' '.join(x)
    data.epoch_init(config, shuffle=False, verbose=False, fix_batch=True)

    while True:
        batch_cnt += 1
        batch = data.next_batch()
        if batch is None:
            break

        outputs, labels = model(batch, mode=GEN, gen_type=config.gen_type)
        labels = labels.cpu()
        pred_labels = [t.cpu().data.numpy() for t in outputs[DecoderRNN.KEY_SEQUENCE]]
        pred_labels = np.array(pred_labels, dtype=int).squeeze(-1).swapaxes(0, 1)  # (batch_size, max_dec_len)
        true_labels = labels.data.numpy()  # (batch_size, output_seq_len)
        sample_y = outputs['sample_z'].cpu().data.numpy().reshape(-1, config.y_size, config.k_size)
        log_qy = outputs['log_qy'].cpu().data.numpy().reshape(-1, config.y_size, config.k_size)

        if config.dec_use_attn:
            attns = [t.cpu().data.numpy() for t in outputs[DecoderRNN.KEY_ATTN_SCORE]]
            attns = np.array(attns).squeeze(2).swapaxes(0, 1)
        else:
            attns = None

        # get context
        ctx = batch.get('contexts')  # (batch_size, max_ctx_len, max_utt_len)
        ctx_len = batch.get('context_lens')  # (batch_size, )
        keys = batch['keys']

        for b_id in range(pred_labels.shape[0]):
            pred_str = get_sent(model.vocab, de_tknize, pred_labels, b_id)
            true_str = get_sent(model.vocab, de_tknize, true_labels, b_id)
            prev_ctx = ''
            if ctx is not None:
                ctx_str = []
                for t_id in range(ctx_len[b_id]):
                    temp_str = get_sent(model.vocab, de_tknize, ctx[:, t_id, :], b_id, stop_eos=False)
                    ctx_str.append(temp_str)
                prev_ctx = 'Source context: {}'.format(ctx_str)

            latent_results[keys[b_id]].append({'context': prev_ctx, 'gt_resp': true_str,
                                               'pred_resp': pred_str, 'domain': batch['goals_list'],
                                               'sample_y': sample_y[b_id], 'log_qy': log_qy[b_id],
                                               'attns': attns[b_id] if attns is not None else None})
    latent_results = dict(latent_results)
    return latent_results












