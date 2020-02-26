import numpy as np
from convlab2.policy.larl.multiwoz.latent_dialog.utils import Pack
from convlab2.policy.larl.multiwoz.latent_dialog.base_data_loaders import BaseDataLoaders
from convlab2.policy.larl.multiwoz.latent_dialog.corpora import USR, SYS
import json


class DealDataLoaders(BaseDataLoaders):
    def __init__(self, name, data, config):
        super(DealDataLoaders, self).__init__(name)
        self.max_utt_len = config.max_utt_len
        self.data = self.flatten_dialog(data, config.backward_size)
        self.data_size = len(self.data)
        self.indexes = list(range(self.data_size))

    def flatten_dialog(self, data, backward_size):
        results = []
        for dlg in data:
            goal = dlg.goal
            for i in range(1, len(dlg.dlg)):
                if dlg.dlg[i].speaker == USR:
                    continue
                e_idx = i
                s_idx = max(0, e_idx - backward_size)
                response = dlg.dlg[i].copy()
                response['utt'] = self.pad_to(self.max_utt_len, response.utt, do_pad=False)
                context = []
                for turn in dlg.dlg[s_idx: e_idx]:
                    turn['utt'] = self.pad_to(self.max_utt_len, turn.utt, do_pad=False)
                    context.append(turn)
                results.append(Pack(context=context, response=response, goal=goal))
        return results

    def epoch_init(self, config, shuffle=True, verbose=True, fix_batch=False):
        super(DealDataLoaders, self).epoch_init(config, shuffle=shuffle, verbose=verbose)

    def _prepare_batch(self, selected_index):
        rows = [self.data[idx] for idx in selected_index]

        ctx_utts, ctx_lens = [], []
        out_utts, out_lens = [], []
        goals, goal_lens = [], []

        for row in rows:
            in_row, out_row, goal_row = row.context, row.response, row.goal

            # source context
            batch_ctx = []
            for turn in in_row:
                batch_ctx.append(self.pad_to(self.max_utt_len, turn.utt, do_pad=True))
            ctx_utts.append(batch_ctx)
            ctx_lens.append(len(batch_ctx))

            # target response
            out_utt = [t for idx, t in enumerate(out_row.utt)]
            out_utts.append(out_utt)
            out_lens.append(len(out_utt))

            # goal
            goals.append(goal_row)
            goal_lens.append(len(goal_row))

        vec_ctx_lens = np.array(ctx_lens) # (batch_size, ), number of turns
        max_ctx_len = np.max(vec_ctx_lens)
        vec_ctx_utts = np.zeros((self.batch_size, max_ctx_len, self.max_utt_len), dtype=np.int32)
        # confs is used to add some hand-crafted features
        vec_ctx_confs = np.ones((self.batch_size, max_ctx_len), dtype=np.float32)
        vec_out_lens = np.array(out_lens) # (batch_size, ), number of tokens
        max_out_len = np.max(vec_out_lens)
        vec_out_utts = np.zeros((self.batch_size, max_out_len), dtype=np.int32)

        max_goal_len, min_goal_len = max(goal_lens), min(goal_lens)
        if max_goal_len != min_goal_len or max_goal_len != 6:
            print('FATAL ERROR!')
            exit(-1)
        self.goal_len = max_goal_len
        vec_goals = np.zeros((self.batch_size, self.goal_len), dtype=np.int32)

        for b_id in range(self.batch_size):
            vec_ctx_utts[b_id, :vec_ctx_lens[b_id], :] = ctx_utts[b_id]
            vec_out_utts[b_id, :vec_out_lens[b_id]] = out_utts[b_id]
            vec_goals[b_id, :] = goals[b_id]

        return Pack(context_lens=vec_ctx_lens, \
                    contexts=vec_ctx_utts, \
                    context_confs=vec_ctx_confs, \
                    output_lens=vec_out_lens, \
                    outputs=vec_out_utts, \
                    goals=vec_goals)


class BeliefDbDataLoaders(BaseDataLoaders):
    def __init__(self, name, data, config):
        super(BeliefDbDataLoaders, self).__init__(name)
        self.max_utt_len = config.max_utt_len
        self.data, self.indexes, self.batch_indexes = self.flatten_dialog(data, config.backward_size)
        self.data_size = len(self.data)
        self.domains = ['hotel', 'restaurant', 'train', 'attraction', 'hospital', 'police', 'taxi']

    def flatten_dialog(self, data, backward_size):
        results = []
        indexes = []
        batch_indexes = []
        resp_set = set()
        for dlg in data:
            goal = dlg.goal
            key = dlg.key
            batch_index = []
            for i in range(1, len(dlg.dlg)):
                if dlg.dlg[i].speaker == USR:
                    continue
                e_idx = i
                s_idx = max(0, e_idx - backward_size)
                response = dlg.dlg[i].copy()
                response['utt'] = self.pad_to(self.max_utt_len, response.utt, do_pad=False)
                resp_set.add(json.dumps(response.utt))
                context = []
                for turn in dlg.dlg[s_idx: e_idx]:
                    turn['utt'] = self.pad_to(self.max_utt_len, turn.utt, do_pad=False)
                    context.append(turn)
                results.append(Pack(context=context, response=response, goal=goal, key=key))
                indexes.append(len(indexes))
                batch_index.append(indexes[-1])
            if len(batch_index) > 0:
                batch_indexes.append(batch_index)
        print("Unique resp {}".format(len(resp_set)))
        return results, indexes, batch_indexes

    def epoch_init(self, config, shuffle=True, verbose=True, fix_batch=False):
        self.ptr = 0
        if fix_batch:
            self.batch_size = None
            self.num_batch = len(self.batch_indexes)
        else:
            self.batch_size = config.batch_size
            self.num_batch = self.data_size // config.batch_size
            self.batch_indexes = []
            for i in range(self.num_batch):
                self.batch_indexes.append(self.indexes[i * self.batch_size: (i + 1) * self.batch_size])
            if verbose:
                print('Number of left over sample = %d' % (self.data_size - config.batch_size * self.num_batch))
        if shuffle:
            if fix_batch:
                self._shuffle_batch_indexes()
            else:
                self._shuffle_indexes()

        if verbose:
            print('%s begins with %d batches' % (self.name, self.num_batch))

    def _prepare_batch(self, selected_index):
        rows = [self.data[idx] for idx in selected_index]

        ctx_utts, ctx_lens = [], []
        out_utts, out_lens = [], []

        out_bs, out_db = [] , []
        goals, goal_lens = [], [[] for _ in range(len(self.domains))]
        keys = []

        for row in rows:
            in_row, out_row, goal_row = row.context, row.response, row.goal

            # source context
            keys.append(row.key)
            batch_ctx = []
            for turn in in_row:
                batch_ctx.append(self.pad_to(self.max_utt_len, turn.utt, do_pad=True))
            ctx_utts.append(batch_ctx)
            ctx_lens.append(len(batch_ctx))

            # target response
            out_utt = [t for idx, t in enumerate(out_row.utt)]
            out_utts.append(out_utt)
            out_lens.append(len(out_utt))

            out_bs.append(out_row.bs)
            out_db.append(out_row.db)

            # goal
            goals.append(goal_row)
            for i, d in enumerate(self.domains):
                goal_lens[i].append(len(goal_row[d]))

        batch_size = len(ctx_lens)
        vec_ctx_lens = np.array(ctx_lens) # (batch_size, ), number of turns
        max_ctx_len = np.max(vec_ctx_lens)
        vec_ctx_utts = np.zeros((batch_size, max_ctx_len, self.max_utt_len), dtype=np.int32)
        vec_out_bs = np.array(out_bs) # (batch_size, 94)
        vec_out_db = np.array(out_db) # (batch_size, 30)
        vec_out_lens = np.array(out_lens)  # (batch_size, ), number of tokens
        max_out_len = np.max(vec_out_lens)
        vec_out_utts = np.zeros((batch_size, max_out_len), dtype=np.int32)

        max_goal_lens, min_goal_lens = [max(ls) for ls in goal_lens], [min(ls) for ls in goal_lens]
        if max_goal_lens != min_goal_lens:
            print('Fatal Error!')
            exit(-1)
        self.goal_lens = max_goal_lens
        vec_goals_list = [np.zeros((batch_size, l), dtype=np.float32) for l in self.goal_lens]

        for b_id in range(batch_size):
            vec_ctx_utts[b_id, :vec_ctx_lens[b_id], :] = ctx_utts[b_id]
            vec_out_utts[b_id, :vec_out_lens[b_id]] = out_utts[b_id]
            for i, d in enumerate(self.domains):
                vec_goals_list[i][b_id, :] = goals[b_id][d]

        return Pack(context_lens=vec_ctx_lens, # (batch_size, )
                    contexts=vec_ctx_utts, # (batch_size, max_ctx_len, max_utt_len)
                    output_lens=vec_out_lens, # (batch_size, )
                    outputs=vec_out_utts, # (batch_size, max_out_len)
                    bs=vec_out_bs, # (batch_size, 94)
                    db=vec_out_db, # (batch_size, 30)
                    goals_list=vec_goals_list, # 7*(batch_size, bow_len), bow_len differs w.r.t. domain
                    keys=keys)
