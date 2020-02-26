import argparse
import csv
import functools
import json
import math
from collections import Counter
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

from convlab2.e2e.sequicity.reader import clean_replace

en_sws = set(stopwords.words())
wn = WordNetLemmatizer()

order_to_number = {
    'first': 1, 'one': 1, 'seco': 2, 'two': 2, 'third': 3, 'three': 3, 'four': 4, 'forth': 4, 'five': 5, 'fifth': 5,
    'six': 6, 'seven': 7, 'eight': 8, 'nin': 9, 'ten': 10, 'eleven': 11, 'twelve': 12
}

def similar(a,b):
    return a == b or a in b or b in a or a.split()[0] == b.split()[0] or a.split()[-1] == b.split()[-1]
    #return a == b or b.endswith(a) or a.endswith(b)    

def setsub(a,b):
    junks_a = []
    useless_constraint = ['temperature','week','est ','quick','reminder','near']
    for i in a:
        flg = False
        for j in b:
            if similar(i,j):
                flg = True
        if not flg:
            junks_a.append(i)
    for junk in junks_a:
        flg = False
        for item in useless_constraint:
            if item in junk:
                flg = True
        if not flg:
            return False
    return True

def setsim(a,b):
    a,b = set(a),set(b)
    return setsub(a,b) and setsub(b,a)

class BLEUScorer(object):
    ## BLEU score calculator via GentScorer interface
    ## it calculates the BLEU-4 by taking the entire corpus in
    ## Calulate based multiple candidates against multiple references
    def __init__(self):
        pass

    def score(self, parallel_corpus):

        # containers
        count = [0, 0, 0, 0]
        clip_count = [0, 0, 0, 0]
        r = 0
        c = 0
        weights = [0.25, 0.25, 0.25, 0.25]

        # accumulate ngram statistics
        for hyps, refs in parallel_corpus:
            hyps = [hyp.split() for hyp in hyps]
            refs = [ref.split() for ref in refs]
            for hyp in hyps:

                for i in range(4):
                    # accumulate ngram counts
                    hypcnts = Counter(ngrams(hyp, i + 1))
                    cnt = sum(hypcnts.values())
                    count[i] += cnt

                    # compute clipped counts
                    max_counts = {}
                    for ref in refs:
                        refcnts = Counter(ngrams(ref, i + 1))
                        for ng in hypcnts:
                            max_counts[ng] = max(max_counts.get(ng, 0), refcnts[ng])
                    clipcnt = dict((ng, min(count, max_counts[ng])) \
                                   for ng, count in hypcnts.items())
                    clip_count[i] += sum(clipcnt.values())

                # accumulate r & c
                bestmatch = [1000, 1000]
                for ref in refs:
                    if bestmatch[0] == 0: break
                    diff = abs(len(ref) - len(hyp))
                    if diff < bestmatch[0]:
                        bestmatch[0] = diff
                        bestmatch[1] = len(ref)
                r += bestmatch[1]
                c += len(hyp)

        # computing bleu score
        p0 = 1e-7
        bp = 1 if c > r else math.exp(1 - float(r) / float(c))
        p_ns = [float(clip_count[i]) / float(count[i] + p0) + p0 \
                for i in range(4)]
        s = math.fsum(w * math.log(p_n) \
                      for w, p_n in zip(weights, p_ns) if p_n)
        bleu = bp * math.exp(s)
        return bleu


def report(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        res = func(*args, **kwargs)
        args[0].metric_dict[func.__name__ + ' '+str(args[2])] = res
        return res
    return wrapper


class GenericEvaluator:
    def __init__(self, result_path):
        self.file = open(result_path,'r')
        self.meta = []
        self.metric_dict = {}
        self.entity_dict = {}
        filename = result_path.split('/')[-1]
        dump_dir = './sheets/' + filename.replace('.csv','.report.txt')
        self.dump_file = open(dump_dir,'w')

    def _print_dict(self, dic):
        for k, v in sorted(dic.items(),key=lambda x:x[0]):
            print(k+'\t'+str(v))

    @report
    def bleu_metric(self,data,type='bleu'):
        gen, truth = [],[]
        for row in data:
            gen.append(row['generated_response'])
            truth.append(row['response'])
        wrap_generated = [[_] for _ in gen]
        wrap_truth = [[_] for _ in truth]
        sc = BLEUScorer().score(zip(wrap_generated, wrap_truth))
        return sc

    def run_metrics(self):
        raise ValueError('Please specify the evaluator first, bro')

    def read_result_data(self):
        while True:
            line = self.file.readline()
            if 'START_CSV_SECTION' in line:
                break
            self.meta.append(line)
        reader = csv.DictReader(self.file)
        data = [_ for _ in reader]
        return data

    def _extract_constraint(self, z):
        z = z.split()
        if 'EOS_Z1' not in z:
            return set(z).difference(['name', 'address', 'postcode', 'phone', 'area', 'pricerange', 'restaurant',
                                           'restaurants', 'style', 'price', 'food', 'EOS_M'])
        else:
            idx = z.index('EOS_Z1')
            return set(z[:idx]).difference(['name', 'address', 'postcode', 'phone', 'area', 'pricerange', 'restaurant',
                                           'restaurants', 'style', 'price', 'food', 'EOS_M'])

    def _extract_request(self, z):
        z = z.split()
        if 'EOS_Z1' not in z or z[-1] == 'EOS_Z1':
            return set()
        else:
            idx = z.index('EOS_Z1')
            return set(z[idx+1:])

    def pack_dial(self,data):
        dials = {}
        for turn in data:
            dial_id = int(turn['dial_id'])
            if dial_id not in dials:
                dials[dial_id] = []
            dials[dial_id].append(turn)
        return dials

    def dump(self):
        self.dump_file.writelines(self.meta)
        self.dump_file.write('START_REPORT_SECTION\n')
        for k,v in self.metric_dict.items():
            self.dump_file.write('{}\t{}\n'.format(k,v))


    def clean(self,s):
        s = s.replace('<go> ', '').replace(' SLOT', '_SLOT')
        s = '<GO> ' + s + ' </s>'
        for item in self.entity_dict:
            # s = s.replace(item, 'VALUE_{}'.format(self.entity_dict[item]))
            s = clean_replace(s, item, '{}_SLOT'.format(self.entity_dict[item]))
        return s


class CamRestEvaluator(GenericEvaluator):
    def __init__(self, result_path):
        super().__init__(result_path)
        self.entities = []
        self.entity_dict = {}

    def run_metrics(self):
        raw_json = open('camrest/data/CamRest676.json')
        raw_entities = open('camrest/data/CamRestOTGY.json')
        raw_data = json.loads(raw_json.read().lower())
        raw_entities = json.loads(raw_entities.read().lower())
        self.get_entities(raw_entities)
        data = self.read_result_data()
        for i, row in enumerate(data):
            data[i]['response'] = self.clean(data[i]['response'])
            data[i]['generated_response'] = self.clean(data[i]['generated_response'])
        bleu_score = self.bleu_metric(data,'bleu')
        success_f1 = self.success_f1_metric(data, 'success')
        match = self.match_metric(data, 'match', raw_data=raw_data)
        self._print_dict(self.metric_dict)
        return -success_f1[0]

    def get_entities(self, entity_data):
        for k in entity_data['informable']:
            self.entities.extend(entity_data['informable'][k])
            for item in entity_data['informable'][k]:
                self.entity_dict[item] = k

    def _extract_constraint(self, z):
        z = z.split()
        if 'EOS_Z1' not in z:
            s = set(z)
        else:
            idx = z.index('EOS_Z1')
            s = set(z[:idx])
        if 'moderately' in s:
            s.discard('moderately')
            s.add('moderate')
        #print(self.entities) 
        #return s
        return s.intersection(self.entities)
        #return set(z).difference(['name', 'address', 'postcode', 'phone', 'area', 'pricerange'])

    def _extract_request(self, z):
        z = z.split()
        return set(z).intersection(['address', 'postcode', 'phone', 'area', 'pricerange','food'])

    @report
    def match_metric(self, data, sub='match',raw_data=None):
        dials = self.pack_dial(data)
        match,total = 0,1e-8
        success = 0
        # find out the last placeholder and see whether that is correct
        # if no such placeholder, see the final turn, because it can be a yes/no question or scheduling dialogue
        for dial_id in dials:
            truth_req, gen_req = [], []
            dial = dials[dial_id]
            gen_bspan, truth_cons, gen_cons = None, None, set()
            truth_turn_num = -1
            truth_response_req = []
            for turn_num,turn in enumerate(dial):
                if 'SLOT' in turn['generated_response']:
                    gen_bspan = turn['generated_bspan']
                    gen_cons = self._extract_constraint(gen_bspan)
                if 'SLOT' in turn['response']:
                    truth_cons = self._extract_constraint(turn['bspan'])
                gen_response_token = turn['generated_response'].split()
                response_token = turn['response'].split()
                for idx, w in enumerate(gen_response_token):
                    if w.endswith('SLOT') and w != 'SLOT':
                        gen_req.append(w.split('_')[0])
                    if w == 'SLOT' and idx != 0:
                        gen_req.append(gen_response_token[idx - 1])
                for idx, w in enumerate(response_token):
                    if w.endswith('SLOT') and w != 'SLOT':
                        truth_response_req.append(w.split('_')[0])
            if not gen_cons:
                gen_bspan = dial[-1]['generated_bspan']
                gen_cons = self._extract_constraint(gen_bspan)
            if truth_cons:
                if gen_cons == truth_cons:
                    match += 1
                else:
                    print(gen_cons, truth_cons)
                total += 1

        return match / total, success / total

    @report
    def success_f1_metric(self, data, sub='successf1'):
        dials = self.pack_dial(data)
        tp,fp,fn = 0,0,0
        for dial_id in dials:
            truth_req, gen_req = set(),set()
            dial = dials[dial_id]
            for turn_num, turn in enumerate(dial):
                gen_response_token = turn['generated_response'].split()
                response_token = turn['response'].split()
                for idx, w in enumerate(gen_response_token):
                    if w.endswith('SLOT') and w != 'SLOT':
                        gen_req.add(w.split('_')[0])
                for idx, w in enumerate(response_token):
                    if w.endswith('SLOT') and w != 'SLOT':
                        truth_req.add(w.split('_')[0])

            gen_req.discard('name')
            truth_req.discard('name')
            for req in gen_req:
                if req in truth_req:
                    tp += 1
                else:
                    fp += 1
            for req in truth_req:
                if req not in gen_req:
                    fn += 1
        precision, recall = tp / (tp + fp + 1e-8), tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return f1, precision, recall

class KvretEvaluator(GenericEvaluator):
    def __init__(self, result_path):
        super().__init__(result_path)
        ent_json = open('./data/kvret/kvret_entities.json')
        self.ent_data = json.loads(ent_json.read().lower())
        ent_json.close()
        self._get_entity_dict(self.ent_data)
        raw_json = open('./data/kvret/kvret_test_public.json')
        self.raw_data = json.loads(raw_json.read().lower())
        raw_json.close()

    def run_metrics(self):
        data = self.read_result_data()
        for i, row in enumerate(data):
            data[i]['response'] = self.clean_by_intent(data[i]['response'],int(data[i]['dial_id']))
            data[i]['generated_response'] = self.clean_by_intent(data[i]['generated_response'],int(data[i]['dial_id']))
        match_rate = self.match_rate_metric(data, 'match')
        bleu_score = self.bleu_metric(data,'bleu')
        success_f1 = self.success_f1_metric(data,'success_f1')
        self._print_dict(self.metric_dict)

    def clean_by_intent(self,s,i):
        s = s.replace('<go> ', '').replace(' SLOT', '_SLOT')
        s = '<GO> ' + s + ' </s>'
        intent = self.raw_data[i]['scenario']['task']['intent']
        slot = {
            'weather':['weather_attribute','location','weekly_time'],
            'navigate':['poi','poi_type','distance','traffic','address'],
            'schedule':['event','date','time','party','room','agenda']
        }

        for item in self.entity_dict:
            if self.entity_dict[item] in slot[intent]:
                # s = s.replace(item, 'VALUE_{}'.format(self.entity_dict[item]))
                s = clean_replace(s, item, '{}_SLOT'.format(self.entity_dict[item]))
        return s


    def _extract_constraint(self, z):
        z = z.split()
        if 'EOS_Z1' not in z:
            s = set(z)
        else:
            idx = z.index('EOS_Z1')
            s = set(z[:idx])
        reqs = ['address', 'traffic', 'poi', 'poi_type', 'distance', 'weather', 'temperature', 'weather_attribute',
                'date', 'time', 'location', 'event', 'agenda', 'party', 'room', 'weekly_time', 'forecast']
        informable = {
            'weather': ['date','location','weather_attribute'],
            'navigate': ['poi_type','distance'],
            'schedule': ['event', 'date', 'time', 'agenda', 'party', 'room']
        }
        infs = []
        for v in informable.values():
            infs.extend(v)
        junk = ['good','great','quickest','shortest','route','week','fastest','nearest','next','closest','way','mile',
               'activity','restaurant','appointment' ]
        s = s.difference(junk).difference(en_sws).difference(reqs)
        res = set()
        for item in s:
            if item in junk:
                continue
            flg = False
            for canon_ent in sorted(list(self.entity_dict.keys())):
                if self.entity_dict[canon_ent] in infs:
                    if similar(item, canon_ent):
                        flg = True
                        junk.extend(canon_ent.split())
                        res.add(canon_ent)
                    if flg:
                        break
        return res

    def constraint_same(self, truth_cons, gen_cons):
        if not truth_cons and not gen_cons:
            return True
        if not truth_cons or not gen_cons:
            return False
        return setsim(gen_cons, truth_cons)

    def _get_entity_dict(self, entity_data):
        entity_dict = {}
        for k in entity_data:
            if type(entity_data[k][0]) is str:
                for entity in entity_data[k]:
                    entity = self._lemmatize(self._tokenize(entity))
                    entity_dict[entity] = k
                    if k in ['event','poi_type']:
                        entity_dict[entity.split()[0]] = k
            elif type(entity_data[k][0]) is dict:
                for entity_entry in entity_data[k]:
                    for entity_type, entity in entity_entry.items():
                        entity_type = 'poi_type' if entity_type == 'type' else entity_type
                        entity = self._lemmatize(self._tokenize(entity))
                        entity_dict[entity] = entity_type
                        if entity_type in ['event', 'poi_type']:
                            entity_dict[entity.split()[0]] = entity_type
        self.entity_dict = entity_dict

    @report
    def match_rate_metric(self, data, sub='match',bspans='./data/kvret/test.bspan.pkl'):
        dials = self.pack_dial(data)
        match,total = 0,1e-8
        #bspan_data = pickle.load(open(bspans,'rb'))
        # find out the last placeholder and see whether that is correct
        # if no such placeholder, see the final turn, because it can be a yes/no question or scheduling conversation
        for dial_id in dials:
            dial = dials[dial_id]
            gen_bspan, truth_cons, gen_cons = None, None, set()
            truth_turn_num = -1
            for turn_num,turn in enumerate(dial):
                if 'SLOT' in turn['generated_response']:
                    gen_bspan = turn['generated_bspan']
                    gen_cons = self._extract_constraint(gen_bspan)
                if 'SLOT' in turn['response']:
                    truth_cons = self._extract_constraint(turn['bspan'])

            # KVRET dataset includes "scheduling" (so often no SLOT decoded in ground truth)
            if not truth_cons:
                truth_bspan = dial[-1]['bspan']
                truth_cons = self._extract_constraint(truth_bspan)
            if not gen_cons:
                gen_bspan = dial[-1]['generated_bspan']
                gen_cons = self._extract_constraint(gen_bspan)

            if truth_cons:
                if self.constraint_same(gen_cons, truth_cons):
                    match += 1
                    #print(gen_cons, truth_cons, '+')
                else:
                    print(gen_cons, truth_cons, '-')
                total += 1

        return match / total

    def _tokenize(self, sent):
        return ' '.join(word_tokenize(sent))

    def _lemmatize(self, sent):
        words = [wn.lemmatize(_) for _ in sent.split()]
        #for idx,w in enumerate(words):
        #    if w !=
        return ' '.join(words)

    @report
    def success_f1_metric(self, data, sub='successf1'):
        dials = self.pack_dial(data)
        tp,fp,fn = 0,0,0
        for dial_id in dials:
            truth_req, gen_req = set(),set()
            dial = dials[dial_id]
            for turn_num, turn in enumerate(dial):
                gen_response_token = turn['generated_response'].split()
                response_token = turn['response'].split()
                for idx, w in enumerate(gen_response_token):
                    if w.endswith('SLOT') and w != 'SLOT':
                        gen_req.add(w.split('_')[0])
                for idx, w in enumerate(response_token):
                    if w.endswith('SLOT') and w != 'SLOT':
                        truth_req.add(w.split('_')[0])
            gen_req.discard('name')
            truth_req.discard('name')
            for req in gen_req:
                if req in truth_req:
                    tp += 1
                else:
                    fp += 1
            for req in truth_req:
                if req not in gen_req:
                    fn += 1
        precision, recall = tp / (tp + fp + 1e-8), tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return f1

class MultiWozEvaluator(GenericEvaluator):
    def __init__(self, result_path):
        super().__init__(result_path)
        self.entities = []
        self.entity_dict = {}

    def run_metrics(self):
        with open('multiwoz/data/test.json') as f:
            raw_data = json.loads(f.read().lower())
        with open('multiwoz/data/entities.json') as f:
            raw_entities = json.loads(f.read().lower())
        self.get_entities(raw_entities)
        data = self.read_result_data()
        for i, row in enumerate(data):
            data[i]['response'] = self.clean(data[i]['response'])
            data[i]['generated_response'] = self.clean(data[i]['generated_response'])
        bleu_score = self.bleu_metric(data,'bleu')
        success_f1 = self.success_f1_metric(data, 'success')
        match = self.match_metric(data, 'match', raw_data=raw_data)
        self._print_dict(self.metric_dict)
        return -success_f1[0]

    def get_entities(self, entity_data):
        for k in entity_data:
            k_attr = k.split('_')[1][:-1]
            self.entities.extend(entity_data[k])
            for item in entity_data[k]:
                self.entity_dict[item] = k_attr

    def _extract_constraint(self, z):
        z = z.split()
        if 'EOS_Z1' not in z:
            s = set(z)
        else:
            idx = z.index('EOS_Z1')
            s = set(z[:idx])
        if 'moderately' in s:
            s.discard('moderately')
            s.add('moderate')
        #print(self.entities) 
        #return s
        return s.intersection(self.entities)
        #return set(z).difference(['name', 'address', 'postcode', 'phone', 'area', 'pricerange'])

    def _extract_request(self, z):
        z = z.split()
        return set(z).intersection(['address', 'postcode', 'phone', 'area', 'pricerange','food'])

    @report
    def match_metric(self, data, sub='match',raw_data=None):
        dials = self.pack_dial(data)
        match,total = 0,1e-8
        # find out the last placeholder and see whether that is correct
        # if no such placeholder, see the final turn, because it can be a yes/no question or scheduling dialogue
        for dial_id in dials:
            truth_req, gen_req = [], []
            dial = dials[dial_id]
            gen_bspan, truth_cons, gen_cons = None, None, set()
            truth_turn_num = -1
            truth_response_req = []
            for turn_num,turn in enumerate(dial):
                if 'SLOT' in turn['generated_response']:
                    gen_bspan = turn['generated_bspan']
                    gen_cons = self._extract_constraint(gen_bspan)
                if 'SLOT' in turn['response']:
                    truth_cons = self._extract_constraint(turn['bspan'])
                gen_response_token = turn['generated_response'].split()
                response_token = turn['response'].split()
                for idx, w in enumerate(gen_response_token):
                    if w.endswith('SLOT') and w != 'SLOT':
                        gen_req.append(w.split('_')[0])
                    if w == 'SLOT' and idx != 0:
                        gen_req.append(gen_response_token[idx - 1])
                for idx, w in enumerate(response_token):
                    if w.endswith('SLOT') and w != 'SLOT':
                        truth_response_req.append(w.split('_')[0])
            if not gen_cons:
                gen_bspan = dial[-1]['generated_bspan']
                gen_cons = self._extract_constraint(gen_bspan)
            if truth_cons:
                if gen_cons == truth_cons:
                    match += 1
                else:
                    pass
#                    print(gen_cons, truth_cons)
                total += 1

        return match / total

    @report
    def success_f1_metric(self, data, sub='successf1'):
        dials = self.pack_dial(data)
        tp,fp,fn = 0,0,0
        for dial_id in dials:
            truth_req, gen_req = set(),set()
            dial = dials[dial_id]
            for turn_num, turn in enumerate(dial):
                gen_response_token = turn['generated_response'].split()
                response_token = turn['response'].split()
                for idx, w in enumerate(gen_response_token):
                    if w.endswith('SLOT') and w != 'SLOT':
                        gen_req.add(w.split('_')[0])
                for idx, w in enumerate(response_token):
                    if w.endswith('SLOT') and w != 'SLOT':
                        truth_req.add(w.split('_')[0])

            gen_req.discard('name')
            truth_req.discard('name')
            for req in gen_req:
                if req in truth_req:
                    tp += 1
                else:
                    fp += 1
            for req in truth_req:
                if req not in gen_req:
                    fn += 1
        precision, recall = tp / (tp + fp + 1e-8), tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return f1, precision, recall

def metric_handler():
    parser = argparse.ArgumentParser()
    parser.add_argument('-file')
    parser.add_argument('-type')
    args = parser.parse_args()
    ev_class = None
    if args.type == 'camrest':
        ev_class = CamRestEvaluator
    elif args.type == 'kvret':
        ev_class = KvretEvaluator
    elif args.type == 'multiwoz':
        ev_class = MultiWozEvaluator
    ev = ev_class(args.file)
    ev.run_metrics()
    ev.dump()

if __name__ == '__main__':
    metric_handler()
