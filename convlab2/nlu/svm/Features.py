import itertools
import math
from collections import defaultdict

from convlab2.nlu.svm import Tuples


class lastSys(object):
    def __init__(self, config):
        pass
        
    def calculate(self, log_turn,log_input_key="batch"):
        acts = log_turn["output"]["dialog-acts"]
        out = defaultdict(float)
        for act in acts:
            act_type =act["act"]
            out[(act_type,)] += 1
            for slot,value in act["slots"]:
                if act_type == "request" :
                    out[("request", value)] += 1
                else :
                    out[(act_type,slot)] += 1
                    out[(act_type, slot, value)]+=1
                    out[(slot,value)]+=1
        return out
    
    def tuple_calculate(self, this_tuple, log_turn,log_input_key="batch"):
        return {}
        


class valueIdentifying(object):
    def __init__(self, config):
        pass
        
    def calculate(self, log_turn,log_input_key="batch"):
        return {}
    
    def tuple_calculate(self, this_tuple, log_turn,log_input_key="batch"):
        if Tuples.is_generic(this_tuple[-1]) :
            return {"<generic_value="+this_tuple[-1].value+">":1}
        return {}
    
class nbest(object):
    def __init__(self, config):
        self.max_length = 3
        if config.has_option("classifier", "max_ngram_length") :
            self.max_length = int(config.get("classifier", "max_ngram_length"))
        self.skip_ngrams = False
        if config.has_option("classifier","skip_ngrams") :
            self.skip_ngrams = config.get("classifier","skip_ngrams")=="True"
        self.skip_ngram_decay = 0.9
        if config.has_option("classifier","skip_ngram_decay") :
            self.skip_ngram_decay = float(config.get("classifier","skip_ngram_decay"))
        self.max_ngrams = 200
        if config.has_option("classifier", "max_ngrams") :
            self.max_ngrams = int(config.get("classifier", "max_ngrams"))
    
    def calculate(self, log_turn,log_input_key="batch"):
        
        asr_hyps = [(hyp["score"],hyp["asr-hyp"]) for hyp in log_turn["input"][log_input_key]["asr-hyps"]]
        asr_hyps = [(score, hyp) for score,hyp in asr_hyps if score > -100]
        # do exp of scores and normalise
        if not asr_hyps:
            return {}
        
        min_score = min([score for score, _hyp in asr_hyps])
        
        asr_hyps = [(math.exp(score+min_score), hyp) for score, hyp in asr_hyps]
        total_p = sum([score for score, _hyp in asr_hyps])
        
        if total_p == 0:
            print(asr_hyps)
        asr_hyps = [(score/total_p, hyp) for score, hyp in asr_hyps]
        
        ngrams = defaultdict(float)
        
        for p, asr_hyp in asr_hyps:
            these_ngrams = get_ngrams(asr_hyp.lower(), self.max_length, skip_ngrams=self.skip_ngrams)
            for ngram, skips in these_ngrams :
                skip_decay = 1.0
                for skip in skips:
                    skip *= (self.skip_ngram_decay**(skip-1))
                ngrams[ngram]+=p * skip_decay
        
        self.final_ngrams = ngrams.items()
        self.final_ngrams = sorted(self.final_ngrams, key = lambda x:-x[1])
        self.final_ngrams = self.final_ngrams[:self.max_ngrams]
        return ngrams

    def calculate_sent(self, log_turn,log_input_key="batch"):

        asr_hyps = [(hyp["score"],hyp["asr-hyp"]) for hyp in log_turn["asr-hyps"]]
        asr_hyps = [(score, hyp) for score,hyp in asr_hyps if score > -100]
        # do exp of scores and normalise
        if not asr_hyps:
            return {}

        min_score = min([score for score, _hyp in asr_hyps])

        asr_hyps = [(math.exp(score+min_score), hyp) for score, hyp in asr_hyps]
        total_p = sum([score for score, _hyp in asr_hyps])

        if total_p == 0:
            print(asr_hyps)
        asr_hyps = [(score/total_p, hyp) for score, hyp in asr_hyps]

        ngrams = defaultdict(float)

        for p, asr_hyp in asr_hyps:
            these_ngrams = get_ngrams(asr_hyp.lower(), self.max_length, skip_ngrams=self.skip_ngrams)
            for ngram, skips in these_ngrams :
                skip_decay = 1.0
                for skip in skips:
                    skip *= (self.skip_ngram_decay**(skip-1))
                ngrams[ngram]+=p * skip_decay

        self.final_ngrams = ngrams.items()
        self.final_ngrams = sorted(self.final_ngrams,key = lambda x:-x[1])
        self.final_ngrams = self.final_ngrams[:self.max_ngrams]
        return ngrams

    def tuple_calculate(self, this_tuple, log_turn,log_input_key="batch"):
        final_ngrams = self.final_ngrams
        # do we need to add generic ngrams?
        new_ngrams = []
        
        if Tuples.is_generic(this_tuple[-1]) :
            gvalue = this_tuple[-1]
            for ngram, score in final_ngrams:
                if gvalue.value is not None:
                    if gvalue.value.lower() in ngram :
                        new_ngram = ngram.replace(gvalue.value.lower(), "<generic_value>")
                        new_ngrams.append((new_ngram,score))

        return dict(new_ngrams)
            

def get_ngrams(sentence, max_length, skip_ngrams=False, add_tags = True):
    # return ngrams of length up to max_length as found in sentence.
    out = []
    words = sentence.split()
    if add_tags :
        words = ["<s>"]+words+["</s>"]
    if not skip_ngrams :
        for i in range(len(words)):
            for n in range(1,min(max_length+1, len(words)-i+1)): 
                this_ngram = " ".join(words[i:i+n])
                out.append((this_ngram,[]))
    else :
        for n in range(1, max_length+1):
            subsets = set(itertools.combinations(range(len(words)), n))
            for subset in subsets:
                subset = sorted(subset)
                dists = [(subset[i]-subset[i-1]) for i in range(1, len(subset))]
                out.append((" ".join([words[j] for j in subset]), dists))
            
        
    return out
        


        
class nbestLengths(object) :
    def __init__(self, config):
        pass
    def calculate(self, log_turn,log_input_key="batch"):
        out = {}
        hyps = [hyp["asr-hyp"] for hyp in log_turn["input"][log_input_key]["asr-hyps"]]
        for i, hyp in enumerate(hyps):
            out[i] = len(hyp.split())
        return out
        
    def tuple_calculate(self, this_tuple, log_turn ,log_input_key="batch"):
        return {}

class nbestScores(object) :
    def __init__(self, config):
        pass
    def calculate(self, log_turn,log_input_key="batch"):
        out = {}
        scores = [hyp["score"] for hyp in log_turn["input"][log_input_key]["asr-hyps"]]
        for i, score in enumerate(scores):
            out[i] = score
        return out
        
    def tuple_calculate(self, this_tuple, log_turn,log_input_key="batch" ):
        return {}


class cnet(object):
    def __init__(self, config):
        import json
        self.slots_enumerated = json.loads(config.get("grammar", "slots_enumerated"))
        self.max_length = 3
        if config.has_option("classifier", "max_ngram_length") :
            self.max_length = int(config.get("classifier", "max_ngram_length"))
        self.max_ngrams = 200
        if config.has_option("classifier", "max_ngrams") :
            self.max_ngrams = int(config.get("classifier", "max_ngrams"))
        self.final_ngrams = None
        self.last_parse = None
            
    def calculate(self, log_turn,log_input_key="batch"):
        if self.last_parse == log_turn["input"]["audio-file"] :
            return dict([(ng.string_repn(), ng.score()) for ng in self.final_ngrams])
        cnet = log_turn["input"][log_input_key]["cnet"]
        self.final_ngrams = get_cnngrams(cnet,self.max_ngrams, self.max_length)
        self.last_parse = log_turn["input"]["audio-file"]
        return dict([(ng.string_repn(), ng.score()) for ng in self.final_ngrams])
    
    
    def tuple_calculate(self, this_tuple, log_turn,log_input_key="batch"):
        final_ngrams = self.final_ngrams
        # do we need to add generic ngrams?
        new_ngrams = []
        if Tuples.is_generic(this_tuple[-1]) :
            gvalue = this_tuple[-1]
            for ngram in final_ngrams:
                new_ngram = cn_ngram_replaced(ngram, gvalue.value.lower(), "<generic_value>")
                if new_ngram != False:
                    new_ngrams.append(new_ngram)

        return dict([(ng.string_repn(), ng.score()) for ng in new_ngrams])
        
    
    
def get_cnngrams(cnet, max_ngrams, max_length):
    active_ngrams = []
    finished_ngrams = []
    threshold = -5
    for sausage in cnet:
        new_active_ngrams = []
        
        for arc in sausage['arcs']:
            if arc['score'] < threshold :
                continue
            this_ngram = cnNgram(arc['word'].lower(), arc['score'])
            for ngram in active_ngrams:
                
                new_ngram = ngram + this_ngram
                if len(new_ngram) < max_length :
                    new_active_ngrams.append(new_ngram)
                    # don't add ones ending in !NULL to finished
                    # as they need to end on a real word
                    # otherwise HELLO, HELLO !NULL, HELLO !NULL !NULL ...will accumulate
                    if arc['word'] != "!null" :
                        finished_ngrams.append(new_ngram)
                elif arc['word'] != "!null" :
                    
                    finished_ngrams.append(new_ngram)
                    
            if this_ngram:
                new_active_ngrams.append(this_ngram)
                finished_ngrams.append(this_ngram)
        
        active_ngrams = cn_ngram_prune((new_active_ngrams[:]), int(1.5*max_ngrams))
       
    return  cn_ngram_prune(cn_ngram_merge(finished_ngrams), max_ngrams)    

    
class cnNgram(object):
    
    def __init__(self, words, logp, delta=0):
        if not isinstance(words, type([])) :
            words = words.split()
        self.words = words
        self.logp = logp
        self.active = True
        self.replacement_length_delta = delta
    
    
    def logscore(self):
        return self.logp / len(self)
    
    def score(self):
        return math.exp(self.logscore())
    
    
    def __add__(self, other):
        return cnNgram(self.words + other.words, self.logp+other.logp)
    
    def __repr__(self, ):
        return "%s : %.7f" % (" ".join(self.words), self.logp)
    
    def __len__(self):
        return len([x for x in self.words if x != "!null"]) + self.replacement_length_delta
        
    def word_list(self, ):
        return [word for word in self.words if word != "!null"]
    
    def string_repn(self, ):
        return " ".join(self.word_list())
    
    
    def __hash__(self):
        # means sets work
        string =  self.string_repn()
        return string.__hash__()
    
    def __eq__(self, other):
        return self.string_repn() == other.string_repn()
  
def cn_ngram_merge(ngrams) :  
    # merge a list of ngrams
    merged = {}
    for ngram in ngrams:
        if ngram not in merged :
            merged[ngram] = ngram.logp
        else : 
            merged[ngram] = math.log( math.exp(ngram.logp) + math.exp(merged[ngram])  )
            
    new_ngrams = []
    for ngram in merged:
        ngram.logp = merged[ngram]
        new_ngrams.append(ngram)
    return new_ngrams

def cn_ngram_prune(ngrams, n):
    if len(ngrams) < n :
        return ngrams
    ngrams.sort(key=lambda x:-x.logscore())
    return ngrams[:n]

def cn_ngram_replaced(ngram, searchwords, replacement):
    words = ngram.word_list()
    searchwords = searchwords.split()
    new_words = []
    found = False
    i=0
    while i < len(words):
        if words[i:i+len(searchwords)] == searchwords:
            new_words.append(replacement)
            found = True
            i+=len(searchwords)
        else :
            new_words.append(words[i])
            i+=1
    if not found :
        return False
    out = cnNgram(new_words, ngram.logp, delta=len(searchwords) - 1)
    return out
    


    

if __name__ == '__main__':
    cn = [
        {"arcs":[{"word":"<s>","score":0.0}]},
        {"arcs":[{"word":"hi","score":0.0}]},
        {"arcs":[{"word":"there","score":-math.log(2)}, {"word":"!null","score":-math.log(2)}]},
        {"arcs":[{"word":"how","score":0.0}]},
        {"arcs":[{"word":"are","score":0.0}]},
        {"arcs":[{"word":"you","score":0.0}]},
        {"arcs":[{"word":"</s>","score":0.0}]}
        
    ]
    final_ngrams = get_cnngrams(cn,200,3)
    print(dict([(ng.string_repn(), ng.score()) for ng in final_ngrams]))
    import configparser, json, Tuples
    config = configparser.ConfigParser()
    config.read("output/experiments/feature_set/run_1.cfg")
    nb = cnet(config)
    log_file = json.load(open("corpora/data/Mar13_S2A0/voip-318851c80b-20130328_224811/log.json"))
    log_turn = log_file["turns"][2]
    print(nb.calculate(
        log_turn
    ))
    tup = ("inform", "food", Tuples.genericValue("food", "modern european"))
    print(nb.tuple_calculate(tup, log_turn))
