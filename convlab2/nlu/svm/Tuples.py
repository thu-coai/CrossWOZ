import math
import os
import re
import json

from convlab2.nlu.svm import sutils


class tuples(object):
    def __init__(self, config):
        self.acts = json.loads(config.get("grammar", "acts"))
        # self.nonempty_acts = json.loads(config.get("grammar", "nonempty_acts"))
        # self.nonfull_acts = [act for act in self.acts if act not in self.nonempty_acts]
        
        rootpath=os.path.dirname(os.path.abspath(__file__))
        # if "semi" not in rootpath:
        #     rootpath+="/semi/CNetTrain/"
        # else:
        #     rootpath+="/CNetTrain/"
        self.ontology = json.load(
            open(rootpath+'/'+config.get("grammar", "ontology"))
        )
        
        self.slots_informable = self.ontology["informable"]
        self.slots =  self.ontology["requestable"]
        
        self.slots_enumerated = json.loads(config.get("grammar", "slots_enumerated"))
        self.config = config
        self.all_tuples = self._getAllTuples()
        self.max_active = 10
        if config.has_option("decode","max_active_tuples") :
            self.max_active = int(config.get("decode","max_active_tuples"))
            
        self.tail_cutoff = 0.001
        if config.has_option("decode","tail_cutoff") :
            self.tail_cutoff = float(config.get("decode","tail_cutoff"))
        self.log_tail_cutoff = math.log(self.tail_cutoff)

    def uactsToTuples(self, uacts):
        out = []
        for uact in uacts:
            act =uact["act"]
            if uact["slots"] == [] :
                out.append((act,))
            for slot,value in uact["slots"]:
                if act == "request" :
                    out.append(("request", value))
                elif slot in self.slots_informable or slot == "this":
                    if slot in self.slots_enumerated or slot == "this":
                        out.append((act,slot,value))
                    else :
                        out.append((act,slot, genericValue(slot, value)))
        return out

    def _getAllTuples(self):
        out = []
        for slot in self.slots:
            out.append(("request", slot))
        for x in self.ontology["all_tuples"]:
            slot = x[1]
            if slot in self.slots_enumerated:
                out.append(tuple(x))
            else:
                out.append((x[0], slot, genericValue(slot)))
            out.append((x[0], slot, "do n't care"))
        return list(set(out))
    
    def activeTuples(self, log_turn):
        asr_hyps = log_turn["input"]["live"]["asr-hyps"]
        out = []
        asr_hyps_conc = ", ".join([asr_hyp['asr-hyp'].lower() for asr_hyp in asr_hyps])
        for this_tuple in self.all_tuples:
            if is_generic(this_tuple[-1]) :
                # this is a generic value
                act, slot, gvalue = this_tuple
                for value in self.ontology["informable"][this_tuple[-2]]:
                    if value.lower() in asr_hyps_conc :
                        out.append((act, slot, genericValue(slot, value)))
                if slot == 'Phone':
                    matchObj = re.search(r'\d{11}',asr_hyps_conc)
                    if matchObj:
                        out.append((act, slot, genericValue(slot, matchObj.group())))
                elif slot == 'Ticket':
                    matchObj = re.search(r'([0-9.]*?) (GBP|gbp)', asr_hyps_conc)
                    if matchObj:
                        out.append((act, slot, genericValue(slot, matchObj.group())))
                elif slot == 'Ref':
                    matchObj = re.search(r'reference number is(\s*?)([a-zA-Z0-9]+)', asr_hyps_conc)
                    if matchObj:
                        out.append((act, slot, genericValue(slot, matchObj.group(2))))
                elif slot == 'Time' or slot == 'Arrive' or slot == 'Leave':
                    matchObj = re.search(r'\d+?:\d\d', asr_hyps_conc)
                    if matchObj:
                        out.append((act, slot, genericValue(slot, matchObj.group(0))))
            else :
                out.append(this_tuple)
        return out

    def activeTuples_sent(self, log_turn):
        asr_hyps = log_turn["asr-hyps"]
        out = []
        asr_hyps_conc = ", ".join([asr_hyp['asr-hyp'].lower() for asr_hyp in asr_hyps])
        for this_tuple in self.all_tuples:
            if is_generic(this_tuple[-1]) :
                # this is a generic value
                act, slot, gvalue = this_tuple
                if slot not in self.ontology["informable"]:
                    continue
                for value in self.ontology["informable"][this_tuple[-2]]:
                    if value.lower() in asr_hyps_conc :
                        out.append((act, slot, genericValue(slot, value)))
                if slot == 'Phone':
                    matchObj = re.search(r'\d{11}',asr_hyps_conc)
                    if matchObj:
                        out.append((act, slot, genericValue(slot, matchObj.group())))
                elif slot == 'Ticket':
                    matchObj = re.search(r'([0-9.]*?) (GBP|gbp)', asr_hyps_conc)
                    if matchObj:
                        out.append((act, slot, genericValue(slot, matchObj.group())))
                elif slot == 'Ref':
                    matchObj = re.search(r'reference number is(\s*?)([a-zA-Z0-9]+)', asr_hyps_conc)
                    if matchObj:
                        out.append((act, slot, genericValue(slot, matchObj.group(2))))
                elif slot == 'Time' or slot == 'Arrive' or slot == 'Leave':
                    matchObj = re.search(r'\d+?:\d\d', asr_hyps_conc)
                    if matchObj:
                        out.append((act, slot, genericValue(slot, matchObj.group(0))))
            else :
                out.append(this_tuple)
        return out

    def distributionToNbest(self, tuple_distribution):
        # convert a tuple distribution to an nbest list
        tuple_distribution = tuple_distribution.items()
        output = []
        ps = [p for _t,p in tuple_distribution]
        eps = 0.00001
        tuple_distribution = [(t, math.log(max(eps,p)), math.log(max(eps, 1-p))) for t,p in tuple_distribution if p > 0]
        tuple_distribution = sorted(tuple_distribution,key=lambda x:-x[1])
        # prune
        tuple_distribution = tuple_distribution[:self.max_active]
       
        n = len(tuple_distribution)
        powerset = sutils.powerset(range(n))
        acts = []
        for subset in powerset:
            act = []
            score = 0
            for i in range(n):
                this_tuple, logp, log1_p = tuple_distribution[i] 
                if i in subset :
                    act.append(this_tuple)
                    score += logp
                else :
                    score += log1_p
            if (score> self.log_tail_cutoff or not act) and makes_valid_act(act) :
                acts.append((act,score))
                if not act:
                    null_score = score
        acts = sorted(acts,key=lambda x:-x[1])
        
        acts = acts[:10]
        found_null = False
        for act,score in acts:
            if not act:
                found_null = True
                break
        if not found_null :
            acts.append(([], null_score))
        
        #normalise
        acts = [(act,math.exp(logp)) for act,logp in acts]
        totalp = sum([p for act,p in acts])
        acts = [{"slu-hyp":[tuple_to_act(a) for a in act],"score":p/totalp} for act,p in acts]
        return acts


def tuple_to_act(t) :
    if len(t) == 1 :
        return {"act":t[0],"slots":[]}
    elif len(t) == 2:
        assert t[0] == "request"
        return {"act": "request", "slots": [["slot", t[1]]]}
    return {"act": t[0], "slots": [[t[1], t[2]]]}


def makes_valid_act(tuples):
    # check if uacts is a valid list of tuples
    # - can't affirm and negate
    # - can't deny and inform same thing
    # - can't inform(a=x) inform(a=y) if x!=u
    singles = [t for t in tuples if len(t)==1]
    if ("affirm",) in tuples and ("negate",) in tuples :
        return False
    triples = [t for t in tuples if len(t)==3]
    informed = [(slot, value) for act,slot,value in triples if act=="inform"]
    denied   = [(slot, value) for act,slot,value in triples if act=="deny"  ]
    for s,v in informed:
        if (s,v) in denied:
            return False
    informed_slots = [slot for slot, _value in informed]
    if len(informed_slots) != len(set(informed_slots)) :
        return False
    return True


def actual_value(value):
    try:
        return value.value
    except AttributeError:
        return value

                  
class genericValue(object):
    # useful class to use to represent a generic value
    # x = genericValue("food")
    # y = genericValue("food","chinese")
    # z = genericValue("food","indian")
    # x == y
    # y in [x]
    # y.value != z.value
    
    def __init__(self, slot, value=None):
        self.slot = slot
        self.value = value
        
    def __str__(self):
        paren = "" 
        if self.value is not None :
            paren = " (%s)" % self.value
        return ("(generic value for %s"% self.slot) + paren + ")"
    
    def __repr__(self):
        return self.__str__()
    
    def __eq__(self, other):
        try:
            return self.slot == other.slot
        except AttributeError :
            return False
    
    def __hash__(self):
        return self.slot.__hash__()
    
    
def is_generic(value):
    return not isinstance(value, str)


def generic_to_specific(tup) :
    if len(tup) == 3 :
        act,slot,value = tup
        value = actual_value(value)
        return (act,slot,value)
    return tup
