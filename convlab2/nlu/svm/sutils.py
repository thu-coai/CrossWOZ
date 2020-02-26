import imp
import os


def import_class(cl):
    d = cl.rfind(".")
    classname = cl[d+1:len(cl)]
    m = __import__(cl[0:d], globals(), locals(), [classname])
    return getattr(m, classname)

from itertools import chain, combinations

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def svm_to_libsvm(model, labels=None) :
    # convert an sklearn model object into a file in the format of LIBSVM's sparse SVMs
    # (actually return the files lines in an array)
    lines = []
    n_classes = model.coef_.shape[0]+1
    total_n_SV, n_feats= model.support_vectors_.shape
    n_SV = model.n_support_
    
    SV = model.support_vectors_
    
    dual_coef = model.dual_coef_.todense()
    b = model.intercept_
    
    probA = model.probA_
    probB = model.probB_
    
    lines.append("svm_type")
    lines.append("nr_class %i" % n_classes)
    lines.append("total_sv %i" % total_n_SV)
    
    lines.append("rho "+" ".join(["%.12f" % -c for c in b]))
    
    if labels is None:
        labels = map(str, range(n_classes))
    lines.append("label " +  " ".join(labels))
            
    lines.append("probA "+" ".join(["%.12f" % v for v in probA]))
    lines.append("probB "+" ".join(["%.12f" % v for v in probB]))
    
    lines.append("nr_sv "+" ".join(["%i" % v for v in n_SV]))
    
    lines.append("SV")
    SV = SV.tocsc()
    
    for i in range( total_n_SV) :
            # coefs are in the jth column of coef
        this_line = ""
        for c in dual_coef[:,i] :
            this_line += ("%.12f " % c)
        sv = SV[i,:].tocoo()
        
        for j,v in zip(sv.col, sv.data) :
            this_line += ("%i:%.12f " % (j,v))
        lines.append(this_line)
    return lines
