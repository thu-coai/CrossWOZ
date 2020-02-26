from copy import deepcopy
import numpy as np
import torch

def padding(old, l):
    """
    pad a list of different lens "old" to the same len "l"
    """
    new = deepcopy(old)
    for i, j in enumerate(new):
        new[i] += [0] * (l - len(j))
        new[i] = j[:l]
    return new

def padding_data(data):
    batch_goals, batch_usrdas, batch_sysdas = deepcopy(data)
    
    batch_input = {}
    posts_length = []
    posts = []
    origin_responses = []
    origin_responses_length = []
    goals_length = []
    goals = []
    terminated = []

    ''' start padding '''
    max_goal_length = max([len(sess_goal) for sess_goal in batch_goals]) # G
    sentence_num = [len(sess) for sess in batch_sysdas]
    # usr begins the session
    max_sentence_num = max(max(sentence_num)-1, 1) # S
        
    # goal & terminated
    for i, l in enumerate(sentence_num):
        goals_length += [len(batch_goals[i])] * l
        goals_padded = list(batch_goals[i]) + [0] * (max_goal_length - len(batch_goals[i]))
        goals += [goals_padded] * l
        terminated += [0] * (l-1) + [1]
        
    # usr
    for sess in batch_usrdas:
        origin_responses_length += [len(sen) for sen in sess]
    max_response_length = max(origin_responses_length) # R
    for sess in batch_usrdas:
        origin_responses += padding(sess, max_response_length)
        
    # sys
    for sess in batch_sysdas:
        sen_length = [len(sen) for sen in sess]
        for j in range(len(sen_length)):
            if j == 0:
                posts_length.append(np.array([1] + [0] * (max_sentence_num - 1)))
            else:
                posts_length.append(np.array(sen_length[:j] + [0] * (max_sentence_num - j)))
    posts_length = np.array(posts_length)
    max_post_length = np.max(posts_length) # P
    for sess in batch_sysdas:
        sen_padded = padding(sess, max_post_length)
        for j, sen in enumerate(sess):
            if j == 0:
                post_single = np.zeros([max_sentence_num, max_post_length], np.int)
            else:
                post_single = posts[-1].copy()
                post_single[j-1, :] = sen_padded[j-1]
            
            posts.append(post_single)
    ''' end padding '''

    batch_input['origin_responses'] = torch.LongTensor(origin_responses) # [B, R]
    batch_input['origin_responses_length'] = torch.LongTensor(origin_responses_length) #[B]
    batch_input['posts_length'] = torch.LongTensor(posts_length) # [B, S]
    batch_input['posts'] = torch.LongTensor(posts) # [B, S, P]
    batch_input['goals_length'] = torch.LongTensor(goals_length) # [B]
    batch_input['goals'] = torch.LongTensor(goals) # [B, G]
    batch_input['terminated'] = torch.Tensor(terminated) # [B]
    
    return batch_input

def kl_gaussian(argu):
    recog_mu, recog_logvar, prior_mu, prior_logvar = argu
    # find the KL divergence between two Gaussian distribution
    loss = 1.0 + (recog_logvar - prior_logvar)
    loss -= (recog_logvar.exp() + torch.pow(recog_mu - prior_mu, 2)) / prior_logvar.exp()
    kl_loss = -0.5 * loss.sum(dim=1)
    avg_kl_loss = kl_loss.mean()
    return avg_kl_loss

def capital(da):
    for d_i in da:
        pairs = da[d_i]
        for s_v in pairs:
            if s_v[0] != 'none':
                s_v[0] = s_v[0].capitalize()
    
    da_new = {}
    for d_i in da:
        d, i = d_i.split('-')
        if d != 'general':
            d = d.capitalize()
            i = i.capitalize()
        da_new['-'.join((d, i))] = da[d_i]
        
    return da_new
