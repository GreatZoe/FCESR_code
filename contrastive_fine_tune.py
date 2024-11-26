from __future__ import absolute_import, division, print_function

import sys
import os
import argparse
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Bernoulli

import numpy as np
from itertools import chain
import pandas as pd
import pickle
from functools import reduce
import warnings
warnings.filterwarnings('ignore')



from rl_env import BatchKGEnvironment 

from NARM import NARM
from SRGNN import SRGNN
from GRU4REC import GRU4REC
from GCSAN import GCSAN
from bert_modules.bert import BERT
from utils import *
import math
from sklearn.model_selection import train_test_split
import itertools

logger = None

base_model='srgnn' #srgnn/narm/gru4rec/GCSAN/bert4rec

class ACDataLoader(object):
    def __init__(self, data, batch_size):
        ## data:dataframe uid:user_id session:list  target:item_id
        self.data=data
        self.num_sessions = len(data)
        self.batch_size = batch_size
        self.reset()

    def reset(self):
        self._start_idx = 0
        self._has_next = True

    def has_next(self):
        return self._has_next

    def get_batch(self,kz):
        if not self._has_next:
            return None
          
        
        end_idx = min(self._start_idx + self.batch_size, self.num_sessions)

        batch_sessions = self.data['session'].iloc[self._start_idx:end_idx].tolist()

        batch_target=self.data['label'].iloc[self._start_idx:end_idx] # target
        batch_len=self.data['length'].iloc[self._start_idx:end_idx].tolist()

        batch_sids = self.data['session_id'].iloc[self._start_idx:end_idx].tolist()
    
        padded_sesss_item = torch.zeros(end_idx-self._start_idx, max(batch_len)).long()
        #print('==================================',len(padded_sesss_item))
        for i,se in enumerate(batch_sessions):
            padded_sesss_item[i, :batch_len[i]] = torch.LongTensor(se)
        
        self._has_next = self._has_next and end_idx < self.num_sessions
        self._start_idx = end_idx

        return padded_sesss_item.transpose(0, 1), batch_len, torch.tensor(batch_target.tolist()).long(),  batch_sids  


def train(args):
    policy_file = args.dataset + '/'+base_model+'/recommend_model.ckpt'
    
    path='./data/'+args.dataset + '/'
    train_data =  pd.read_csv( path + 'train_data.csv')
    val_data =  pd.read_csv(path + 'valid_data.csv')
    test_data =  pd.read_csv(path + 'test_data.csv')
    
    train_data['session'] = train_data['session'].apply(lambda x: eval(x))
    test_data['session'] = test_data['session'].apply(lambda x: eval(x))
    val_data['session'] = val_data['session'].apply(lambda x: eval(x))

    if args.dataset == 'ml-1m':
        all_sessions = train_data['session'].tolist() + test_data['session'].tolist() + val_data['session'].tolist()
        unique_items = set(item for session in all_sessions for item in session)
        max_item_id = max(unique_items)
        n_items = max_item_id + 1
        print('n_items',n_items)
    else:
        
        n_items=len(np.load(path+'item_dict.npy',allow_pickle=True).item())
        print('n_items',n_items)
        
    if base_model=='srgnn':
        session=SRGNN(args.embed_dim,n_items).to(args.device)
    elif base_model=='narm':
        session=NARM(n_items, int(args.embed_dim/2), args.embed_dim, args.batch_size).to(args.device)
    elif base_model=='gru4rec':
        session=GRU4REC(args.embed_dim,args.embed_dim,n_items).to(args.device)
    elif base_model=='GCSAN':
        session=GCSAN(args.embed_dim,n_items).to(args.device)
    elif base_model=='bert4rec':
        session=BERT(args, n_items).to(args.device)
    session.load_state_dict(torch.load(policy_file, map_location=torch.device('cpu')))
    print("Model loaded from " + policy_file)
    
    
    with open(args.dataset + '/explanations/'+base_model+'/factual_explanation.pkl', 'rb') as file:
        fe_explanations = pickle.load(file)
    with open(args.dataset +'/explanations/'+base_model+'/counterfactual_explanation.pkl', 'rb') as file:
        cfe_explanations = pickle.load(file)


    fe_explanations_df = pd.DataFrame({
    'session': fe_explanations, 
    'label': train_data['label'], 
    'session_id': train_data['session_id']})
    
    cf_explanations_df = pd.DataFrame({
    'session': cfe_explanations, 
    'label': train_data['label'], 
    'session_id': train_data['session_id']})
    
    fe_explanations_df['length'] = fe_explanations_df['session'].apply(lambda x: len(x))
    cf_explanations_df['length'] = cf_explanations_df['session'].apply(lambda x: len(x))

    train_dataloader = ACDataLoader(train_data, args.batch_size)
    val_dataloader = ACDataLoader(val_data, args.batch_size) 
    test_dataloader = ACDataLoader(test_data, 100)
    
    pos_dataloader = ACDataLoader(fe_explanations_df, args.batch_size)
    neg_dataloader = ACDataLoader(cf_explanations_df, args.batch_size)
    
    optimizer = optim.Adam(params=chain(session.parameters()), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    total_losses = []
    step = 0
    session_dic = {}
    last_hr = [0.0,0.0,0.0]
    last_ndcg = [0.0,0.0,0.0]
    hr, ndcg = test(args,session, test_dataloader, 'test')
    logger.info(f" original result: hr: {hr}, ndcg: {ndcg}")
    best_hr, best_ndcg = test(args,session, val_dataloader,'val')

    for epoch in range(1, args.epochs + 1):
        kz = 0
        session.train()
        train_dataloader.reset()
        pos_dataloader.reset()
        neg_dataloader.reset()
        while train_dataloader.has_next():
            optimizer.zero_grad()
            batch_sessions, seq_length, targets,  batch_sids = train_dataloader.get_batch(kz)
            neg_batch_sessions, neg_seq_length, neg_targets, neg_batch_sids = neg_dataloader.get_batch(kz)
            pos_batch_sessions, pos_seq_length, pos_targets, pos_batch_sids = pos_dataloader.get_batch(kz)
            
            session_rep,score = get_session_representation(base_model, batch_sessions, seq_length, args, session)

            neg_sesssion_rep,neg_score = get_session_representation(base_model, neg_batch_sessions, neg_seq_length, args, session)
            pos_sesssion_rep,pos_score = get_session_representation(base_model, pos_batch_sessions, pos_seq_length, args, session)
            
            dict1 = dict(zip(batch_sids, session_rep.tolist()))
            session_dic.update(dict1)
            batch_sessions=batch_sessions.transpose(0,1)
            
            lr = args.lr * max(1e-4, 1.0 - float(step) / (args.epochs * len(train_data) / args.batch_size))
            for pg in optimizer.param_groups:
                    pg['lr'] = lr
            if neg_score.numel() == 0:
                neg_score = torch.zeros_like(score)      
                neg_sesssion_rep = torch.zeros_like(session_rep)
                
            rec_loss = criterion(score.to(args.device),targets.to(args.device))

            pos_loss = loss_pos(score, pos_score, session_rep, neg_sesssion_rep, pos_sesssion_rep)
            loss = rec_loss + args.lamb1 * pos_loss
            loss.backward()
            optimizer.step()
            
            total_losses.append(loss.item())
            step += 1

            # Report performance
            if step > 0 and step % 10 == 0:
                avg_loss = np.mean(total_losses)
                total_losses = []
                logger.info(
                    'epoch/step={:d}/{:d}'.format(epoch, step) +
                    ' | loss={:.5f}'.format(avg_loss))

        best_hr, best_ndcg = test(args,session,val_dataloader,'val')
        
        if best_hr[2] > last_hr[2] and best_ndcg[1] > last_ndcg[1]:
            last_hr = best_hr
            last_ndcg = best_ndcg
            last_hr, last_ndcg = test(args,session,test_dataloader,'test')

    return  last_hr, last_ndcg



def test(args,session,test_dataloader,type):

    session.eval()

    with torch.no_grad():

        test_dataloader.reset()
        
        all_targets = []
        hits5, ndcgs5 = [], []
        hits10, ndcgs10 = [], []
        hits20, ndcgs20 = [], []
        session_dic = {}
        kz=0
        best_hr_5, best_ndcg_5,best_hr_10, best_ndcg_10,best_hr_20, best_ndcg_20 = 0,0,0,0,0,0
        while test_dataloader.has_next():
            batch_sessions, seq_length, targets,  batch_sids = test_dataloader.get_batch(kz)

            if base_model=='srgnn' or base_model=='GCSAN':
                sequences=batch_sessions.transpose(0,1)
                x = []
                batch=[]
                senders,receivers=[],[]
                i,j=0,0
                for sequence in sequences:
                    sender = []
                    nodes = {}
                    for node in sequence:
                        if node not in nodes:
                            nodes[node] = i
                            x.append([node])
                            i += 1
                        batch.append(j)
                        sender.append(nodes[node])
                    j += 1
                    receiver = sender[:]
                    if len(sender) != 1:
                        del sender[-1]
                        del receiver[0]
                    senders.extend(sender)
                    receivers.extend(receiver)
                edge_index = torch.tensor([senders, receivers], dtype=torch.long)
                x = torch.tensor(x, dtype=torch.long)
                batch = torch.tensor(batch, dtype=torch.long)
                session_rep,score=session(x,edge_index,batch)
            elif base_model=='narm':
                session_rep,score=session(batch_sessions.to(args.device),seq_length) 
            elif base_model=='gru4rec':
                session_rep,score=session(batch_sessions.to(args.device),seq_length)
            elif base_model=='bert4rec':
                if batch_sessions.shape[0]<args.bert_max_len:
                    batch_sessions=torch.cat((batch_sessions,torch.zeros(args.bert_max_len-batch_sessions.shape[0],batch_sessions.shape[1],dtype=torch.long)))           
                session_rep, score=session(batch_sessions.transpose(0,1).to(args.device))
  
            dict1 = dict(zip(batch_sids, session_rep.tolist()))
            session_dic.update(dict1)
            batch_sessions=batch_sessions.transpose(0,1)
            kz=1
            all_targets.extend(targets)
            preds = score
            topk_preds = torch.topk(preds, 5, dim=1)[1]
            for i, target in enumerate(targets):
                hits5.append(get_hit_ratio(args,topk_preds[i], target))
                ndcgs5.append(get_ndcg(args,topk_preds[i], target))
            
            topk_preds = torch.topk(preds, 10, dim=1)[1]
            for i, target in enumerate(targets):
                hits10.append(get_hit_ratio(args,topk_preds[i], target))
                ndcgs10.append(get_ndcg(args,topk_preds[i], target))
            topk_preds = torch.topk(preds, 20, dim=1)[1]

            for i, target in enumerate(targets):
                hits20.append(get_hit_ratio(args,topk_preds[i], target))
                ndcgs20.append(get_ndcg(args,topk_preds[i], target))
    best_hr_5 = np.mean(hits5)
    best_ndcg_5 = np.mean(ndcgs5)
    
    best_hr_10 = np.mean(hits10)
    best_ndcg_10 = np.mean(ndcgs10)
    
    best_hr_20 = np.mean(hits20)
    best_ndcg_20 = np.mean(ndcgs20)
    
    if type == 'test':
        print("----------test----------")
        print("top-5-best-hit", best_hr_5)
        print("top-5-best-ndcg",best_ndcg_5)
        print("top-10-best-hit", best_hr_10)
        print("top-10--best-ndcg",best_ndcg_10)
        print("top-20-best-hit", best_hr_20)
        print("top-20-best-ndcg",best_ndcg_20)
        print("--------------------")
    return [best_hr_5, best_hr_10,best_hr_20], [best_ndcg_5,best_ndcg_10, best_ndcg_20]



def get_session_representation(base_model, batch_sessions, seq_length, args, session):
    if base_model in ['srgnn', 'GCSAN']:
        sequences = batch_sessions.transpose(0, 1)  # Convert to 2D matrix
        x, batch, senders, receivers = [], [], [], []
        for i, sequence in enumerate(sequences):
            sender, nodes = [], {}
            for node in sequence:
                if node not in nodes:
                    nodes[node] = len(x)  # index of the node
                    x.append([node])
                batch.append(i)
                sender.append(nodes[node])
            receiver = sender[:]
            if len(sender) > 1:
                sender = sender[:-1]
                receiver = receiver[1:]
            senders.extend(sender)
            receivers.extend(receiver)
        x = torch.tensor(x, dtype=torch.long).to(args.device)
        edge_index = torch.tensor([senders, receivers], dtype=torch.long).to(args.device)
        batch = torch.tensor(batch, dtype=torch.long).to(args.device)
        session_rep, scores = session(x, edge_index, batch)
    elif base_model in ['narm', 'gru4rec']:
        # Directly use batch_sessions and seq_length
        session_rep, scores = session(batch_sessions.to(args.device), seq_length)
    elif base_model == 'bert4rec':
        # Handle potential padding for BERT-based model
        if batch_sessions.shape[0] < 5:
            padding = torch.zeros((5 - batch_sessions.shape[0], batch_sessions.shape[1]), dtype=torch.long)
            batch_sessions = torch.cat((batch_sessions, padding), dim=0)
        session_rep, scores = session(batch_sessions.transpose(0, 1).to(args.device))
    return session_rep, scores


def contrastive_loss(z_i, z_j, z_k, tau=0.5):
    numerator = torch.exp(torch.cosine_similarity(z_i, z_j) / tau)
    denominator = numerator + torch.sum(torch.exp(torch.cosine_similarity(z_i, z_k) / tau))
    loss = -torch.log(numerator / denominator)
    return loss


def loss_neg(score, score_neg):
        epsilon = 1e-8
        neg_loss = -torch.log(torch.sigmoid(score - score_neg)+epsilon).mean()

        return neg_loss
    
def loss_pos(score, score_pos,session_rep, neg_sesssion_rep, pos_sesssion_rep):
    epsilon = 1e-8
    predict_pos = torch.topk(score_pos, 1, dim=1)[1]
    predict = torch.topk(score, 1, dim=1)[1]
    equal_indices = torch.nonzero(torch.eq(predict_pos, predict), as_tuple=False)
    filtered_scores = session_rep[equal_indices[:, 0], :]
    filtered_scores_pos = pos_sesssion_rep[equal_indices[:, 0], :]
    filtered_scores_neg = neg_sesssion_rep[equal_indices[:, 0], :]

    loss = 0.0
    num_pairs = len(filtered_scores_pos)
    
    if filtered_scores_pos.numel() == 0 or filtered_scores_neg.numel() == 0:
        return torch.tensor(0.0)
    positive_similarity = F.cosine_similarity(filtered_scores, filtered_scores_pos, dim=1)
    
    negative_similarity = F.cosine_similarity(filtered_scores, filtered_scores_neg, dim=1)
    
    negative_similarity_sum = torch.exp(negative_similarity)
    positive_similarity_exp = torch.exp(positive_similarity)
    
    contrastive_loss = -torch.log(positive_similarity_exp / (positive_similarity_exp + negative_similarity_sum + epsilon))
    loss = contrastive_loss.mean()

    return loss

    


def get_hit_ratio(args,preds, target):
    target = target.to(args.device)
    preds = preds.to(args.device)
    if target in preds:
        return 1
    return 0


def get_ndcg(args,preds, target):
    target = target.to(args.device)
    preds = preds.to(args.device)
    for i in range(len(preds)):
        if preds[i] == target:
            return 1 / math.log(i+2,2)
    return 0

    
def main():
    boolean = lambda x: (str(x).lower() == 'true')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=ML1M, help='One of {cell, beauty, cd}')
    parser.add_argument('--name', type=str, default='train_agent', help='directory name.')
    parser.add_argument('--seed', type=int, default=123, help='random seed.')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device.')
    parser.add_argument('--epochs', type=int, default=20, help='Max number of epochs.') 
    parser.add_argument('--batch_size', type=int, default=256, help='batch size.')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate.')

    parser.add_argument('--gamma', type=float, default=0.99, help='reward discount factor.')
    parser.add_argument('--hidden', type=int, nargs='*', default=400, help='number of samples')
    parser.add_argument('--embed_dim', type=int, default=400, help='item embedding size of NARM') 
    parser.add_argument('--state_dim', type=int, default=400, help='dimension of state vector')
    parser.add_argument('--add_products', type=boolean, default=True, help='Add predicted products up to 10')
 
    parser.add_argument('--lamb1', type=float, default= 15, help='rate of contrastive')  
    parser.add_argument('--lamb2', type=float, default=10, help='rate of contrastive')  

    parser.add_argument('--bert_max_len', type=int, default=100, help='Length of sequence for bert')
    parser.add_argument('--bert_num_blocks', type=int, default=2, help='Number of transformer layers')
    parser.add_argument('--bert_num_heads', type=int, default=5, help='Number of heads for multi-attention')
    parser.add_argument('--bert_dropout', type=float, default=0.1, help='Dropout probability to use throughout the model')
    parser.add_argument('--bert_hidden_units', type=int, default=400, help='Size of hidden vectors (d_model)')
    
    args = parser.parse_args()
    args.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

    args.log_dir = '{}/{}'.format(TMP_DIR[args.dataset], args.name)
    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)

    global logger
    logger = get_logger(args.log_dir + '/fine_tune.txt')
    logger.info(args)

    # set_random_seed(args.seed)
    hr, ndcg = train(args)
    print("result: ",args.lr,args.lamb1,': hr:',hr,'ndcg:', ndcg)

if __name__ == '__main__':
    main()