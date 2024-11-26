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
from torch.distributions import Categorical

import numpy as np
from itertools import chain
import pandas as pd
import pickle
from functools import reduce
import warnings
warnings.filterwarnings('ignore')


from NARM import NARM
from SRGNN import SRGNN
from GRU4REC import GRU4REC
from GCSAN import GCSAN
from bert_modules.bert import BERT
from utils import *
import math
from sklearn.model_selection import train_test_split

logger = None

base_model='narm' srgnn/narm/gru4rec/GCSAN/bert4rec


class ACDataLoader(object):
    '''a custom data loader for the recommendation system data'''
    def __init__(self, data, batch_size):
        self.data=data
        self.num_sessions = len(data)
        self.batch_size = batch_size
        self.reset()

    def reset(self):
        self._start_idx = 0
        self._start_idx = 0
        self._has_next = True

    def has_next(self):
        return self._has_next

    def get_batch(self,kz):
        if not self._has_next:
            return None
          
        end_idx = min(self._start_idx + self.batch_size, self.num_sessions)
        

        batch_sessions = self.data['session'].iloc[self._start_idx:end_idx].tolist()

        batch_target=self.data['label'].iloc[self._start_idx:end_idx] 
        batch_len=self.data['length'].iloc[self._start_idx:end_idx].tolist() 
        batch_sids = self.data['session_id'].iloc[self._start_idx:end_idx].tolist()
        

        padded_sesss_item = torch.zeros(end_idx-self._start_idx, max(batch_len)).long()
 
        for i,se in enumerate(batch_sessions):
            padded_sesss_item[i, :batch_len[i]] = torch.LongTensor(se)

        self._has_next = self._has_next and end_idx < self.num_sessions
        self._start_idx = end_idx
        
        return padded_sesss_item.transpose(0, 1), batch_len, torch.tensor(batch_target.tolist()).long(),  batch_sids   # batch_uids


def train(args, base_model):
    
    path='./data/'+args.dataset + '/'
    train_data =  pd.read_csv( path + 'train_data.csv')
    val_data =  pd.read_csv(path + 'valid_data.csv')
    test_data =  pd.read_csv(path + 'test_data.csv')
    
    train_data['session'] = train_data['session'].apply(lambda x: eval(x))
    test_data['session'] = test_data['session'].apply(lambda x: eval(x))
    val_data['session'] = val_data['session'].apply(lambda x: eval(x))

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
        session=BERT(args,n_items).to(args.device)
        
    
    
    train_dataloader = ACDataLoader(train_data, args.batch_size) 
    val_dataloader = ACDataLoader(val_data, args.batch_size)  
    test_dataloader = ACDataLoader(test_data, args.batch_size)

    optimizer = optim.Adam(params=chain(session.parameters()), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    total_losses, total_plosses, total_entropy_act, total_entropy, total_rewards = [], [], [], [], []
    step = 0

    session_dic = {}
    last_hr = [0.0,0.0,0.0]
    last_ndcg = [0.0,0.0,0.0]

    # test
    test(args,base_model,session,test_dataloader)

    for epoch in range(1, args.epochs + 1):
        ### Start epoch ###
        kz=0

        session.train()
        train_dataloader.reset()

        while train_dataloader.has_next():
            optimizer.zero_grad()
            batch_sessions,seq_length,targets,batch_sids= train_dataloader.get_batch(kz) 
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
                session_rep,scores=session(x,edge_index,batch) 
                
            elif base_model=='narm':
                session_rep,scores=session(batch_sessions.to(args.device),seq_length) 
            elif base_model=='gru4rec':
                session_rep,scores=session(batch_sessions.to(args.device),seq_length) 
            elif base_model=='bert4rec':
                if batch_sessions.shape[0]<5:
                    batch_sessions=torch.cat((batch_sessions,torch.zeros(5-batch_sessions.shape[0],batch_sessions.shape[1],dtype=torch.long)))
                session_rep,scores=session(batch_sessions.transpose(0,1).to(args.device))
            
            batch_sessions=batch_sessions.transpose(0,1)
            dict1 = dict(zip(batch_sids, session_rep.tolist()))
            session_dic.update(dict1)
            
                    
            lr = args.lr * max(1e-4, 1.0 - float(step) / (args.epochs * len(train_data) / args.batch_size))
            for pg in optimizer.param_groups:
                pg['lr'] = lr
                
        
            loss = get_loss(scores.to(args.device),targets.to(args.device),criterion)
            loss.backward()
            
            optimizer.step()
            


            total_losses.append(loss.item())
            step += 1

            # Report performance
            if step > 0 and step % 100 == 0:
                avg_reward = np.mean(total_rewards) / args.batch_size
                avg_loss = np.mean(total_losses)
                avg_ploss = np.mean(total_plosses)
                avg_entropy_act=np.mean(total_entropy_act)
                avg_entropy = np.mean(total_entropy)
                total_losses, total_plosses,total_entropy_act, total_entropy, total_rewards = [], [], [], [], []
                logger.info(
                    'epoch/step={:d}/{:d}'.format(epoch, step) +
                    ' | loss={:.5f}'.format(avg_loss) +
                    ' | ploss={:.5f}'.format(avg_ploss) +
                    ' | ealoss={:.5f}'.format(avg_entropy_act) +
                    ' | entropy={:.5f}'.format(avg_entropy) +
                    ' | reward={:.5f}'.format(avg_reward))
            ### END of epoch ###
        policy_path = args.dataset + '/' + base_model
        if not os.path.isdir(policy_path):
            os.makedirs(policy_path)
        
        best_hr, best_ndcg = test(args,base_model,session,val_dataloader)
        if best_hr[0] > last_hr[0]:
            last_hr = best_hr
            last_ndcg = best_ndcg
            print("best_model ", epoch,"\nbest_hr@5,10,20", [float("{:.4f}".format(i)) for i in last_hr], "\nbest_ndcg@5,10,20", [float("{:.4f}".format(i)) for i in last_ndcg])
            policy_file = '{}/recommend_model.ckpt'.format(policy_path) #args.log_dir
            torch.save(session.state_dict(), policy_file)
            logger.info("Model saved to " + policy_file)
    
    
    policy_file = '{}/recommend_model.ckpt'.format(policy_path)
    session.load_state_dict(torch.load(policy_file))
    print("Model loaded from " + policy_file)
    print("--------------------test result ----------------------- ")
    test(args,base_model,session,test_dataloader)
    
        



def test(args,base_model,session,test_dataloader):
    
    session.eval()

    with torch.no_grad():
        all_targets = []
        hits, ndcgs = [], []
        hits10, ndcgs10 = [], []
        hits20, ndcgs20 = [], []
        test_dataloader.reset()
        session_dic = {}
        kz=0
        while test_dataloader.has_next():
            batch_sessions, seq_length, targets, batch_sids = test_dataloader.get_batch(kz)
            # print('targets',targets)
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
                session_rep,score=session(batch_sessions.to(args.device),seq_length) #[bs,embed_dim]
            elif base_model=='gru4rec':
                session_rep,score=session(batch_sessions.to(args.device),seq_length) #[bs,embed_dim]
            elif base_model=='bert4rec':
                if batch_sessions.shape[0]<5:
                    batch_sessions=torch.cat((batch_sessions,torch.zeros(5-batch_sessions.shape[0],batch_sessions.shape[1],dtype=torch.long)))           
                session_rep,score=session(batch_sessions.transpose(0,1).to(args.device))
  
            dict1 = dict(zip(batch_sids, session_rep.tolist()))
            session_dic.update(dict1)
            batch_sessions=batch_sessions.transpose(0,1)
            
            
            kz=1
            all_targets.extend(targets)
            preds = score
            
            
            
            topk_preds = torch.topk(preds, 5, dim=1)[1]
            
            for i, target in enumerate(targets):
                hits.append(get_hit_ratio(topk_preds[i], target, args))
                ndcgs.append(get_ndcg(topk_preds[i], target, args))
            
            topk_preds = torch.topk(preds, 10, dim=1)[1]
            for i, target in enumerate(targets):
                hits10.append(get_hit_ratio(topk_preds[i], target, args))
                ndcgs10.append(get_ndcg(topk_preds[i], target, args))
            
            topk_preds = torch.topk(preds, 20, dim=1)[1]
            
            
            
            for i, target in enumerate(targets):
                hits20.append(get_hit_ratio(topk_preds[i], target, args))
                ndcgs20.append(get_ndcg(topk_preds[i], target, args))

        avg_hit5 = np.mean(hits)*100
        avg_ndcg5 = np.mean(ndcgs)*100
        print("top-5-avg-hit",avg_hit5)
        print("top-5-avg-ndcg",avg_ndcg5)
        
        avg_hit10 = np.mean(hits10)*100
        avg_ndcg10 = np.mean(ndcgs10)*100
        print("top-10-avg-hit",avg_hit10)
        print("top-10-avg-ndcg",avg_ndcg10)
        
        avg_hit20 = np.mean(hits20)*100
        avg_ndcg20 = np.mean(ndcgs20)*100
        print("top-20-avg-hit",avg_hit20)
        print("top-20-avg-ndcg",avg_ndcg20)
    
    return  [avg_hit5, avg_hit10,avg_hit20], [avg_ndcg5,avg_ndcg10, avg_ndcg20]

def get_hit_ratio(preds, target, args):
    target = target.to(args.device)
    preds = preds.to(args.device)
    if target in preds:
        return 1
    return 0


def get_ndcg(preds, target, args):
    target = target.to(args.device)
    preds = preds.to(args.device)
    for i in range(len(preds)):
        if preds[i] == target:
            return math.log(2) / math.log(i+2)
    return 0


def get_loss(score, target, criterion):
  device=torch.device('cuda') if torch.cuda.is_available() else 'cpu'
  entropy_loss = criterion(score, target)
  return entropy_loss

def main():
    boolean = lambda x: (str(x).lower() == 'true')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="baby", help='One of {ML1M, BEAUTY, BABY}')
    parser.add_argument('--name', type=str, default='train_agent', help='directory name.')
    parser.add_argument('--seed', type=int, default=123, help='random seed.')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device.')
    parser.add_argument('--epochs', type=int, default=100, help='Max number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size.')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate.')
    parser.add_argument('--max_acts', type=int, default=100, help='Max number of actions.')
    parser.add_argument('--max_path_len', type=int, default=2, help='Max path length.')  
    parser.add_argument('--gamma', type=float, default=0.99, help='reward discount factor.')
    parser.add_argument('--ent_weight', type=float, default=1e-2, help='weight factor for entropy loss')
    parser.add_argument('--act_dropout', type=float, default=0.7, help='action dropout rate.')
    parser.add_argument('--state_history', type=int, default=1, help='state history length')
    parser.add_argument('--hidden', type=int, nargs='*', default=400, help='number of samples')
    parser.add_argument('--hidden_size', type=int, nargs='*', default=200, help='hidden size')
    parser.add_argument('--embed_dim', type=int, default=400, help='item embedding size of NARM') 
    parser.add_argument('--state_dim', type=int, default=400, help='dimension of state vector')
    parser.add_argument('--add_products', type=boolean, default=True, help='Add predicted products up to 10')
    parser.add_argument('--topk', type=int, nargs='*', default=[100,1], help='number of samples')  
    
    parser.add_argument('--bert_max_len', type=int, default=100, help='Length of sequence for bert') # 保留了最长的
    parser.add_argument('--bert_num_blocks', type=int, default=2, help='Number of transformer layers')
    parser.add_argument('--bert_num_heads', type=int, default=5, help='Number of heads for multi-attention')
    parser.add_argument('--bert_dropout', type=float, default=0.1, help='Dropout probability to use throughout the model')
    parser.add_argument('--bert_hidden_units', type=int, default=400, help='Size of hidden vectors (d_model)')
    
    args = parser.parse_args()
    #os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

    args.log_dir = '{}/{}'.format(TMP_DIR[args.dataset], args.name)
    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)

    global logger
    logger = get_logger(args.log_dir + '/train_log.txt')
    logger.info(args)

    set_random_seed(args.seed)
    base_model = "narm"
    train(args,base_model)


if __name__ == '__main__':
    main()