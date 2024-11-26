from __future__ import absolute_import, division, print_function

import sys

import numpy as np
from itertools import chain
import pandas as pd
import pickle
from functools import reduce
import warnings
warnings.filterwarnings('ignore')

import os
import argparse
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Bernoulli




from rl_env import BatchKGEnvironment 

from NARM import NARM
from SRGNN import SRGNN
from GRU4REC import GRU4REC
from GCSAN import GCSAN
from bert_modules.bert import BERT
from utils import *
from sklearn.model_selection import train_test_split
import math

logger = None


SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
base_model='narm' #srgnn/narm/gru4rec/GCSAN/bert4rec

class ActorCritic(nn.Module):
    '''
    return: actons, act_scores, session_emb
    '''
    def __init__(self,  gamma=0.99, hidden_sizes=400, embedding_size = 400):
        super(ActorCritic, self).__init__()
        
        self.gamma = gamma
        
        self.hidden_sizes = hidden_sizes
        self.emb_size = embedding_size
        self.l1 = nn.Linear(embedding_size, hidden_sizes)
        self.l2 = nn.Linear(embedding_size * 2, hidden_sizes)  # Assuming the concatenated state size is double of embedding size
        self.critic = nn.Linear(hidden_sizes, 1)
        
        self.saved_actions = []
        self.rewards = []
        self.entropy = []


    def forward(self, S_plus_emb, narm_state, act_embedding):

        # Convert to tensors if not already
        if not isinstance(S_plus_emb, torch.Tensor):
            S_plus_emb = torch.tensor(S_plus_emb, requires_grad=True) # batch_size, embedding_size*2?
        if not isinstance(narm_state, torch.Tensor):
            narm_state = torch.tensor(narm_state, requires_grad=True)

        
        cur_state = torch.cat((S_plus_emb, narm_state), dim=1)
        # Actor
        act = self.l1(act_embedding) 
        act_rep = F.dropout(F.elu(act), p = 0.5) 
        
        # Critic
        state = self.l2(cur_state.float())
        state_rep = F.dropout(F.elu(state), p = 0.5)
        
        # combine
        act_logit = torch.matmul(act_rep, state_rep.T) 
        act_logit = torch.sum(act_logit, dim=1).unsqueeze(1)
        act_probs = F.sigmoid(act_logit) 
        
        state_values = self.critic(state_rep) 
        return act_probs, state_values 

  
  
  
    def select_action(self, batch_state,narm_state,action_embedding, device):

        S_plus_emb = batch_state[0].to(device)
        narm_state = narm_state.to(device)
        action_embedding=torch.FloatTensor(action_embedding).to(device)  
        probs, state_values=self(S_plus_emb, narm_state, action_embedding) 
        m = Bernoulli(probs) 
        action = m.sample()  
        self.saved_actions.append(SavedAction(m.log_prob(action), state_values))
        self.entropy.append(m.entropy())
        action = action.squeeze(1)
        return action

    def update(self):
        del self.rewards[:]
        del self.saved_actions[:]
        del self.entropy[:]


class ACDataLoader(object):
    def __init__(self, data, batch_size):
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
        for i,se in enumerate(batch_sessions):
            padded_sesss_item[i, :batch_len[i]] = torch.LongTensor(se)
        
        self._has_next = self._has_next and end_idx < self.num_sessions
        self._start_idx = end_idx
        return padded_sesss_item.transpose(0, 1), batch_len, torch.tensor(batch_target.tolist()).long(), batch_sids  


def train(args,base_model):
    
    
    path='./data/'+args.dataset + '/'
    train_data =  pd.read_csv( path + 'train_data.csv')
    val_data =  pd.read_csv(path + 'valid_data.csv')
    test_data =  pd.read_csv(path + 'test_data.csv')
    
    train_data['session'] = train_data['session'].apply(lambda x: eval(x))
    test_data['session'] = test_data['session'].apply(lambda x: eval(x))
    val_data['session'] = val_data['session'].apply(lambda x: eval(x))
    n_items=len(np.load(path+'item_dict.npy',allow_pickle=True).item())
    print('n_items',n_items)
    
    policy_file = args.dataset + '/'+base_model+'/recommend_model.ckpt'
    
    
    # 模型初始化
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
    print("Model loaded from " + policy_file)
    session.load_state_dict(torch.load(policy_file,map_location=torch.device('cpu')))
    print("Model loaded from " + policy_file)
    
    
    
    train_dataloader = ACDataLoader(train_data, args.batch_size)

    env = BatchKGEnvironment(args.dataset, args.batch_size, args.max_acts, base_model)
    model = ActorCritic( gamma=args.gamma, hidden_sizes=args.hidden).to(args.device)
    logger.info('Parameters:' + str([i[0] for i in model.named_parameters()]))
    optimizer = optim.Adam(params=chain(model.parameters()), lr=args.lr)


    total_losses, total_plosses, total_entropy_act, total_entropy, total_rewards = [], [], [], [], []
    step = 0
    session_dic = {}
    K = 1
    best_reward = float('-inf')
    for epoch in range(1, args.epochs + 1):
        ### Start epoch ###
        total_avg_reward = 0
        kz=0
        #env.train()
        train_dataloader.reset()
        print("args.epochs",args.epochs,epoch)
        
        temp_i = 0
        factual_explanations = []
        counterfactual_explanations = []
        while train_dataloader.has_next():
            optimizer.zero_grad()
            batch_sessions,seq_length,targets,batch_sids= train_dataloader.get_batch(kz)
            session_rep, scores=get_session_representation(base_model, batch_sessions, seq_length, args, session)
            batch_sessions=batch_sessions.transpose(0,1)
            dict1 = dict(zip(batch_sids, session_rep.tolist()))
            session_dic.update(dict1)
            
            sub_session = []           
            session_emb_size = session_rep.shape[1]
            batch_state = env.reset(session_dic, batch_sids,  batch_sessions,  targets, session_emb_size) 

            done = False 
            rl_step = 0
            
            temp_i += 1
            S_plus,S_minus = env._get_S()
            
            S_plus = remove_zeros_and_pad(S_plus)
            S_minus = remove_zeros_and_pad(S_minus)
            # S_plus_emb
            S_plus = torch.tensor(S_plus,dtype=torch.int32) # batch_size,length
            S_plus_length = (S_plus != 0).sum(dim=1).long()
            S_plus_emb,S_plus_scores = get_session_representation(base_model, S_plus.transpose(0,1), S_plus_length, args, session)
            #S_minus_emb
            S_minus = torch.tensor(S_minus,dtype=torch.int32) # batch_size,length
            S_minus_length = (S_minus != 0).sum(dim=1).long()
            S_minus_emb,S_minus_scores=get_session_representation(base_model, S_minus.transpose(0,1), S_minus_length, args, session)
            f_preds = S_plus_scores
            f_topk_preds = torch.topk(f_preds, K, dim=1)[1]
            
            cf_preds = S_plus_scores
            cf_topk_preds = torch.topk(cf_preds, K, dim=1)[1]
            our_scores,batch_curr_actions = [],[] 
            while not done:
 
                batch_action_embedding = env.get_action_embedding(env._batch_curr_actions)  
                action_scores  = model.select_action(batch_state, session_rep, batch_action_embedding, args.device)  
                batch_state, batch_reward, done = env.batch_step(rl_step, action_scores, session_dic, targets, batch_sessions,S_plus_emb,S_minus_emb, f_topk_preds, cf_topk_preds)
                S_plus,S_minus = env._get_S()
                if base_model in ['narm', 'gru4rec']:
                    S_plus = remove_zeros_and_pad(S_plus)
                    S_minus = remove_zeros_and_pad(S_minus)
                S_plus = torch.tensor(S_plus,dtype=torch.int32)
                S_plus_length = (S_plus != 0).sum(dim=1).long()
                S_plus_emb,S_plus_scores = get_session_representation(base_model, S_plus.transpose(0,1), S_plus_length, args, session)
                S_minus = torch.tensor(S_minus,dtype=torch.int32)
                S_minus_length = (S_minus != 0).sum(dim=1).long()
                S_minus_emb,S_minus_scores=get_session_representation(base_model, S_minus.transpose(0,1), S_minus_length, args, session)

                rl_step += 1
                f_preds = S_plus_scores
                f_topk_preds = torch.topk(f_preds, K, dim=1)[1]
                cf_preds = S_minus_scores
                cf_topk_preds = torch.topk(cf_preds, K, dim=1)[1]
                batch_curr_actions=env._batch_curr_actions               
                model.rewards.append(batch_reward) 
            
            ### End of episodes ###   
            
            lr = args.lr * max(1e-4, 1.0 - float(step) / (args.epochs * len(train_data) / args.batch_size))
            for pg in optimizer.param_groups:
                pg['lr'] = lr
            
            policy_loss, value_loss, entropy_loss = compute_loss(model.saved_actions, model.rewards, model.gamma, entropy_coefficient=0.01, device=args.device)
            total_loss = policy_loss + value_loss + entropy_loss
            total_loss = total_loss.float()
            total_loss.backward()
            optimizer.step()
            model_rewards = np.array(model.rewards)
            total_rewards.append(np.sum(np.sum(model.rewards,axis = 1)))
            total_losses.append(total_loss.item())
            
            total_plosses.append(policy_loss.item())
            total_entropy_act.append(entropy_loss.item())
            model.update()
            total_avg_reward +=  np.mean(total_rewards)
            
            # Report performance
            if step > 0 and step % 100 == 0: 
                avg_reward = np.sum(total_rewards) / args.batch_size 
                avg_loss = np.mean(total_losses)
                avg_ploss = np.mean(total_plosses)
                avg_entropy_act=np.mean(total_entropy_act)
                total_losses, total_plosses,total_entropy_act, total_entropy, total_rewards = [], [], [], [], []
                logger.info(
                    'epoch/step={:d}/{:d}'.format(epoch, step) +
                    ' | loss={:.5f}'.format(avg_loss) +
                    ' | ploss={:.5f}'.format(avg_ploss) +
                    ' | ealoss={:.5f}'.format(avg_entropy_act)  +
                    ' | reward={:.5f}'.format(avg_reward))
            for sub_session in S_plus:
                non_zero_elements = [sub_session[idx].item() for idx in torch.nonzero(sub_session, as_tuple=True)[0]]
                factual_explanations.append(non_zero_elements)
                        
            for sub_session in S_minus:
                non_zero_elements = [sub_session[idx].item() for idx in torch.nonzero(sub_session, as_tuple=True)[0]]
                counterfactual_explanations.append(non_zero_elements)
            step += 1
            ### END of epoch ###
            
        if total_avg_reward > best_reward: # epoch >= args.epochs or 
            print("best_reward so far:", best_reward)
            best_reward = total_avg_reward
            print(f"Episode {epoch}: New best reward {best_reward}")
        
            explanation_path = args.dataset + '/explanations_fe/'+base_model
            if not os.path.isdir(explanation_path):
                os.makedirs(explanation_path)
            with open(explanation_path + '/factual_explanation.pkl', 'wb') as file:
                pickle.dump(factual_explanations, file)
            with open(explanation_path + '/counterfactual_explanation.pkl', 'wb') as file:
                pickle.dump(counterfactual_explanations, file)

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
        session_rep, scores = session(batch_sessions.to(args.device), seq_length)
    elif base_model == 'bert4rec':
        if batch_sessions.shape[0] < 5:
            padding = torch.zeros((5 - batch_sessions.shape[0], batch_sessions.shape[1]), dtype=torch.long)
            batch_sessions = torch.cat((batch_sessions, padding), dim=0)
        session_rep, scores = session(batch_sessions.transpose(0, 1).to(args.device))
    return session_rep, scores

def remove_zeros_and_pad(sessions):
    processed_sessions = []
    max_len = 0
    
    for session in sessions:
        filtered_session = [item for item in session.tolist() if item != 0 or all(x == 0 for x in session.tolist())]
        processed_sessions.append(torch.tensor(filtered_session, dtype=torch.int32))
        
        if len(filtered_session) > max_len:
            max_len = len(filtered_session)
    padded_sessions = torch.zeros((len(processed_sessions), max_len), dtype=torch.int32)
    for i, session in enumerate(processed_sessions):
        padded_sessions[i, :len(session)] = session
    
    return padded_sessions


def compute_loss(saved_actions, rewards, gamma, entropy_coefficient, device):
    R = 0
    eps = 1e-7  
    returns = []  #
    advantages = []  #
    
    for r in reversed(rewards): 
        R = r + gamma * R 
        returns.insert(0, R)
    returns = torch.tensor(returns, device=device).float()  
    returns = (returns - returns.mean()) / (returns.std() + eps)  
    policy_loss = []
    value_loss = []
    entropy_loss = []
    for (log_prob, value), R in zip(saved_actions, returns):
        R = R.to(device)  
        
        value = value.to(device) 
        advantage = R - value.squeeze() 
        advantages.append(advantage)
        policy_loss.append(-log_prob * advantage)
        value_loss.append(F.smooth_l1_loss(value.squeeze(), R))
        entropy_loss.append(-entropy_coefficient * log_prob.exp() * log_prob)

    return torch.stack(policy_loss).sum().float(), torch.stack(value_loss).sum().float(), torch.stack(entropy_loss).sum().float()


def evaluate_model(model, env, episodes):
    total_reward = 0
    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = model.predict(state)  # Change this to match your model's method
            state, reward, done = env.step(action)
            total_reward += reward
    return total_reward / episodes





def main():
    boolean = lambda x: (str(x).lower() == 'true')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=BABY, help='One of {ML1M, BEAUTY, BABY}')
    parser.add_argument('--name', type=str, default='train_agent', help='directory name.')
    parser.add_argument('--seed', type=int, default=123, help='random seed.')
    parser.add_argument('--gpu', type=str, default='1', help='gpu device.')
    parser.add_argument('--epochs', type=int, default=100, help='Max number of epochs.') 
    parser.add_argument('--batch_size', type=int, default=32, help='batch size.')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate.')
    parser.add_argument('--max_acts', type=int, default=100, help='Max number of actions.')
    parser.add_argument('--gamma', type=float, default=0.8, help='reward discount factor.') 
    parser.add_argument('--ent_weight', type=float, default=1e-2, help='weight factor for entropy loss')
    parser.add_argument('--act_dropout', type=float, default=0.7, help='action dropout rate.')
    parser.add_argument('--state_history', type=int, default=1, help='state history length')
    parser.add_argument('--hidden', type=int, nargs='*', default=400, help='number of samples')
    parser.add_argument('--embed_dim', type=int, default=400, help='item embedding size of NARM') 
    parser.add_argument('--state_dim', type=int, default=400, help='dimension of state vector')
    parser.add_argument('--add_products', type=boolean, default=True, help='Add predicted products up to 10')
    parser.add_argument('--topk', type=int, nargs='*', default=[100,1], help='number of samples')  
    parser.add_argument('--base_model', type=int, nargs='*', default=[100,1], help='recommendation model')  
    parser.add_argument('--bert_max_len', type=int, default=100, help='Length of sequence for bert') 
    parser.add_argument('--bert_num_blocks', type=int, default=2, help='Number of transformer layers')
    parser.add_argument('--bert_num_heads', type=int, default=5, help='Number of heads for multi-attention')
    parser.add_argument('--bert_dropout', type=float, default=0.1, help='Dropout probability to use throughout the model')
    parser.add_argument('--bert_hidden_units', type=int, default=400, help='Size of hidden vectors (d_model)')
    
    args = parser.parse_args() 
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(args.device)
    args.log_dir = '{}/{}'.format(TMP_DIR[args.dataset], args.name)
    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)

    global logger
    logger = get_logger(args.log_dir + '/explain_log.txt')
    logger.info(args)

    set_random_seed(args.seed)
    base_model = "srgnn"
    train(args,base_model)

if __name__ == '__main__':
    main()