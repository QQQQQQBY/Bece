import argparse
import warnings
warnings.filterwarnings('always')  # "error", "ignore", "always", "default", "module" or "once"
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import time
from sklearn.utils import shuffle
# from torch.utils.data import DataLoader,Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import datetime
import logging
from torch_geometric.utils.undirected import to_undirected
from torch_geometric.data import Data
from torch_geometric.nn.conv import HGTConv
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch_geometric.data import HeteroData
import torch_geometric
import pickle
from operator import itemgetter
import math
from torch_sparse import SparseTensor

def get_logger(name,log_dir):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.DEBUG)
    fileHandler = logging.FileHandler(filename=log_dir,mode='w+',encoding='utf-8')
    fileHandler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s|%(levelname)-8s|%(filename)10s:%(lineno)4s|%(message)s")
    consoleHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)
    logger.addHandler(fileHandler)
    return logger


def load_data(seed,bs, datasets_name):
    if datasets_name == "mgtab":
        # load features(value category tweet(mean pooling))
        feature = torch.load(' Dataset/MGTAB/features.pt')
        # The first relationship
        edge_index_0 = torch.load(' Dataset/MGTAB/edge_index_0.pt')
        # The Second relationship
        edge_index_1 = torch.load(' Dataset/MGTAB/edge_index_1.pt')
        # Transform to undirected
        edge_index_0 = to_undirected(edge_index_0)
        edge_index_1 = to_undirected(edge_index_1)
        # load label
        label = torch.load(' Dataset/MGTAB/labels_bot.pt')
        # the data format of HGT needs 
        data = HeteroData()
        data['user'].x = feature
        data['user', 'follower', 'user'].edge_index = edge_index_0
        data['user', 'friend', 'user'].edge_index = edge_index_1
        data['user'].y = label
        def sample_mask(idx, l):
            """Create mask."""
            mask = torch.zeros(l)
            mask[idx] = 1
            return torch.as_tensor(mask, dtype=torch.bool)
        sample_number = len(data['user'].y)
        # shuffle dataset
        shuffled_idx = shuffle(np.array(range(len(data['user'].y))), random_state=seed)    
        train_idx = shuffled_idx[:int(0.7* data['user'].y.shape[0])].tolist()
        val_idx = shuffled_idx[int(0.7*data['user'].y.shape[0]): int(0.9*data['user'].y.shape[0])].tolist()
        test_idx = shuffled_idx[int(0.9*data['user'].y.shape[0]):].tolist()
        train_mask = sample_mask(train_idx, sample_number)
        val_mask = sample_mask(val_idx, sample_number)
        test_mask = sample_mask(test_idx, sample_number)
        data['user'].train_mask = train_mask
        data['user'].val_mask = val_mask
        data['user'].test_mask = test_mask

        return data,train_idx, val_idx, test_idx
    if datasets_name == "cresci-15":
         # load features(value category tweet(mean pooling))
        feature = torch.load('Dataset/Cresci-15/feature_cat_num_tweet.pt')
        # The first relationship
        edge_index_0 = torch.load('Dataset/Cresci-15/edge_index_0.pt')
        # The Second relationship
        edge_index_1 = torch.load('Dataset/Cresci-15/edge_index_1.pt')
        # Transform to undirected
        edge_index_0 = to_undirected(edge_index_0)
        edge_index_1 = to_undirected(edge_index_1)
        # load label
        label = torch.load('Dataset/Cresci-15/label.pt')
        # the data format of HGT needs 
        data = HeteroData()
        data['user'].x = feature
        data['user', 'follower', 'user'].edge_index = edge_index_0
        data['user', 'friend', 'user'].edge_index = edge_index_1
        data['user'].y = label
        def sample_mask(idx, l):
            """Create mask."""
            mask = torch.zeros(l)
            mask[idx] = 1
            return torch.as_tensor(mask, dtype=torch.bool)
        sample_number = len(data['user'].y)
        # follow previous work, no messing with the dataset
        shuffled_idx = np.array(range(len(data['user'].y)))    
        train_idx = shuffled_idx[:int(0.7* data['user'].y.shape[0])].tolist()
        val_idx = shuffled_idx[int(0.7*data['user'].y.shape[0]): int(0.9*data['user'].y.shape[0])].tolist()
        test_idx = shuffled_idx[int(0.9*data['user'].y.shape[0]):].tolist()
        train_mask = sample_mask(train_idx, sample_number)
        val_mask = sample_mask(val_idx, sample_number)
        test_mask = sample_mask(test_idx, sample_number)
        data['user'].train_mask = train_mask
        data['user'].val_mask = val_mask
        data['user'].test_mask = test_mask

        return data,train_idx, val_idx, test_idx
    
    if datasets_name == "twibot-20":
         # load features(value category tweet(mean pooling))
        feature = torch.load('Dataset/Twibot-20/feature_cat_num_tweet_des_6_11.pt')
        # The first relationship
        edge_index_0 = torch.load('Dataset/Twibot-20/edge_index_0.pt')
        # The Second relationship
        edge_index_1 = torch.load('Dataset/Twibot-20/edge_index_1.pt')
        # Transform to undirected
        edge_index_0 = to_undirected(edge_index_0)
        edge_index_1 = to_undirected(edge_index_1)
        # load label
        label = torch.load('Dataset/Twibot-20/node_label.pt')
        # the data format of HGT needs 
        data = HeteroData()
        data['user'].x = feature
        data['user', 'follower', 'user'].edge_index = edge_index_0
        data['user', 'friend', 'user'].edge_index = edge_index_1
        data['user'].y = label
        def sample_mask(idx, l):
            """Create mask."""
            mask = torch.zeros(l)
            mask[idx] = 1
            return torch.as_tensor(mask, dtype=torch.bool)
        sample_number = len(data['user'].y)
        # follow previous work, no messing with the dataset
        shuffled_idx = np.array(range(len(data['user'].y)))    
        train_idx = shuffled_idx[:int(0.7* data['user'].y.shape[0])].tolist()
        val_idx = shuffled_idx[int(0.7*data['user'].y.shape[0]): int(0.9*data['user'].y.shape[0])].tolist()
        test_idx = shuffled_idx[int(0.9*data['user'].y.shape[0]):].tolist()
        train_mask = sample_mask(train_idx, sample_number)
        val_mask = sample_mask(val_idx, sample_number)
        test_mask = sample_mask(test_idx, sample_number)
        data['user'].train_mask = train_mask
        data['user'].val_mask = val_mask
        data['user'].test_mask = test_mask

        return data,train_idx, val_idx, test_idx

# Gaussian processing
class RegHead(nn.Module):

    def __init__(self, feat_dim=128, drop_ratio=0.4):
        super(RegHead, self).__init__()
        self.gamma = nn.Parameter(torch.ones(1) )
        self.beta = nn.Parameter(torch.zeros(1))
        self.mu_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim ))

        # use logvar instead of var !!!
        self.logvar_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim, bias=False),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim, bias=False),
            nn.BatchNorm1d(feat_dim,   affine=False)
        )

    def forward(self, x):
        # Eq.(4)
        mu =  self.mu_head(x)
        logvar = self.logvar_head(x)
        # Gaussian processing
        embedding = self._reparameterize(mu, logvar)
        # Eq.(5)
        logvar = self.gamma * logvar + self.beta
        return (mu, logvar, embedding)

    def _reparameterize(self, mu, logvar):
        std = torch.exp(logvar).sqrt()
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

# KL divergance
def kl_loss_uncertain(mu, logvar):
    kl_loss = -(1 + logvar - mu.pow(2) - logvar.exp()) / 2
    kl_loss = kl_loss.sum(dim=1).mean()
    return kl_loss / mu.size(0)  #*mu.size(1)    

#  Unreliable Edges Removal Module
class SelectorEage(nn.Module):
    def __init__(self,feature_dim,classes):
        super(SelectorEage,self).__init__()       
        with open(' Dataset/MGTAB/relation0.pickle', 'rb') as file:
	        self.relation0 = pickle.load(file)
        file.close()
        
        with open(' Dataset/MGTAB/relation1.pickle', 'rb') as file:
	        self.relation1 = pickle.load(file)
        file.close()
        with open(' Dataset/MGTAB/relation0_label.pickle', 'rb') as file:
	        self.relation0_label = pickle.load(file)
        file.close()
        with open(' Dataset/MGTAB/relation1_label.pickle', 'rb') as file:
	        self.relation1_label = pickle.load(file)
        file.close()

        self.adj_lists = [self.relation0, self.relation1]
        self.relation_label = [self.relation0_label,self.relation1_label]
        self.preception = nn.Linear(feature_dim,int(feature_dim/2))
        # self.label_preception = nn.Linear(int(feature_dim/2), classes)

 
    def filter_neighs(self,center_features, neigh_features, neighs_list, r_labels):
        # samp_neighs -- Neighbor node 
        samp_neighs = []
        # samp_features -- the distance features of central nodes and neighbor nodes
        samp_features = []
        samp_labels = []
        for idx, center_feature in enumerate(center_features):
            center_feature = center_features[idx] # the feature of central nodes
            neigh_feature = neigh_features[idx] # the feature of neighbor nodes
            center_feature = center_feature.repeat(neigh_feature.size()[0], 1)
            neighs_indices = neighs_list[idx] # The index of neighbor nodes
            labels_indices = r_labels[idx]
            # L1 - distance
            feature_diff = torch.abs(center_feature - neigh_feature) 
            samp_neighs.append(neighs_indices)
            samp_features.append(feature_diff.tolist())
            samp_labels.append(labels_indices)
        return samp_neighs, samp_features, samp_labels


    def forward(self,features,nodes_idx, device):
        to_neighs = []
        label_neighs = []
        neighs = []
        # acquire the neighbor nodes of the central node in each relationship
        for adj_list, label_list in zip(self.adj_lists, self.relation_label):
            to_neighs.append([adj_list[int(node)] for node in nodes_idx])
            label_neighs.append([label_list[int(node)] for node in nodes_idx])           
            neighs.append([set(adj_list[int(node)]) for node in nodes_idx])
        # Get the involoved node features in each relationship 
        unique_nodes = set.union(set.union(*neighs[0]),
								 set.union(*neighs[1], set(nodes_idx)))

        batch_features = features[torch.LongTensor(list(unique_nodes))].to(device)        
        batch_features = self.preception(batch_features)

        # get the original id of node and corresponding index
        id_mapping = {node_id: index for node_id, index in zip(unique_nodes, range(len(unique_nodes)))} 

        # Get the prediction score of the labeled nodes
        node_idx_id = itemgetter(*nodes_idx)(id_mapping)

        # Get the features of the labeled nodes
        center_feature = batch_features[node_idx_id,:]

        # Converts the neighbor nodes of central labeled node in each relation to the list type
        r1_list = [list(to_neigh) for to_neigh in to_neighs[0]] # The first Relationship
        r2_list = [list(to_neigh) for to_neigh in to_neighs[1]] # The second Relationship 
        r1_label = [list(label_neigh) for label_neigh in label_neighs[0]] 
        r2_label = [list(label_neigh) for label_neigh in label_neighs[1]] 

        # Get features of central nodes and neighbor nodes
        r1_features = [batch_features[itemgetter(*to_neigh)(id_mapping), :].view(-1, 64) for to_neigh in r1_list] # 标签感知分数
        r2_features = [batch_features[itemgetter(*to_neigh)(id_mapping), :].view(-1, 64) for to_neigh in r2_list]

        # Filter the neighbor nodes in each relationship
        samp_neighs_1, samp_features_1, samp_labels_1 = self.filter_neighs(center_feature, r1_features, r1_list, r1_label)
        samp_neighs_2, samp_features_2, samp_labels_2 = self.filter_neighs(center_feature, r2_features, r2_list, r2_label)

        # Process edges [source, Target]
        # relation0
        follower_source = []
        follower_dest = []
        follower_feature = []
        follower_label = []
        for samp_neigh,samp_feature, samp_label in zip(samp_neighs_1,samp_features_1, samp_labels_1):
            assert len(samp_neigh) == len(samp_feature)
            for i in range(len(list(samp_neigh))):
                follower_source.append(list(samp_neigh)[0])
                follower_dest.append(list(samp_neigh)[i])
                follower_feature.append(samp_feature[i])
                follower_label.append(samp_label[i])

        friend_source = []
        friend_dest = []
        friend_feature = []
        friend_label = []
        for samp_neigh,samp_feature, samp_label in zip(samp_neighs_2, samp_features_2, samp_labels_2):
            assert len(samp_neigh) == len(samp_feature)
            for i in range(len(list(samp_neigh))):
                friend_source.append(list(samp_neigh)[0])
                friend_dest.append(list(samp_neigh)[i])
                friend_feature.append(samp_feature[i])
                friend_label.append(samp_label[i])

        # Transform to Tensor type
        follower = [follower_source,follower_dest]
        follower = torch.tensor(follower).to(device)

        friend = [friend_source,friend_dest]
        friend = torch.tensor(friend).to(device)

        follower_feature = torch.tensor(follower_feature).to(device)
        friend_feature = torch.tensor(friend_feature).to(device)

        follower_label = torch.tensor(follower_label).to(device)
        friend_label = torch.tensor(friend_label).to(device)
        return follower,friend,follower_feature,friend_feature,follower_label,friend_label

class EdgeUpdate(nn.Module):
    def __init__(self, feature_dim, edge_dim, device, load_dir = None):
        super(EdgeUpdate, self).__init__()
        self.feature_dim = feature_dim # 100 788
        self.edge_dim = edge_dim # 1
        self.temp = 0.6
        self.mins = torch.tensor(1e-10).to(device)
        self.relu_fuc = nn.ReLU()
        self.edge_skip_alpha = nn.Parameter(torch.rand(1))
        self.dim = feature_dim # 100 788

        self.l2r = nn.Sequential(
                nn.Linear(self.dim, self.dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(self.dim, 1)
            ) 

    def forward(self, edge_index, edge_attr, device):
        pre_prob = self.l2r(edge_attr).squeeze(-1) # Predict edge label
        m = nn.Sigmoid()
        pre_adj = m(pre_prob) # delete edges
        sampled_edge = torch.bernoulli(pre_adj).to(device)
        # sampled_edge = pyro.distributions.RelaxedBernoulliStraightThrough(temperature=self.temp, probs = pre_adj).rsample() # torch.Size([3846979])
        mask = (sampled_edge > 0.0)
        edge_index = edge_index.T[mask].T
        return edge_index,pre_adj


# Construct Model
class AttrModel(nn.Module):
    def __init__(self, datasets_name, value_num, bool_num, tweet_num, des_num, feature_dim,classes,dropout,data,num_layers,device):
        super(AttrModel,self).__init__()
        self.dropout = dropout
        # three type of features, User Information Embedding, Eq.(1)
        if datasets_name == "twibot-20":
            # Eq.(1)
            self.fc1 = nn.Linear(value_num,feature_dim)
            self.fc2 = nn.Linear(bool_num,feature_dim)
            self.fc3 = nn.Linear(tweet_num,feature_dim)
            self.fc4 = nn.Linear(des_num,feature_dim)
            # Attention Fusion Eq.(2)
            self.attn = nn.MultiheadAttention(embed_dim=feature_dim*4, num_heads=4, dropout=0.3, batch_first=True)
            self.relu = nn.Sequential(
            nn.Linear(feature_dim*4,feature_dim),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.3)
        )
            self.lin_dict = torch.nn.ModuleDict()
            for node_type in data.node_types:
                self.lin_dict[node_type] = torch_geometric.nn.Linear(-1, feature_dim)
        else:
            # Eq.(1)
            self.fc1 = nn.Linear(value_num,feature_dim)
            self.fc2 = nn.Linear(bool_num,feature_dim)
            self.fc3 = nn.Linear(tweet_num,feature_dim)
            # Attention Fusion Eq.(2)
            self.attn = nn.MultiheadAttention(embed_dim=feature_dim*3, num_heads=4, dropout=0.3, batch_first=True)
            self.relu = nn.Sequential(
            nn.Linear(feature_dim*3,feature_dim),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.3)
        )
            self.lin_dict = torch.nn.ModuleDict()
            for node_type in data.node_types:
                self.lin_dict[node_type] = torch_geometric.nn.Linear(-1, feature_dim)
        
        
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.han = HGTConv(feature_dim,feature_dim,metadata=data.metadata(), heads = 4, dropout=0.3,group='mean')
            self.convs.append(self.han)
        
        self.edge_dim_1 = nn.Sequential(nn.Linear(int(feature_dim/2),feature_dim), 
                                        nn.LeakyReLU(inplace = True),
                                        nn.Dropout(0.3))
        self.edge_dim_2 = nn.Sequential(nn.Linear(int(feature_dim/2),feature_dim), 
                                        nn.LeakyReLU(inplace = True),
                                        nn.Dropout(0.3))
        self.fc5 = nn.Linear(feature_dim,classes)
        self.classes = classes

        #  Unreliable Edges Removal Module
        self.edge = SelectorEage(feature_dim,self.classes) 
        self.goss = RegHead(feature_dim)
        
        self.deledge = EdgeUpdate(feature_dim,feature_dim, device)

    def edge_handle(self, samp_score_1, samp_score_2): 
        samp_scores_1 = self.edge_dim_1(samp_score_1)
        samp_scores_2 = self.edge_dim_2(samp_score_2)
        # Gaussian processing
        mu_1, logvar_1, embedding_1 = self.goss(samp_scores_1)
        mu_2, logvar_2, embedding_2 = self.goss(samp_scores_2)
        unsuploss1 = kl_loss_uncertain(mu_1, logvar_1)
        unsuploss2 = kl_loss_uncertain(mu_2, logvar_2)
        return embedding_1,embedding_2,unsuploss1,unsuploss2


    def forward(self, datasets_name, value_feature,bool_feature,tweet_feature,x_dict,edge_index_dict,nodes_idx,device, idx):
        if datasets_name == "twibot-20":
            # Eq.(1)
            value_feature = self.fc1(value_feature)
            bool_feature = self.fc2(bool_feature)
            tweet_feature = self.fc3(tweet_feature)
            des_feature = self.fc4(des_feature)
            # Eq.(2)
            feature = torch.concat((value_feature, bool_feature, tweet_feature, des_feature),dim=1)
            feature,_ = self.attn(feature,feature,feature)
            feature = self.relu(feature)
            for node_type, _ in x_dict.items():
                x_dict[node_type] = self.lin_dict[node_type](feature).relu_()     
        else:
            # Eq.(1)
            value_feature = self.fc1(value_feature)
            bool_feature = self.fc2(bool_feature)
            tweet_feature = self.fc3(tweet_feature)
            # Eq.(2)
            feature = torch.concat((value_feature,bool_feature,tweet_feature),dim=1)
            feature,_ = self.attn(feature,feature,feature)
            feature = self.relu(feature)
            for node_type, _ in x_dict.items():
                x_dict[node_type] = self.lin_dict[node_type](feature).relu_() 
        # nodes_idx -- labeled nodes in this batch
        # center_scores = torch.zeros((len(nodes_idx),2)).to(device)
        unsuploss1_sum = torch.tensor(0.0).to(device)
        unsuploss2_sum = torch.tensor(0.0).to(device)
        for conv in self.convs:
            follower,friend,samp_features_1,samp_features_2, follower_label,friend_label = self.edge(x_dict['user'], nodes_idx, device)
            # Gaussian processing -- Eq.(5)
            samp_features_1 , samp_features_2,unsuploss1,unsuploss2 = self.edge_handle(samp_features_1,samp_features_2) 
            # delete unreliable edges
            follower,pre_adj_1 = self.deledge(follower, samp_features_1, device)
            friend,pre_adj_2 = self.deledge(friend, samp_features_2, device)  
            pre_adj_1 =pre_adj_1.to(device)
            pre_adj_2 = pre_adj_2.to(device) 
            if  'pre_adjs_1' not in locals().keys():
                pre_adjs_1 = torch.zeros(pre_adj_1.shape[0]).to(device)
            pre_adjs_1 += pre_adj_1
            if  'pre_adjs_2' not in locals().keys():
                pre_adjs_2 = torch.zeros(pre_adj_2.shape[0]).to(device)
            pre_adjs_2 += pre_adj_2      
            # unsuploss1, unsuploss2 -- KL loss
            unsuploss1_sum += unsuploss1
            unsuploss2_sum += unsuploss2 
            follower = follower.long()  
            friend = friend.long()
            # Convert the edge format required by HGT Graph          
            # followers = SparseTensor(row=follower[0], col=follower[1],sparse_sizes=(10199,10199))
            # friends = SparseTensor(row=friend[0], col=friend[1],sparse_sizes=(10199,10199))
            edge_index_dict[('user', 'follower', 'user')] = follower
            edge_index_dict[('user', 'friend', 'user')] = friend

            # Aggregate neighbor features by updating edges            
            x_dict = conv(x_dict, edge_index_dict)
        
        # 2 layers
        pre_adjs_2 = pre_adjs_2/2
        pre_adjs_1 = pre_adjs_1/2
        unsuploss1_sum = unsuploss1_sum/2
        unsuploss2_sum = unsuploss2_sum/2

        # predict node
        x = x_dict['user'][nodes_idx]      
        output = self.fc5(x) 

        # select labeled edges with source node and target node both in train set 
        follower_index = []
        follower = follower.tolist()
        for i in range(len(follower[0])): 
            if follower[0][i] in idx and follower[1][i] in idx:
                follower_index.append(i)
        follower_index = torch.tensor(follower_index).to(device)
        friend_index = []
        friend = friend.tolist()
        for i in range(len(friend[1])):           
            if friend[0][i] in idx and friend[1][i] in idx:
                friend_index.append(i)
        friend_index = torch.tensor(friend_index).to(device)

        return output,unsuploss1_sum,unsuploss2_sum,friend_label[friend_index],follower_label[follower_index],pre_adj_1[follower_index],pre_adj_2[friend_index]
        

loss_ce = nn.CrossEntropyLoss()
loss_2 = nn.BCELoss()
# Train
def train(datasets_name, boolean_num, value_num, tweet_num, des_num, data,train_idx, val_idx, test_idx,model,num_epochs,lr,weight_decay,logger,lambda_1,model_file,batch_size,lambda_2,lambda_3,device):
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=lr, weight_decay=weight_decay)
    max_acc = 0
    for epoch in tqdm(range(num_epochs)):
        model.train()
        train_acc_total = 0
        train_loss_total = 0
        if datasets_name == "mgtab":
            val = torch.tensor([3,5,6,7,9,10,11,12,13,14]).to(device)
            boolean = torch.tensor([0,1,2,4,8,15,16,17,18,19]).to(device)
        if datasets_name == "twibot-20":
            val = torch.tensor([11,12,13,14,15,16]).to(device)
            boolean = torch.tensor([0,1,2,3,4,5,6,7,8,9,10]).to(device)
        if datasets_name == "cresci-15":
            boolean = torch.tensor([0]).to(device)
            val = torch.tensor([1,2,3,4,5]).to(device)
        i = 0
        num_batches = int(len(train_idx) / batch_size) + 1
        for batch in range(num_batches):
            data = data.to(device)
            label = data['user'].y.to(device) 
            i_start = batch * batch_size
            i_end = min((batch + 1) * batch_size, len(train_idx))
            batch_nodes = train_idx[i_start:i_end]
            batch_label = label[batch_nodes]
            if datasets_name == "twibot-20":
                value_feature = data['user'].x[:,val].to(device)
                bool_feature = data['user'].x[:,boolean].to(device)
                tweet_feature = data['user'].x[:, boolean_num + value_num: boolean_num + value_num + tweet_num].to(device)
                des_feature = data['user'].x[:, boolean_num + value_num + tweet_num: ].to(device)
                output,unsuploss1_sum,unsuploss2_sum,friend_label,follower_label,pre_adjs_1,pre_adjs_2 = model(datasets_name, value_feature,bool_feature,tweet_feature, des_feature, data.x_dict,data.edge_index_dict,batch_nodes,device, train_idx)
            else:
                value_feature = data['user'].x[:,val].to(device)
                bool_feature = data['user'].x[:,boolean].to(device)
                tweet_feature = data['user'].x[:,boolean_num + value_num:].to(device)                     
                output,unsuploss1_sum,unsuploss2_sum,friend_label,follower_label,pre_adjs_1,pre_adjs_2 = model(datasets_name, value_feature,bool_feature,tweet_feature,data.x_dict,data.edge_index_dict,batch_nodes,device, train_idx)
            pre_adjs_1 = pre_adjs_1.to(torch.float32)
            pre_adjs_2 = pre_adjs_2.to(torch.float32)
            friend_label = friend_label.to(torch.float32)
            follower_label = follower_label.to(torch.float32)
            loss = loss_ce(output,batch_label) + lambda_2*unsuploss1_sum + lambda_2*unsuploss2_sum + lambda_3*loss_2(pre_adjs_1,follower_label) + lambda_3*loss_2(pre_adjs_2,friend_label)
            
            out = output.max(1)[1].to('cpu').detach().numpy()
            batch_label = batch_label.to('cpu').detach().numpy()            
            train_loss_total += loss
            acc_train = accuracy_score(out, batch_label)
            train_acc_total += acc_train
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            i += 1
        loss = train_loss_total/i
        acc_train = train_acc_total/i
        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(loss.item()),
              'acc_train: {:.4f}'.format(acc_train.item()))
        logger.info(f'Epoch: {epoch + 1}, loss_train: {loss.item()}, acc_train: {acc_train.item()}')
        # Validation
        acc_val,f1_val,precision_val,recall_val,loss_val = test(datasets_name, boolean_num, value_num, tweet_num, des_num, data,val_idx,model,loss_ce,lambda_1,batch_size,lambda_2,lambda_3, device)
        print("Val set results:",
          "epoch= {:}".format(epoch+1),
          "test_accuracy= {:.4f}".format(acc_val),
          "precision= {:.4f}".format(precision_val),
          "recall= {:.4f}".format(recall_val),
          "f1_score= {:.4f}".format(f1_val))
        logger.info(f"Val set results:epoch={epoch+1}, val_accuracy= {acc_val}, precision= {precision_val}, recall= {recall_val}, f1_score= {f1_val}")

        # Save model
        if acc_val > max_acc:
            max_acc = acc_val
            print("save model...")
            logger.info("save model...")
            
            torch.save(model.state_dict()," Save_model/{}.pth".format(model_file))
        
        # Test
        acc_test,f1_test,precision_test,recall_test,loss = test(datasets_name, boolean_num, value_num, tweet_num, des_num, data,test_idx,model,loss_ce,lambda_1,batch_size,lambda_2,lambda_3, device)
        print("Test set results:",
          "epoch= {:}".format(epoch),
          "test_accuracy= {:.4f}".format(acc_test),
          "precision= {:.4f}".format(precision_test),
          "recall= {:.4f}".format(recall_test),
          "f1_score= {:.4f}".format(f1_test))
        logger.info(f"Test set results: epoch={epoch+1}, test_accuracy= {acc_test}, precision= {precision_test}, recall= {recall_test}, f1_score= {f1_test}")

    
        
# Test
@torch.no_grad()
def test(datasets_name, boolean_num, value_num, tweet_num, des_num, data, idx, model, loss_ce, lambda_1, batch_size, lambda_2, lambda_3, device):
    if datasets_name == "mgtab":
        val = torch.tensor([3,5,6,7,9,10,11,12,13,14]).to(device)
        boolean = torch.tensor([0,1,2,4,8,15,16,17,18,19]).to(device)
    if datasets_name == "twibot-20":
        val = torch.tensor([11,12,13,14,15,16]).to(device)
        boolean = torch.tensor([0,1,2,3,4,5,6,7,8,9,10]).to(device)
    if datasets_name == "cresci-15":
        boolean = torch.tensor([0]).to(device)
        val = torch.tensor([1,2,3,4,5]).to(device)
    model.eval()
    acc_test_total = 0
    f1_test_total = 0
    precision_test_total = 0
    recall_test_total = 0
    loss_test_total = 0
    i = 0
    num_batches = int(len(idx) / batch_size) + 1
    for batch in range(num_batches):
        data = data.to(device)
        value_feature = data['user'].x[:,val].to(device)
        bool_feature = data['user'].x[:,boolean].to(device)
        tweet_feature = data['user'].x[:, boolean_num + value_num:].to(device)
        label = data['user'].y.to(device)
        i_start = batch * batch_size
        i_end = min((batch + 1) * batch_size, len(train_idx))
        batch_nodes = idx[i_start:i_end]
        batch_label = label[batch_nodes]
        output, unsuploss1_sum, unsuploss2_sum, friend_label, follower_label, pre_adjs_1,pre_adjs_2 = model(datasets_name, value_feature, bool_feature, tweet_feature, data.x_dict, data.edge_index_dict, batch_nodes, device, idx)
        pre_adjs_1 = pre_adjs_1.to(torch.float32)
        pre_adjs_2 = pre_adjs_2.to(torch.float32)
        friend_label = friend_label.to(torch.float32)
        follower_label = follower_label.to(torch.float32)
        loss = loss_ce(output,batch_label) + lambda_2*unsuploss1_sum + lambda_2*unsuploss2_sum + lambda_3*loss_2(pre_adjs_1,follower_label) + lambda_3*loss_2(pre_adjs_2,friend_label)

        batch_label = batch_label.to('cpu').detach().numpy()
        out = output.max(1)[1].to('cpu').detach().numpy()       
        loss_test_total += loss
        if datasets_name == "mgtab":
            acc_test = accuracy_score(out, batch_label)
            acc_test_total += acc_test
            f1_test = f1_score(out, batch_label,average='macro')
            f1_test_total += f1_test
            precision_test = precision_score(out, batch_label,average='macro')
            precision_test_total += precision_test
            recall_test = recall_score(out, batch_label,average='macro')
            recall_test_total += recall_test
            i += 1
        else:
            acc_test = accuracy_score(out, batch_label)
            acc_test_total += acc_test
            f1_test = f1_score(out, batch_label)
            f1_test_total += f1_test
            precision_test = precision_score(out, batch_label)
            precision_test_total += precision_test
            recall_test = recall_score(out, batch_label)
            recall_test_total += recall_test
            i += 1
    acc_test = acc_test_total/i
    f1_test = f1_test_total/i
    precision_test = precision_test_total/i
    recall_test = recall_test_total/i
    loss = loss_test_total/i
    
    return acc_test,f1_test,precision_test,recall_test,loss
    

# Main Function
if __name__ == "__main__":
    
    begin_time = time.localtime()
    begin_time = time.strftime('%Y-%m-%d %H:%M:%S',begin_time)
    print('Begining Time:',begin_time)
    
    # Parameter settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets_name', default="mgtab", type=str)
    parser.add_argument('--random_seed', type=int, default=[10,11,12,13,14], nargs='+', help='selection of random seeds')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of epochs.')
    parser.add_argument('--boolean_num', type=int, default=10, help='Number of boolean type feature.')
    parser.add_argument('--value_num', type=int, default=10, help='Number of value type feature.')
    parser.add_argument('--feature_dim', type=int, default=128, help='Number of attr dim.')
    parser.add_argument('--tweet_num', type=int, default=768, help='Number of tweet type feature.')
    parser.add_argument('--des_num', type=int, default=768, help='Number of deacription type feature.')
    parser.add_argument('--classes', type=int, default=2, help='Number of classes.')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay for optimizer')
    parser.add_argument('--batch_size', type=int, default=1024, help='Number of batch size')
    parser.add_argument('--name', type=str, default='attr', help='name of logger')
    parser.add_argument('--log_dir', type=str, default=' Log/model_edge_feature_1024_log.log', help='dir of logger')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of gnn layers.')
    parser.add_argument('--lambda_1', type=float, default=2, help='Value of lambda_1.')
    parser.add_argument('--lambda_2', type=float, default=1, help='Value of lambda_2.')
    parser.add_argument('--lambda_3', type=float, default=1, help='Value of lambda_3.')
    parser.add_argument('--model_file', type=str, default='model_edge_feature_1024', help='Save model dir.')
    
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    if args.datasets_name == "twibot-20":
        args.cat_dim = 11
        args.num_dim = 6
    if args.datasets_name == "cresci-15":
        args.cat_dim = 1
        args.num_dim = 5
    logger = get_logger(args.name,args.log_dir)
    logger.info('test logger')
    logger.info(f'Beginning Time:{begin_time}')

    # A total of 5 experiments will be carried out and the average performance will be obtained
    acc_list =[]
    precision_list = []
    recall_list = []
    f1_list = []
    for i,seed in enumerate(args.random_seed):
        torch.manual_seed(seed)
        # torch.cuda.manual_seed(seed)
        print('load data...')
        dataloader,train_idx, val_idx, test_idx = load_data(seed,args.batch_size, args.datasets_name)
       
        # Instancing Model
        model = AttrModel(args.datasets_name, args.value_num, args.boolean_num,args.tweet_num, args.des_num, args.feature_dim, args.classes, args.dropout, dataloader, args.num_layers, args.device)
        model = model.to(args.device)
        print('begin the {} training'.format(i+1))
        logger.info(f'begin the {i+1} training')
        train(args.datasets_name, args.boolean_num, args.value_num, args.tweet_num, args.des_num, dataloader,train_idx, val_idx, test_idx,model,args.num_epochs,args.lr,args.weight_decay,logger,args.lambda_1, args.model_file,args.batch_size,args.lambda_2,args.lambda_3,args.device)
        print('End the {} training'.format(i+1))
        logger.info(f'End the {i+1} training')

        # Load best model
        model.load_state_dict(torch.load(' Save_model/{}.pth'.format(args.model_file)))
        acc_test,f1_test,precision_test,recall_test,loss = test(args.datasets_name, args.boolean_num, args.value_num, args.tweet_num, args.des_num, dataloader, test_idx, model, loss_ce, args.lambda_1, args.batch_size, args.lambda_2, args.lambda_3, args.device)
        print("Test set Best results:",
          "test_accuracy= {:.4f}".format(acc_test),
          "precision= {:.4f}".format(precision_test),
          "recall= {:.4f}".format(recall_test),
          "f1_score= {:.4f}".format(f1_test))
        logger.info(f"Test set results: test_accuracy= {acc_test}, precision= {precision_test}, recall= {recall_test}, f1_score= {f1_test}")
        acc_list.append(acc_test*100)
        precision_list.append(precision_test*100)
        recall_list.append(recall_test*100)
        f1_list.append(f1_test*100)
    print('acc:       {:.2f} + {:.2f}'.format(np.array(acc_list).mean(), np.std(acc_list)))
    print('precision: {:.2f} + {:.2f}'.format(np.array(precision_list).mean(), np.std(precision_list)))
    print('recall:    {:.2f} + {:.2f}'.format(np.array(recall_list).mean(), np.std(recall_list)))
    print('f1:        {:.2f} + {:.2f}'.format(np.array(f1_list).mean(), np.std(f1_list))) 
    logger.info('acc:       {:.2f} + {:.2f}'.format(np.array(acc_list).mean(), np.std(acc_list)))
    logger.info('precision: {:.2f} + {:.2f}'.format(np.array(precision_list).mean(), np.std(precision_list)))
    logger.info('recall:    {:.2f} + {:.2f}'.format(np.array(recall_list).mean(), np.std(recall_list)))
    logger.info('f1:        {:.2f} + {:.2f}'.format(np.array(f1_list).mean(), np.std(f1_list))) 

    end_time = time.localtime()
    end_time = time.strftime('%Y-%m-%d %H:%M:%S',end_time)
    print('End Time:',end_time)
    logger.info(f'End Time:{end_time}')
    startTime= datetime.datetime.strptime(begin_time,"%Y-%m-%d %H:%M:%S")
    endTime= datetime.datetime.strptime(end_time,"%Y-%m-%d %H:%M:%S")
    m,s = divmod((endTime- startTime).seconds,60)
    h, m = divmod(m, 60)
    print(f'Spending Time:{h}h{m}m{s}s')
    logger.info(f'Spending Time:{h}h{m}m{s}s')

