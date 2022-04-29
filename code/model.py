"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Define models here
"""
import world
import torch
from dataloader import BasicDataset,Loader,padding_dict_to_ndarray
from torch import nn
import numpy as np
import copy
from collections import defaultdict
from utils import f_pad_sequence

class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def getUsersRating(self, users):
        raise NotImplementedError
    
class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()
    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError
    
class PureMF(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(PureMF, self).__init__()
        self.num_users  = dataset.n_users
        self.num_items  = dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.f = nn.Sigmoid()
        self.__init_weight()
        
    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        print("using Normal distribution N(0,1) initialization for PureMF")
        
    def getUsersRating(self, users):
        users = users.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)
    
    def bpr_loss(self, users, pos, neg):
        users_emb = self.embedding_user(users.long())
        pos_emb   = self.embedding_item(pos.long())
        neg_emb   = self.embedding_item(neg.long())
        pos_scores= torch.sum(users_emb*pos_emb, dim=1)
        neg_scores= torch.sum(users_emb*neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1/2)*(users_emb.norm(2).pow(2) + 
                          pos_emb.norm(2).pow(2) + 
                          neg_emb.norm(2).pow(2))/float(len(users))
        return loss, reg_loss
        
    def forward(self, users, items):
        users = users.long()
        items = items.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item(items)
        scores = torch.sum(users_emb*items_emb, dim=1)
        return self.f(scores)

class LightGCN(nn.Module):
    def __init__(self, 
                 config:dict, w,
                 dataset:Loader):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset : dataloader.Loader = dataset
        self.w = w
        self.__init_weight()

    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.beta = self.config['beta']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        if world.use_neibor_emb:
            self.embedding_item_neibor = torch.nn.Embedding(
                num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        if world.attention_flag:
            # self.mlp_layer = nn.Linear(self.latent_dim, self.latent_dim//2).to(world.device)
            self.out_layer = nn.Linear(self.latent_dim, 1).to(world.device)
        if self.config['pretrain'] == 0:
#             nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
#             nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
#             print('use xavier initilizer')
#             random normal init seems to be a better choice when lightGCN actually don't use any non-linear activation function
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            if world.use_neibor_emb:
                nn.init.normal_(self.embedding_item_neibor.weight, std=0.1)
            if world.attention_flag:
                # nn.init.trunc_normal_(self.mlp_layer.weight)
                nn.init.ones_(self.out_layer.weight)
            world.cprint('use NORMAL distribution initilizer')
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            if world.use_neibor_emb:
                self.embedding_item_neibor.weight.data.copy_(torch.from_numpy(self.config['item_emb_neibor']))
            if world.attention_flag:
                # self.mlp_layer.weight.data.copy_(torch.from_numpy(self.config['mlp_layer']))
                self.out_layer.weight.data.copy_(torch.from_numpy(self.config['out_layer']))
            print('use pretarined data')
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        print(f"lgn is already to go(dropout:{self.config['dropout']})")

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph
    
    def computer(self):
        """
        propagate methods for lightGCN
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        # items_emb_neibor = self.embedding_item_neibor.weight
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        if self.config['dropout']:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph        
        else:
            g_droped = self.Graph    
        
        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        #print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def _attention_MLP(self, q_, mask_mat):
        # input mul with neibor embedding mat and target item(b,n,e), 
        # mask_mat 应该标注了邻居的位置，因为之前采用了填充的方式，所以需要利用mask排除无关id的影响。
        # return attention mat(b,n,1).
        b = q_.shape[0]
        n = q_.shape[1]
        r = self.latent_dim
        ###############################MLP layer in attention module#########################
        if world.user_attention_mlp:
            q_ = torch.reshape(q_, [-1, r])
            # q_ = self.mlp_layer(q_)  # (b*n, e or 2*e) * (e or 2*e, w) + (1, w)
            # q_ = nn.ReLU()(q_)
            attention = self.out_layer(q_)     #(b*n, w) * (w, 1) => (None, 1)
        else:
            attention = q_.sum([2])
        A_ = torch.reshape(attention, [b, n]) #(None, 1) => (b, n)
        # softmax for not mask features
        exp_A_ = torch.exp(A_)    
        # mask_mat = f_pad_sequence(n_neibors_index, n)
        exp_A_ = mask_mat * exp_A_
        exp_sum = torch.unsqueeze(torch.sum(exp_A_, 1), 1)  # (b, 1)
        exp_sum = torch.pow(exp_sum, self.beta)
        #
        A_ = torch.unsqueeze(torch.div(exp_A_, exp_sum+ 1e-8), 2)  # (b, n, 1)
        # A_ = torch.unsqueeze(nn.functional.softmax(A_, 1), 2)
        return A_

    def _attention_MLP_4D(self, q_, mask_mat):
        # input mul with neibor embedding mat and target item(b,n_items,n_neibors,e), n_neibors_index:[batch_size,1]
        # return attention mat(b,n_items,n_neibors,1).
        b = q_.shape[0]
        n_items = q_.shape[1]
        n_neibors = q_.shape[2]
        r = self.latent_dim
        assert r == q_.shape[3], 'check mat shape'
        #################################MLP layer in attention module######################
        if world.user_attention_mlp:
            # q_ = torch.reshape(q_, [-1, r])
            # q_ = self.mlp_layer(q_)
            # q_ = nn.ReLU()(q_)
            attention = self.out_layer(q_)     #(b*n, w) * (w, 1) => (None, 1)
        else:
            attention = q_.sum([3])
        A_ = torch.reshape(attention, [b, n_items, n_neibors]) #(None, 1) => (b, n_items, n_neibors)

        # # softmax for not mask features
        exp_A_ = torch.exp(A_)
        # n_neibors_index = torch.reshape(n_neibors_index, [-1]) #[b* n_neibors]
        # mask_mat = f_pad_sequence(n_neibors_index, n_neibors)
        mask_mat = torch.unsqueeze(mask_mat, 1) #[b,1,n_neibors]
        exp_A_ = mask_mat * exp_A_ #[b,1,n_neibors]*[b,n_items,n_neibors]
        exp_sum = torch.unsqueeze(torch.sum(exp_A_, 2), 2)  # (b, n_items, 1)
        exp_sum = torch.pow(exp_sum, self.beta)

        A_ = torch.unsqueeze(torch.div(exp_A_, exp_sum + 1e-8), 3)  # (b, n_items, n_neibors, 1 )
        # A_ = torch.unsqueeze(nn.functional.softmax(A_, 2), 3)
        return A_

    def std_mapping(simi_feature):
        # input mul with neibor embedding mat and target item(b,n_items,n_neibors,e)
        # return fuse simi feature mat(b,n_items,1,e).
        # sqrt(mean(x^2,2)-mean(x,2)^2)+1e-8
        user_neibor_emb = torch.sqrt(torch.mean(torch.pow(simi_feature,2),2)-torch.pow(torch.mean(simi_feature,2))+1e-8)
        return user_neibor_emb
    def getUsersRating(self, users, items):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]  # [batch, dim]
        users_emb = torch.unsqueeze(users_emb, 1)  # [batch, 1, dim]
        items_emb = all_items[items.long()]  # [batch, test_items, dim]
        rating = self.f(torch.squeeze(torch.bmm(users_emb, items_emb.permute(0, 2, 1))))  # [batch, items]
        return rating

    def getUsersRatingICF(self, users, user_neibor, mask_mat, items,epoch,feature_attention_flag):
        all_users, all_items = self.computer()
        item_emb = all_items[items.long()]  # [batch, items, dim]
        user_neibor = user_neibor[users]  # [batch_size, neibor_nums]
        # item_emb = torch.unsqueeze(item_emb, 0) # [1, all_items, dim]
        # user_neibor_emb = all_items[user_neibor]  # [batch_size, neibor_nums, dim]
        # use_neibor_emb
        mask_embed = torch.zeros(1, self.latent_dim).to(world.device)
        if world.use_neibor_emb:
            neibor_emb = self.embedding_item_neibor.weight
            neibor_emb = torch.cat([neibor_emb, mask_embed], dim=0)
            user_neibor_emb = neibor_emb[user_neibor]
        else:
            all_items = torch.cat([all_items, mask_embed], dim=0)
            user_neibor_emb = all_items[user_neibor]  # [batch_size, neibor_nums, dim]
        
        user_neibor_emb = torch.unsqueeze(user_neibor_emb, 1)  # [batch_size, 1, neibor_nums, dim]
        mask_mat = mask_mat[users] # [batch_size,neibor]
        item_emb = torch.unsqueeze(item_emb,2)
        if world.attention_type=='softmax':
            if world.attention_flag:
                attention = self._attention_MLP_4D(item_emb.mul(user_neibor_emb), mask_mat) #[batch_size, all_items,neibor_num,1]
            else:
                attention = torch.unsqueeze(mask_mat, 1) #[b,1,neibor]
                attention = torch.unsqueeze(attention, 3) #[b,1,neibor,1]
            user_neibor_emb = attention.mul(user_neibor_emb) #[b,n_item,n,e]
        elif world.attention_type == 'std':
            user_neibor_emb = self.std_mapping(item_emb.mul(user_neibor_emb))
        # elif world.attention_type == 'hybrid':
        #     user_neibor_emb
        if feature_attention_flag:
            # feature_attentions = 64*nn.functional.softmax(all_users/5, 1)
            # exp_user = abs(all_users)
            exp_user = torch.exp(all_users/world.T) #T=5
            user_sum = torch.unsqueeze(torch.sum(exp_user,1),1)
            user_sum = torch.pow(user_sum, world.alpha)
            feature_attentions = torch.div(exp_user,user_sum)
            feature_attention = feature_attentions[users.long()]  # [batch_size, dim]
            feature_attention = torch.unsqueeze(feature_attention, 1) #[b,1,e]
            feature_attention = torch.unsqueeze(feature_attention, 2)
            ui_mul = feature_attention.mul(item_emb) #[b,1,1,e]*[b,n_item,1,e]
            icf_rating = ui_mul.mul(user_neibor_emb)
            if world.tensorboard:
                self.w.add_histogram(f'normed user feature',feature_attentions, epoch)
                self.w.add_histogram(f'non-normed user feature',all_users, epoch)
            del ui_mul
        else:
            icf_rating = item_emb.mul(user_neibor_emb)
        del item_emb
        icf_rating = icf_rating.sum([2, 3])
        icf_rating = self.f(icf_rating)

        return icf_rating

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
        return loss, reg_loss
    
    def icf_loss(self,users, pos, user_neibor, mask_mat, neg,feature_attention_flag):
        all_users, all_items = self.computer()
        item_feature = all_items[pos]
        neg_item_feature = all_items[neg]
        mask_embed = torch.zeros(1, self.latent_dim).to(world.device)
        if world.use_neibor_emb:
            neibor_emb = self.embedding_item_neibor.weight
            neibor_emb = torch.cat([neibor_emb, mask_embed], dim=0)
            neibors_feature = neibor_emb[user_neibor]
        else:
            all_items = torch.cat([all_items, mask_embed], dim=0)
            neibors_feature = all_items[user_neibor]
        # pos score
        item_feature = torch.unsqueeze(item_feature, 1)
        if world.attention_flag:
            attention_pos = self._attention_MLP(item_feature.mul(neibors_feature), mask_mat) # (b, n, 1)
        else:
            attention_pos = torch.unsqueeze(mask_mat, 2) # (b, n, 1)
        neibors_feature_pos = attention_pos.mul(neibors_feature) #(b,n,1)*(b,n,e)->(b,n,e)
        if feature_attention_flag:
            # feature_attentions = 64*nn.functional.softmax(all_users/5, 1)
            # exp_user = abs(all_users)
            exp_user = torch.exp(all_users/world.T)
            user_sum = torch.unsqueeze(torch.sum(exp_user,1),1)
            user_sum = torch.pow(user_sum, world.alpha)
            feature_attentions = torch.div(exp_user,user_sum)
            feature_attention = feature_attentions[users.long()]  # [batch_size, dim]
            feature_attention = torch.unsqueeze(feature_attention, 1) #[b,1,e]
            ui_mul = feature_attention.mul(item_feature) #[b,1,1,e]*[b,1,e]
            icf_pos_scores = ui_mul.mul(neibors_feature_pos)
        else:
            icf_pos_scores = item_feature.mul(neibors_feature_pos)
        icf_pos_scores = icf_pos_scores.sum([1,2])
        # neg score
        neg_item_feature = torch.unsqueeze(neg_item_feature, 1)
        if world.attention_flag:
            attention_neg = self._attention_MLP(neg_item_feature.mul(neibors_feature), mask_mat)
        else:
            attention_neg = torch.unsqueeze(mask_mat, 2) # (b, n, 1)
        neibors_feature_neg = attention_neg.mul(neibors_feature) #(b,n,1)*(b,n,e)->(b,n,e)
        if feature_attention_flag:
            ui_mul_neg = feature_attention.mul(neg_item_feature) #[b,1,e]*[b,1,e]
            icf_neg_scores = ui_mul_neg.mul(neibors_feature_neg)
        else:
            icf_neg_scores = neg_item_feature.mul(neibors_feature_neg)
        icf_neg_scores = icf_neg_scores.sum([1,2])

        # BPR loss
        icf_loss = torch.mean(torch.nn.functional.softplus(icf_neg_scores - icf_pos_scores))
        # RMSE loss
        # critical = torch.nn.MSELoss()
        # # print(icf_pos_scores.shape,label.shape)
        # icf_loss = torch.sqrt(critical(icf_pos_scores,label))
        return icf_loss

    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        # print('forward')
        #all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)
        return gamma
