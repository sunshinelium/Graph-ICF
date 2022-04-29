'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Go lightGCN")
    parser.add_argument('--bpr_batch', type=int,default=2048,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--recdim', type=int,default=64,
                        help="the embedding size of lightGCN")
    parser.add_argument('--layer', type=int,default=3,
                        help="the layer num of lightGCN")
    parser.add_argument('--lr', type=float,default=0.001,
                        help="the learning rate")
    parser.add_argument('--decay', type=float,default=1e-4,
                        help="the weight decay for l2 normalizaton")
    parser.add_argument('--dropout', type=int,default=0,
                        help="using the dropout or not")
    parser.add_argument('--keepprob', type=float,default=0.6,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--a_fold', type=int,default=100,
                        help="the fold num used to split large adj matrix, like gowalla")
    parser.add_argument('--testbatch', type=int,default=50,
                        help="the batch size of users for testing")
    parser.add_argument('--dataset', type=str,default='ml-1m',
                        help="available datasets: [lastfm, gowalla, yelp2018, amazon-book]")
    parser.add_argument('--path', type=str,default="./checkpoints",
                        help="path to save weights")
    parser.add_argument('--topks', nargs='?',default="[5,10,20]",
                        help="@k test list")
    parser.add_argument('--tensorboard', type=int,default=1,
                        help="enable tensorboard")
    parser.add_argument('--comment', type=str,default="")
    parser.add_argument('--load', type=int,default=0)
    parser.add_argument('--epochs', type=int,default=1000)
    parser.add_argument('--multicore', type=int, default=0, help='whether we use multiprocessing or not in test')
    parser.add_argument('--pretrain', type=int, default=0, help='whether we use pretrained weight or not')
    parser.add_argument('--seed', type=int, default=2020, help='random seed')
    parser.add_argument('--model', type=str, default='icf', help='rec-model, support [mf, lgn, icf, graph_icf]')
    parser.add_argument('--icf_ratio', type=float, default=0.5, help='icf loss/scores ratio')
    parser.add_argument('--beta', type=float, default=0.5, help='power weight of exp in attention sum')
    parser.add_argument('--hist_len', type=int, default=100, help='user interaction history list')
    parser.add_argument('--use_neibor_emb', type=int, default=0, help='use neibor emb or not')
    parser.add_argument('--attention_flag', type=int, default=1, help='use attention or not')
    parser.add_argument('--feature_attention_flag', type=int, default=1, help='use attention or not')
    parser.add_argument('--user_attention_mlp', type=int, default=0, help='use attention or not')
    parser.add_argument('--del_pos', type=int, default=1, help='del pos item of test items or not')
    parser.add_argument('--T', type=int, default=5, help='del pos item of test items or not')
    parser.add_argument('--alpha', type=float, default=0.3, help='control fenmu in fa')
    parser.add_argument('--attention_type', type=str, default='softmax', help='control fenmu in fa')
    return parser.parse_args()
