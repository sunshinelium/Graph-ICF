'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Design training and test process
'''
from distutils.log import error
import world
import numpy as np
import torch
import utils
import dataloader
from dataloader import padding_dict_to_ndarray, padding_dict_to_ndarray_delpos, padding_dict_to_ndarray_test
from pprint import pprint
from utils import timer
from time import time
from tqdm import tqdm
import model
import multiprocessing
from sklearn.metrics import roc_auc_score


CORES = multiprocessing.cpu_count() // 2


def BPR_train_original(dataset, recommend_model, loss_class, epoch, neg_k=1, w=None):
    Recmodel = recommend_model
    Recmodel.train()
    bpr: utils.BPRLoss = loss_class
    
    with timer(name="Sample"):
        S = utils.UniformSample_original(dataset)
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    users = users.to(world.device)
    posItems = posItems.to(world.device)
    negItems = negItems.to(world.device)

    user_neibor_dict = dataset.user_neibor_dict
    # user_neibor_ndarray = padding_dict_to_ndarray(user_neibor_dict, dataset.n_users, world.hist_len)
    # user_neibor = torch.from_numpy(user_neibor_ndarray)
    # user_neibor = user_neibor.to(world.device)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    # delpos_time = 0
    if world.model_name in ['icf','graph_icf']:
        batch_delpos_start = time()
        neibor_delpos, n_neibor_index = utils.f_padding_dict_to_ndarray_delpos(user_neibor_dict, users, posItems, world.hist_len, dataset.m_items, world.del_pos)
        mask_mat = utils.f_pad_sequence(n_neibor_index, world.hist_len)
        neibor_delpos = torch.Tensor(neibor_delpos).long().to(world.device)
        n_neibor_index = torch.Tensor(n_neibor_index).long().to(world.device)
        # mask_mat = torch.Tensor(mask_mat).long().to(world.device)
        delpos_time = time() - batch_delpos_start
        print("delpos_time:%s"%delpos_time)
        dataloader = utils.minibatch(users,posItems,negItems, neibor_delpos,mask_mat,
                                    batch_size=world.config['bpr_batch_size'])
    elif world.model_name in ['mf','lgn']:   
        dataloader = utils.minibatch(users,posItems,negItems,
                                    batch_size=world.config['bpr_batch_size'])
    else:
        raise error           
    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    aver_loss, aver_bpr_loss, aver_icf_loss = 0.,0.,0.
    for (batch_i,data) in enumerate(dataloader):
        if world.model_name in ['icf','graph_icf']:
            batch_users, batch_pos, batch_neg, batch_neibor, batch_mask_mat=data
        elif world.model_name in ['mf','lgn']:   
            batch_users, batch_pos, batch_neg = data
            batch_neibor, batch_mask_mat = None,None
        label = torch.ones(len(batch_users)).to(world.device)
        # batch_delpos_start = time()
        # delpos_time = delpos_time + time() - batch_delpos_start
        # print(batch_n_neibor_index)
        if world.feature_attention_flag and epoch>=0:
            feature_attention_flag=1
        else:
            feature_attention_flag=0
        loss, bpr_loss, icf_loss = bpr.stageOne(batch_users, batch_pos, batch_neg, batch_neibor, batch_mask_mat, label,feature_attention_flag)
        aver_loss += loss
        aver_bpr_loss += bpr_loss
        aver_icf_loss += icf_loss
        if world.tensorboard:
            w.add_scalar(f'Loss/Loss', loss, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
            w.add_scalar(f'Loss/BPRLoss', bpr_loss, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
            w.add_scalar(f'Loss/ICFLoss', icf_loss, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
    # print("delpos_time:%s"%delpos_time)
    aver_loss = aver_loss / total_batch
    aver_bpr_loss = aver_bpr_loss / total_batch
    aver_icf_loss = aver_icf_loss / total_batch
    time_info = timer.dict()
    timer.zero()
    return f"loss{aver_loss:.3f} bpr_loss{aver_bpr_loss:.3f} icf_loss{aver_icf_loss:.3f}-{time_info}"
    
def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg, mAP = [], [], [], []
    for k in world.topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        # pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue,r,k))
        mAP.append(utils.MAP(groundTrue, r, k))
    return {'recall':np.array(recall), 
            # 'precision':np.array(pre), 
            'ndcg':np.array(ndcg),
            'mAP':np.array(mAP)
            }
        
            
def Test(writer, dataset, Recmodel, epoch, max_results, w=None, multicore=0):
    u_batch_size = world.config['test_u_batch_size']
    dataset: utils.LooLoader
    testDict: dict = dataset.testDict
    testItems = dataset.testItems
    Recmodel: model.LightGCN
    # eval mode with no dropout
    Recmodel = Recmodel.eval()
    max_K = max(world.topks)
    if world.feature_attention_flag and epoch>=0:
        feature_attention_flag=1
    else:
        feature_attention_flag=0
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    results = {'recall': np.zeros(len(world.topks)),
               'ndcg': np.zeros(len(world.topks)),
               'mAP': np.zeros(len(world.topks))}
    with torch.no_grad():
        users = list(testDict.keys())
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundTrue_list = []
        # auc_record = []
        # ratings = []
        test_items_arr = dataset.test_items_arr
        if world.model_name in ['icf','graph_icf']:
            user_neibor_dict = dataset.user_neibor_dict
            # np.random.shuffle(test_items_arr.transpose())
            # test_items_arr = test_items_arr.transpose()
            start = time()
            # user_neibor_ndarray, n_neibors_index = padding_dict_to_ndarray_test(user_neibor_dict, dataset.n_users, world.hist_len, dataset.m_items)
            user_neibor_ndarray, n_neibors_index = utils.f_padding_dict_to_ndarray_test(user_neibor_dict, dataset.n_users, world.hist_len, dataset.m_items)
            print("test_padding:%s"%(time()-start))
            mask_mat = utils.f_pad_sequence(n_neibors_index, world.hist_len)
            user_neibor = torch.Tensor(user_neibor_ndarray).long()
            user_neibor = user_neibor.to(world.device)
        elif world.model_name in ['mf','lgn']:   
            user_neibor, mask_mat = None,None
        # n_neibors_index = torch.Tensor(n_neibors_index).long().to(world.device)
        # mask_mat = torch.Tensor(mask_mat).long().to(world.device)
        # print(user_neibor, n_neibors_index)
        total_batch = len(users) // u_batch_size + 1
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.device)
            batch_test_items = test_items_arr[batch_users]
            batch_test_items_gpu = torch.Tensor(batch_test_items).long()
            batch_test_items_gpu = batch_test_items_gpu.to(world.device)
            # print(n_neibors_index)
            if world.model_name == 'lgn':
                rating = Recmodel.getUsersRating(batch_users_gpu, batch_test_items_gpu)
            elif world.model_name == 'icf':
                rating = Recmodel.getUsersRatingICF(batch_users_gpu, user_neibor, mask_mat, batch_test_items_gpu,epoch,feature_attention_flag)
            elif world.model_name == 'graph_icf':
                bpr_rating = Recmodel.getUsersRating(batch_users_gpu, batch_test_items_gpu)
                icf_rating = Recmodel.getUsersRatingICF(batch_users_gpu, user_neibor, mask_mat, batch_test_items_gpu)
                rating = (1 - world.ratio) * bpr_rating + world.ratio * icf_rating
                del bpr_rating
                del icf_rating
            else:
                raise NotImplementedError(f"Haven't supported {world.model_name} yet!, try {world.all_models}")
            # bpr_rating = Recmodel.getUsersRating(batch_users_gpu)
            # icf_rating = Recmodel.getUsersRatingICF(batch_users_gpu, user_neibor)
            # rating = (1 - world.ratio) * bpr_rating + world.ratio * icf_rating
            # rating.add_(icf_rating)
            #rating = rating.cpu()
            # exclude_index = []
            # exclude_items = []
            # for range_i, items in enumerate(allPos):
            #     exclude_index.extend([range_i] * len(items))
            #     exclude_items.extend(items)
            # rating[exclude_index, exclude_items] = -(1<<10)

            # loo choose 100 (1 test + 99 neg items) test items
            # if testItems is not None:
            #     batch_test_items = testItems[batch_users]
            #     rating[np.where(batch_test_items.todense() == 0)] = float('-inf')
            _, rating_K = torch.topk(rating, k=max_K)
            rating_K = torch.gather(batch_test_items_gpu,dim=1,index=rating_K)
            # rating = rating.cpu().numpy()
            # aucs = [
            #         utils.AUC(rating[i],
            #                   dataset,
            #                   test_data) for i, test_data in enumerate(groundTrue)
            #     ]
            # auc_record.extend(aucs)
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        if multicore == 1:
            pre_results = pool.map(test_one_batch, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x))
        scale = float(u_batch_size/len(users))
        for result in pre_results:
            results['recall'] += result['recall']
            # results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
            results['mAP'] += result['mAP']
        results['recall'] /= float(len(users))
        # results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        results['mAP'] /= float(len(users))
        # results['auc'] = np.mean(auc_record)
        for key, values in results.items():
            writer.write(str(key) + ": " + " ".join(map(str, values)) + ", ")
        writer.write("\n")
        writer.flush()
        for i in range(len(world.topks)):
            if results['recall'][i] > max_results['recall'][i]:
                max_results['recall'][i] = results['recall'][i]
                # max_results['precision'][i] = results['precision'][i]
                max_results['ndcg'][i] = results['ndcg'][i]
                max_results['mAP'][i] = results['mAP'][i]
        if world.tensorboard:
            w.add_scalars(f'Test/Recall@{world.topks}',
                          {str(world.topks[i]): results['recall'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/MAP@{world.topks}',
                          {str(world.topks[i]): results['mAP'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/NDCG@{world.topks}',
                          {str(world.topks[i]): results['ndcg'][i] for i in range(len(world.topks))}, epoch)
        if multicore == 1:
            pool.close()
        print(results)
        return results
