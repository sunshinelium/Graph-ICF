import world
import dataloader
import model
import utils
from pprint import pprint

if world.dataset in ['gowalla', 'yelp2018', 'amazon-book']:
    dataset = dataloader.Loader(path="../data/"+world.dataset)
elif world.dataset == 'lastfm':
    dataset = dataloader.LastFM()
elif world.dataset in ['ml-1m', 'pinterest-20','yelp']:
    dataset = dataloader.LooLoader(path='../data/'+world.dataset)

print('===========config================')
print("hist_len:", world.hist_len)
print("use_neibor_emb:", world.use_neibor_emb)
print("attention_flag:", world.attention_flag)
print("del_pos:", world.del_pos)
print("icf_ratio:", world.icf_ratio)
print('T:',world.T)
print('alpha:',world.alpha)
pprint(world.config)
print("cores for test:", world.CORES)
print("comment:", world.comment)
print("tensorboard:", world.tensorboard)
print("LOAD:", world.LOAD)
print("Weight path:", world.PATH)
print("Test Topks:", world.topks)
print("using bpr loss")
print('===========end===================')

MODELS = {
    'mf': model.PureMF,
    'lgn': model.LightGCN,
    'icf': model.LightGCN,
    'graph_icf': model.LightGCN
}