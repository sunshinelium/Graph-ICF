import world
import utils
from world import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
from os.path import join
import os
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset

def early_stopping(log_value, best_value, stopping_step, expected_order='acc', flag_step=100):
    # early stopping strategy:
    assert expected_order in ['acc', 'dec']

    if (expected_order == 'acc' and log_value < best_value) or (expected_order == 'dec' and log_value > best_value):
        stopping_step += 1

    if stopping_step >= flag_step:
        print("Early stopping is trigger at step: {} log:{}".format(flag_step, log_value))
        should_stop = True
    else:
        should_stop = False
    return stopping_step, should_stop

root_path = join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.model_name+'--%s'%world.comment)
# init tensorboard
if world.tensorboard:
    w : SummaryWriter = SummaryWriter(
                                    root_path
                                    )
else:
    w = None
    world.cprint("not enable tensorflowboard")

Recmodel = register.MODELS[world.model_name](world.config,w, dataset)
# # Parallel
# if torch.cuda.device_count() > 1:
#     print("Let's use", torch.cuda.device_count(), "GPUs!")
#     Recmodel = torch.nn.DataParallel(Recmodel)
#     Recmodel = Recmodel.module
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)

weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
Neg_k = 1


writer = open(root_path + "/log.txt", 'w')
writer.write('===========config================' + '\n')
writer.write("SEED: " + str(world.seed) + '\n')
writer.write("dataset: " + str(world.dataset) + '\n')
writer.write("model_name: " + str(world.model_name) + '\n')
writer.write("topK: " + str(world.topks) + '\n')
writer.write("hist_len: " + str(world.hist_len) + '\n')
writer.write("icf_ratio: " + str(world.icf_ratio) + '\n')
writer.write("use_neibor_emb: " + str(world.use_neibor_emb) + '\n')
writer.write("attention_flag: " + str(world.attention_flag) + '\n')
writer.write("feature_attention_flag: " + str(world.feature_attention_flag) + '\n')
writer.write("user_attention_mlp: " + str(world.user_attention_mlp) + '\n')
writer.write("del_pos: " + str(world.del_pos) + '\n')
for key, value in world.config.items():
    writer.write(str(key) + ": " + str(value) + '\n')
writer.write("cores for test: " + str(world.CORES) + '\n')
writer.write("comment: " + str(world.comment) + '\n')
writer.write("tensorboard: " + str(world.tensorboard) + '\n')
writer.write("LOAD: " + str(world.LOAD) + '\n')
writer.write("Weight path: " + str(world.PATH) + '\n')
writer.write("Test Topks: " + str(world.topks) + '\n')
writer.write("T: " + str(world.T) + '\n')
writer.write("alpha: " + str(world.alpha) + '\n')
writer.write("using bpr loss" + '\n')
writer.write('===========end===================' + '\n')

max_results = dict()
max_results['recall'] = [0.]*len(world.topks)
max_results['ndcg'] = [0.]*len(world.topks)
max_results['mAP'] = [0.]*len(world.topks)
stopping_step = 0

try:
    for epoch in range(world.TRAIN_epochs):
        start = time.time()
        if epoch % 5 == 0:
            cprint("[TEST]")
            writer.write("[TEST]" + '\n')
            result = Procedure.Test(writer, dataset, Recmodel, epoch, max_results, w, world.config['multicore'])
        if epoch % 20 == 0:
            writer.write(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] max_results \n')
            for key, value in max_results.items():
                writer.write(str(key) + ": " + " ".join(map(str, value)) + ", ")
            writer.write('\n')
        stopping_step,should_stop = early_stopping(result['recall'][0],max_results['recall'][0],stopping_step,flag_step=30)
        if should_stop:
            break
        time1=time.time()
        output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
        end_time = time.time()
        print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information} Train-Time: {end_time-time1}')
        writer.write("EPOCH" + str(epoch+1) + "/" + str(world.TRAIN_epochs) + " " + output_information + '\n')
        torch.save(Recmodel.state_dict(), weight_file)
    writer.write("===========max_results================" + '\n')
    for key, value in max_results.items():
        writer.write(str(key) + ": " + " ".join(map(str, value)) + ", ")
    writer.write('\n')
    writer.write("===========max_results end================" + '\n')
finally:
    if world.tensorboard:
        w.close()
