from model_pytorch import symcnn_model
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from create_dataset import Data_gener
import torch.autograd as autograd
import torch.optim as optim

import time
import numpy as np
from sklearn.metrics import roc_auc_score

BATCH_SIZE = 8196
EMBEDDING_DIM = 128
CONV_NUM_KENEL = 256
FC1_NUM = 128
FC2_NUM = 32
TRAIN_STEPS = 50001
INIT_LEARNING_RATE = 0.000005
DEBUG = False
EXPER_COMMENT = 'continue yesterdays training'

experiment_start_time = time.strftime('%Y-%m-%d-%H-%M',time.localtime(time.time()))
if DEBUG:
    f_log = open('/home/song/change_recommend_pytorch/logs/debug.log','w')
    f_log.write('experiment_start_time:\t'+experiment_start_time+'\n')
else:
    f_experiment_log = open('/home/song/change_recommend_pytorch/logs/experiment_directory.log','a')
    f_log = open('/home/song/change_recommend_pytorch/logs/exper-%s.log'%experiment_start_time,'w')
    f_experiment_log.write(experiment_start_time+': '+EXPER_COMMENT+'\n')
    f_experiment_log.close()
    f_log.write('experiment_start_time:\t'+experiment_start_time+'\n')

g = Data_gener('wine', batch_size = BATCH_SIZE)
gg = g.gener('train')
criterion = nn.MSELoss()
net = symcnn_model(embedding_dim = EMBEDDING_DIM, conv_num_kernel = CONV_NUM_KENEL, fc1_num = FC1_NUM, fc2_num = FC2_NUM).cuda()
#net.load_state_dict(torch.load('/home/song/change_recommend_pytorch/models/model-pars-first-night.pkl'))
net.load_state_dict(torch.load('/home/song/change_recommend_pytorch/models/model-0.5929-pars-2018-04-17-00-15.pkl'))
optimizer = optim.RMSprop(net.parameters(), lr=INIT_LEARNING_RATE)

def analysis_result(output, xy, label, loss, hit = False):
    mse_loss = float(loss.data)
    y_numpy  = output.cpu().data.numpy().squeeze()
    y_around = np.around(y_numpy).astype('int64')
    label_np = label.cpu().numpy().squeeze().astype('int64')
    acc = (np.dot(y_around, label_np)+np.dot(1-y_around, 1-label_np))/BATCH_SIZE
    auc = roc_auc_score(label_np, y_numpy)
    report_str = 'mse_loss: %.5f,\tacc: %.3f,\tauc: %.3f'%(mse_loss,acc,auc)
    if hit == True:
        sorted_ix = sorted(range(y_numpy.shape[0]), key = lambda x:y_numpy[x], reverse = True)
        hits_info = list(map(lambda x:label_np[x], sorted_ix[:10]))
        top_10_hit, top_10_hit_num = 1 if sum(hits_info) > 0 else 0, sum(hits_info)
        top_5_hit, top_5_hit_num = 1 if sum(hits_info[:5]) > 0 else 0, sum(hits_info[:5])
        top_3_hit, top_3_hit_num = 1 if sum(hits_info[:3]) > 0 else 0, sum(hits_info[:3])
        top_1_hit, top_1_hit_num = 1 if sum(hits_info[:1]) > 0 else 0, sum(hits_info[:1])
        return [report_str, mse_loss, acc, auc, [top_10_hit, top_5_hit, top_3_hit, top_1_hit], [top_10_hit_num, top_5_hit_num, top_3_hit_num, top_1_hit_num]]
    else:    
        return [report_str, mse_loss, acc, auc]

gt = g.gener('test')
xy_t = next(gg)
x_t = [autograd.Variable(i.cuda()) for i in xy_t[:2]]
label_t = xy_t[2].cuda()    
target_t = autograd.Variable(label_t)

def validation(val_interv):
    test_commits_size = len(g.test_commits)
    #test_commits_size = 200
    top_10_hits, top_5_hits = [[] for ix in val_interv], [[] for ix in val_interv]
    top_3_hits, top_1_hits = [[] for ix in val_interv], [[] for ix in val_interv]
    mse_losses, acces, auces = [[] for ix in val_interv], [[] for ix in val_interv], [[] for ix in val_interv]

    for commit_ix in val_interv:
        if commit_ix%100 == 0:
            print('test commit id:',commit_ix)
        for file_ix in range(len(g.test_commits[commit_ix][0])):
            left_samples, right_samples, label_samples = g.commit_validation_generation(commit_ix, file_ix)
            x = [autograd.Variable(i.cuda()) for i in [left_samples, right_samples]]
            label = label_samples.cuda()
            output = net(x)
            target = autograd.Variable(label)
            loss = criterion(output, target)

            report_str, mse_loss, acc, auc, \
            [top_10_hit, top_5_hit, top_3_hit, top_1_hit], \
            [top_10_hit_num, top_5_hit_num, top_3_hit_num, top_1_hit_num] \
            = analysis_result(output, [left_samples, right_samples, label_samples], label, loss, hit = True)
            top_10_hits[commit_ix].append(top_10_hit)
            top_5_hits[commit_ix].append(top_5_hit)
            top_3_hits[commit_ix].append(top_3_hit)
            top_1_hits[commit_ix].append(top_1_hit)
            mse_losses[commit_ix].append(mse_loss)
            acces[commit_ix].append(acc)
            auces[commit_ix].append(auc)
    top_10_hit = np.mean( list(map(lambda x:sum(x)/len(x), top_10_hits)) )
    top_5_hit  = np.mean( list(map(lambda x:sum(x)/len(x), top_5_hits)) )
    top_3_hit  = np.mean( list(map(lambda x:sum(x)/len(x), top_3_hits)) )
    top_1_hit  = np.mean( list(map(lambda x:sum(x)/len(x), top_1_hits)) )
    acc        = np.mean( list(map(lambda x:sum(x)/len(x), acces)) )
    auc        = np.mean( list(map(lambda x:sum(x)/len(x), auces)) )
    mse_loss   = np.mean( list(map(lambda x:sum(x)/len(x), mse_losses)) )
    validation_report = ''
    validation_report += 'top_10_hit:\t%.4f\n'%(top_10_hit)
    validation_report += 'top_5_hit:\t%.4f\n'%(top_5_hit)
    validation_report += 'top_3_hit:\t%.4f\n'%(top_3_hit)
    validation_report += 'top_1_hit:\t%.4f\n'%(top_1_hit)
    validation_report += 'acc:\t%.4f\n'%(acc)
    validation_report += 'auc:\t%.4f\n'%(auc)
    validation_report += 'mse_losses:\t%.4f\n'%(mse_loss)
    short_report = 'top_10_hit: %.4f, top_5_hit: %.4f, top_3_hit: %.4f, top_1_hit: %.4f, auc: %.4f'%(top_10_hit, top_5_hit, top_3_hit, top_1_hit, auc)
    return [validation_report, top_10_hit, top_5_hit, top_3_hit, top_1_hit, acc, auc, mse_loss, short_report]

for cnt in range(TRAIN_STEPS):
    xy = next(gg)
    x = [autograd.Variable(i.cuda()) for i in xy[:2]]
    label = xy[2].cuda()

    output = net(x)
    target = autograd.Variable(label)
    loss = criterion(output, target)
    
    if cnt%100 == 0:
        output_t = net(x_t)
        loss_t = criterion(output_t, target_t)
        short_report = validation(range(10))[8]
        temple_report = '%05d:  '%(cnt)+ analysis_result(output, xy, label, loss)[0] +'\t validation:\t' + short_report
        print(temple_report)
        f_log.write(temple_report + '\n')
    if cnt%5000 == 0 and cnt > 0:
        val_report_data = validation(range(len(g.test_commits)))
        val_report = val_report_data[8]
        top_10_hit = val_report_data[1]
        cur_time = time.strftime('%Y-%m-%d-%H-%M',time.localtime(time.time()))
        torch.save(net, '/home/song/change_recommend_pytorch/models/model-%.4f-%s.pkl'%(top_10_hit, cur_time)) 
        torch.save(net.state_dict(), '/home/song/change_recommend_pytorch/models/model-%.4f-pars-%s.pkl'%(top_10_hit, cur_time))
        f_log.write('model saved:\t%s\n'%cur_time)
        f_log.write(val_report)
        print(val_report)
    loss.backward()
    optimizer.step()

#val_report = validation()[0]
#f_log.write(val_report)
experiment_end_time = time.strftime('%Y-%m-%d-%H-%M',time.localtime(time.time()))
f_log.write('experiment_end_time:\t'+experiment_end_time+'\n')
f_log.close()

#for i in range(6):
#    print(times[i,:].sum())
