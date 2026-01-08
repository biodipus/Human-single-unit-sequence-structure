# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 18:05:28 2024

@author: Wen
"""

get_ipython().run_line_magic('reset', '-sf')
style_path = r'C:\Wen\OneDrive\CodeHub\Python\style_paper.mplstyle'

import numpy as np
import pandas as pd
import os
import scipy.io as sio
from scipy import stats
import scipy
import matplotlib.pyplot as plt
# import h5py
import pickle as pkl
import glob
import random

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import svm
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, RepeatedKFold, KFold,LeaveOneOut
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, ConfusionMatrixDisplay
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from scipy.ndimage import gaussian_filter1d
from numpy.lib.stride_tricks import sliding_window_view

#%% function 
def remove_baseline(x):
    basemean = np.nanmean(x[:,:,:10], axis=2)
    for t in range(x.shape[2]):
        x[:,:,t] -= basemean 
    return x

def plot_acc_mat(scores_mean, rank, mkts_merge, figName):
    # scores_mean = scoresAll[rank-1].mean(0).mean(0)
    with plt.style.context(style_path):
        fig = plt.figure(figsize = (1.8, 1.6), dpi = 300)
        ax1 = fig.add_subplot(1, 1, 1)
        h1 = ax1.pcolormesh(scores_mean, cmap = plt.cm.RdBu_r, vmin=.15, vmax=.35)
        # ttime = scores_mean.shape[0]
        ax1.vlines(mkts_merge[0], 0, mkts_merge[1][-1], color='w', linestyle = '-')
        ax1.hlines(mkts_merge[0], 0, mkts_merge[1][-1], color='w', linestyle = '-')
        # ax1.vlines([t1off, t2off], [0]*len(mkts_merge[0]), [ttime]*len(mkts_merge[0]), color='gray', linestyle = '--', linewidth=1)
        # ax1.hlines([t1off, t2off], [0]*len(mkts_merge[0]), [ttime]*len(mkts_merge[0]), color='gray', linestyle = '--', linewidth=1)
        # ax1.set_xticks(mkts_merge[0])
        # ax1.set_xticklabels(mktime_merge[0])
        
        # ax1.set_yticks(mkts_merge[0])
        # ax1.set_yticklabels(mktime_merge[0])
    
        ax1.set_xlabel('Test time')
        ax1.set_ylabel('Train time')
        ax1.set_title(figName)
        fig.colorbar(h1)
        plt.show()
        
def decoding_chunk():
    # decoding
    scoresAll = [None]
    fea_weight = [None]
    
    x = fr_all.copy() # neuron*trial*time
    y = yy.copy()
    scoresAll = fun_ct_decoding_target.ct_decoding(x, y, train_r=0, test_r=0)
    
    scores_mean = scoresAll.mean(1).mean(0)
    figName = monkey + 'Rank' + str(i+1) + ' frontal_' + osc_band[bandid]
    # fun_ct_decoding_target.plot_acc_mat(scores_mean, i, mkts_merge, figName)
    plot_acc_mat(scores_mean, i, mkts_merge, figName)
    
def moving_average_3d(arr, window_size):
    """
    Compute the moving average along the third axis of a 3D array.
    
    Parameters:
        arr (np.ndarray): Input 3D array.
        window_size (int): Size of the moving window.
        
    Returns:
        np.ndarray: 3D array with the moving average computed along the third axis.
    """
    if window_size < 1:
        raise ValueError("Window size must be at least 1.")
    
    # Get the shape of the input array
    d1, d2, d3 = arr.shape
    
    # Initialize an array to store the moving averages
    ma_arr = np.zeros((d1, d2, d3 - window_size + 1))
    
    # Calculate the moving average along the third axis
    for i in range(d3 - window_size + 1):
        ma_arr[:, :, i] = np.mean(arr[:, :, i:i+window_size], axis=2)
    
    return ma_arr  

def mergeSub_noBoot(x):
    task = trialInfo[0]['task'].unique()
    psudoTrial_num = 8
    
    task_allsub = []
    fr_allsub_alltask = []
    for i in range(len(task)):
        task_allsub.append(task[i])
        fr_allsub = [None]*len(subs)
        for s in range(len(subs)): # len(subs)
            idx = trialInfo[s]['task'] == task[i]
            tmp = x[s][:,idx,:]
            tmp2 = np.zeros([x[s].shape[0], psudoTrial_num, x[s].shape[2]])
            tmp2 = tmp[:,np.random.choice(tmp.shape[1], size=8, replace=False),:]
            
            fr_allsub[s] = tmp2
        fr_allsub = np.concatenate(fr_allsub, axis=0)
        fr_allsub_alltask.append(fr_allsub)
    
    fr_allsub_alltask = np.concatenate(fr_allsub_alltask, axis=1)
    trialInfo_allsub = pd.DataFrame({'task':np.repeat(task_allsub,psudoTrial_num)})
    
    fr_allsub_alltask = fr_allsub_alltask[:,:,:]
    chNum, trialNum, timeNum = fr_allsub_alltask.shape
    
    for n in range(chNum):
        for tr in range(trialNum):
            fr_allsub_alltask[n,tr,:] = gaussian_filter1d(fr_allsub_alltask[n,tr,:], 1)
            
    return fr_allsub_alltask, trialInfo_allsub
#%% load FR data
path = r'F:\Human_chunking_Huashan_intraoperative\Data\Recording\PFC'
# data = pkl.load(open(path+'\\fr_good_in3task(245).pkl', 'rb'))
# data = pkl.load(open(path+'\\fr_good_in3task(145).pkl', 'rb'))
# data = pkl.load(open(path+'\\fr_beha_PFC(9sub_153)_witherror.pkl', 'rb'))
# data = pkl.load(open(path+'\\fr_beha_PFC(9sub_155)_errormark_drift.pkl', 'rb'))
# data = pkl.load(open(path+'\\fr_beha_PFC(9sub_155)_errormark_drift_new.pkl', 'rb'))
data = pkl.load(open(path+'\\fr_beha_PFC(10sub_160)_errormark_drift_new2.pkl', 'rb'))

fr = data['fr'] # neuron*trial*time
neu_info = data['neu_info']
trialInfo = data['trialInfo']
window = data['window']
step = data['step']
subs= data['subs']
mk_ts = data['mk_ts']

aline_ts = int(window/step/2)
mk_ts = mk_ts #-aline_ts
neuNum = sum([len(i) for i in neu_info])
mk_name = ['0', '2', '4','Resp']

#### 选择正确/错误trial
sel_trialtype = 0
if sel_trialtype ==1:
    trialInfo_correct = [None] * len(subs)
    fr_correct = [None] * len(subs)
    for sub in range(len(subs)):
        x = fr[sub].copy()
        info = trialInfo[sub].copy()
        
        idx = info['acc'] == 1
        trialInfo_correct[sub] = info.loc[idx,:]
        fr_correct[sub] = x[:,idx,:]
        
    fr = fr_correct
    trialInfo = trialInfo_correct
    
#### bin 成8个点
for s in range(len(fr)):
    x = fr[s]
    tmp = np.zeros(x.shape)[:,:,:8]
    for i in range(8):
        win = [mk_ts[0]+i*10+0, mk_ts[0]+i*10+5]
        # print(win)
        tmp[:,:,i] = x[:,:,win[0]:win[1]].mean(2)
    fr[s] = tmp

# for s in range(len(fr)):
#     tmp = fr[s][:,:,10:94]
#     tt = sliding_window_view(tmp, window_shape=5, axis=2)
#     fr[s] = tt[:,:,::2].mean(axis=-1)
#%% merge data, boot, svm, within chunk decoding
from sklearn.metrics import roc_auc_score

def perm_data(dotask):
    task = ['geo_ir', 'geo_r4', 'geo_r2']
    psudoTrial_num = 8
    
    task_allsub = []
    fr_allsub_alltask = []
    for i in range(len(task)):
        task_allsub.append(task[i])
        fr_allsub = [None]*len(subs)
        for s in range(len(subs)):
            idx = trialInfo[s]['task'] == task[i]
            tmp = fr[s][:,idx,:].copy()
            # print(tmp.shape)
            tmp2 = np.zeros([fr[s].shape[0], psudoTrial_num, fr[s].shape[2]])
            if tmp.shape[1]>0:
                # for r in range(psudoTrial_num):
                tmp2[:,:,:] = tmp[:,np.random.default_rng().choice(tmp.shape[1], size=8, replace=False),:]
                    # tmp2[:,r,:] = tmp[:,np.random.default_rng().choice(tmp.shape[1], size=12, replace=True),:].mean(1)
            fr_allsub[s] = tmp2
        fr_allsub = np.concatenate(fr_allsub, axis=0)
        fr_allsub_alltask.append(fr_allsub)
    
    fr_allsub_alltask = np.concatenate(fr_allsub_alltask, axis=1)
    trialInfo_allsub = pd.DataFrame({'task':np.repeat(task_allsub,psudoTrial_num)})
    
    fr_allsub_alltask = fr_allsub_alltask[:,:,:]
    chNum, trialNum, timeNum = fr_allsub_alltask.shape
    
    #### Norm
    # print(dotask)
    sel_task = trialInfo_allsub.query('task.str.contains(@dotask)').index.values
    trialInfo_seltask = trialInfo_allsub.iloc[sel_task,:].reset_index()
    fr_allsub_seltask = fr_allsub_alltask[:,sel_task,:]
    
    scaler = StandardScaler()
    x = fr_allsub_seltask[:,:,:].copy()
    shape_3d = x.shape
    xx = x.reshape(shape_3d[0], -1)
    x = scaler.fit_transform(xx.T).T
    return x


####
task = ['geo_ir', 'geo_r4', 'geo_r2']
taskid = 2
shuffle = 1

scores_mean_perm = []
cm_mean_perm = []
for repeat in range(1000):
    x = perm_data(task[taskid])
    #### svm
    y = np.tile(np.repeat([1,2,3,4,5,6,7,8], 1), int(x.shape[1]/8))
    y = y.reshape([-1,1])
    
    C = 1
    gamma = 1
    n_fold = 5
    n_repeats = 10
    
    clf = svm.SVC(kernel='linear', C=C, gamma=gamma,probability=True)
    scores =  np.full((n_repeats, n_fold), np.nan)
    cm_all =  np.full((8,8,n_repeats, n_fold), np.nan)
    for r in range(n_repeats):
        if shuffle == 1: 
            np.random.shuffle(y)
            
        rkf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=None)  # for within rank
        # rkf = KFold(n_splits=n_fold, shuffle=True, random_state=None)  # for cross rank
        for i, (train, test) in enumerate(rkf.split(x[:,:].T, y[:,0])):
            y_train = y[train, 0]
            
            x_train = x[:,train].T
            clf.fit(x_train, y_train)
                
            x_test = x[:,test].T
            y_test = y[test, 0]
                    
            scores[r,i] = clf.score(x_test, y_test)
            
            # 得到预测概率（用于 AUC）
            # y_score = clf.predict_proba(x_test)
            # auc = roc_auc_score(y_test, y_score, multi_class='ovo')
            # scores[r,i] = auc
                
            predictions = clf.predict(x_test)
            tmp  = confusion_matrix(y_test, predictions, labels=clf.classes_)
            aa = np.sum(tmp,1)
            aa = np.tile(aa, (8, 1)).T
            tmp = tmp/aa
            cm_all[:,:,r,i] = tmp
            
    scores_mean = scores.mean(0).mean(0)
    scores_mean_perm.append(scores_mean)
    cm_mean_perm.append(cm_all.mean(-1).mean(-1))

scores_mean_perm = np.array(scores_mean_perm)
print(np.nanmean(scores_mean_perm))
cm = np.stack(cm_mean_perm)
cm_mean = cm.mean(0)

if shuffle == 0:
    pkl.dump({'cm':cm, 'scores_mean_perm':scores_mean_perm}, 
             open(r'F:\Human_chunking_Huashan_intraoperative\Result\decoding_rank\\'+task[taskid]+'(perm100)_t2_5_new.pkl','wb'))
if shuffle == 1:
    pkl.dump({'cm':cm, 'scores_mean_perm':scores_mean_perm}, 
             open(r'F:\Human_chunking_Huashan_intraoperative\Result\decoding_rank\\'+task[taskid]+'_sf(perm1000_10)_t2_5_new.pkl','wb'))    
#%% merge data, boot, cross chunk decoding
from sklearn.metrics import roc_auc_score

def perm_data(dotask):
    task = ['geo_ir', 'geo_r4', 'geo_r2']
    psudoTrial_num = 8
    
    task_allsub = []
    fr_allsub_alltask = []
    for i in range(len(task)):
        task_allsub.append(task[i])
        fr_allsub = [None]*len(subs)
        for s in range(len(subs)):
            idx = trialInfo[s]['task'] == task[i]
            tmp = fr[s][:,idx,:].copy()
            # print(tmp.shape)
            tmp2 = np.zeros([fr[s].shape[0], psudoTrial_num, fr[s].shape[2]])
            if tmp.shape[1]>0:
                # for r in range(psudoTrial_num):
                tmp2[:,:,:] = tmp[:,np.random.default_rng().choice(tmp.shape[1], size=8, replace=False),:]
                    # tmp2[:,r,:] = tmp[:,np.random.default_rng().choice(tmp.shape[1], size=12, replace=True),:].mean(1)
            fr_allsub[s] = tmp2
        fr_allsub = np.concatenate(fr_allsub, axis=0)
        fr_allsub_alltask.append(fr_allsub)
    
    fr_allsub_alltask = np.concatenate(fr_allsub_alltask, axis=1)
    trialInfo_allsub = pd.DataFrame({'task':np.repeat(task_allsub,psudoTrial_num)})
    
    fr_allsub_alltask = fr_allsub_alltask[:,:,:]
    chNum, trialNum, timeNum = fr_allsub_alltask.shape
    
    #### Norm
    # print(dotask)
    sel_task = trialInfo_allsub.query('task.str.contains(@dotask)').index.values
    trialInfo_seltask = trialInfo_allsub.iloc[sel_task,:].reset_index()
    fr_allsub_seltask = fr_allsub_alltask[:,sel_task,:]
    
    scaler = StandardScaler()
    x = fr_allsub_seltask[:,:,:].copy()
    shape_3d = x.shape
    xx = x.reshape(shape_3d[0], -1)
    x = scaler.fit_transform(xx.T).T
    return x


####
task = ['geo_ir', 'geo_r4', 'geo_r2']
taskid = 1
shuffle = 1

scores_mean_perm = []
cm_mean_perm = []
for repeat in range(1000):
    x = perm_data(task[taskid])
    y = np.tile(np.repeat([1,2,3,4,5,6,7,8], 1), int(x.shape[1]/8))
    y = y.reshape([-1,1])
    
    # xtrian = x[:,y[:,0]<5]
    # xtest = x[:,y[:,0]>4]
    
    # ytrain = y[y[:,0]<5]
    # ytest= y[y[:,0]>4]-4
    
    xtest = x[:,y[:,0]<5]
    xtrian = x[:,y[:,0]>4]
    
    ytest = y[y[:,0]<5]
    ytrain= y[y[:,0]>4]-4
    #### svm
    C = 1
    gamma = 1
    n_fold = 3
    n_repeats = 10
    
    clf = svm.SVC(kernel='linear', C=C, gamma=gamma,probability=True)
    scores =  np.full((n_repeats, n_fold), np.nan)
    cm_all =  np.full((8,8,n_repeats, n_fold), np.nan)
    for r in range(n_repeats):
        if shuffle == 1: 
            np.random.shuffle(ytrain)
            
        rkf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=None)  # for within rank
        # rkf = KFold(n_splits=n_fold, shuffle=True, random_state=None)  # for cross rank
        for i, (train, test) in enumerate(rkf.split(xtrian[:,:].T, ytrain[:,0])):
            y_train = ytrain[train, 0]
            
            x_train = xtrian[:,train].T
            clf.fit(x_train, y_train)
                
            x_test = xtrian[:,test].T
            y_test = ytrain[test, 0]
                    
            scores[r,i] = clf.score(x_test, y_test)
            
            # 得到预测概率（用于 AUC）
            # y_score = clf.predict_proba(x_test)
            # auc = roc_auc_score(y_test, y_score, multi_class='ovo')
            # scores[r,i] = auc
                
            # predictions = clf.predict(x_test)
            # tmp  = confusion_matrix(y_test, predictions, labels=clf.classes_)
            # aa = np.sum(tmp,1)
            # aa = np.tile(aa, (8, 1)).T
            # tmp = tmp/aa
            # cm_all[:,:,r,i] = tmp
            
    scores_mean = scores.mean(0).mean(0)
    scores_mean_perm.append(scores_mean)
    cm_mean_perm.append(cm_all.mean(-1).mean(-1))

scores_mean_perm = np.array(scores_mean_perm)
print(np.nanmean(scores_mean_perm))
cm = np.stack(cm_mean_perm)
cm_mean = cm.mean(0)


if shuffle == 0:
    pkl.dump({'cm':cm, 'scores_mean_perm':scores_mean_perm}, 
             open(r'F:\Human_chunking_Huashan_intraoperative\Result\decoding_rank\\crossrank2'+task[taskid]+'(perm100)_t2_5_newC2.pkl','wb'))
if shuffle == 1:
    pkl.dump({'cm':cm, 'scores_mean_perm':scores_mean_perm}, 
             open(r'F:\Human_chunking_Huashan_intraoperative\Result\decoding_rank\\crossrank2'+task[taskid]+'_sf(perm1000_10)_t2_5_newC2.pkl','wb'))    
#%% plot 8*8 confusion matrix
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

result = [None]*10
file = r'F:\Human_chunking_Huashan_intraoperative\Result\decoding_rank'+'\\crossrankgeo_ir(perm100)_t2_5_new.pkl'
result[0] = pkl.load(open(file, 'rb'))
file = r'F:\Human_chunking_Huashan_intraoperative\Result\decoding_rank'+'\\crossrankgeo_ir_sf(perm1000_10)_t2_5_new.pkl'
result[1] = pkl.load(open(file, 'rb'))
file = r'F:\Human_chunking_Huashan_intraoperative\Result\decoding_rank'+'\\crossrank2geo_r4(perm100)_t2_5_newC1.pkl'
result[2] = pkl.load(open(file, 'rb'))
file = r'F:\Human_chunking_Huashan_intraoperative\Result\decoding_rank'+'\\crossrank2geo_r4_sf(perm1000_10)_t2_5_newC1.pkl'
result[3] = pkl.load(open(file, 'rb'))
file = r'F:\Human_chunking_Huashan_intraoperative\Result\decoding_rank'+'\\crossrank2geo_r2(perm100)_t2_5_newC1.pkl'
result[4] = pkl.load(open(file, 'rb'))
file = r'F:\Human_chunking_Huashan_intraoperative\Result\decoding_rank'+'\\crossrank2geo_r2_sf(perm1000_10)_t2_5_newC1.pkl'
result[5] = pkl.load(open(file, 'rb'))

file = r'F:\Human_chunking_Huashan_intraoperative\Result\decoding_rank'+'\\crossrank2geo_r4(perm100)_t2_5_newC2.pkl'
result[6] = pkl.load(open(file, 'rb'))
file = r'F:\Human_chunking_Huashan_intraoperative\Result\decoding_rank'+'\\crossrank2geo_r4_sf(perm1000_10)_t2_5_newC2.pkl'
result[7] = pkl.load(open(file, 'rb'))
file = r'F:\Human_chunking_Huashan_intraoperative\Result\decoding_rank'+'\\crossrank2geo_r2(perm100)_t2_5_newC2.pkl'
result[8] = pkl.load(open(file, 'rb'))
file = r'F:\Human_chunking_Huashan_intraoperative\Result\decoding_rank'+'\\crossrank2geo_r2_sf(perm1000_10)_t2_5_newC2.pkl'
result[9] = pkl.load(open(file, 'rb'))

cm_mean = result[4]['cm'][:50,:,:].mean(0)
with plt.style.context(style_path):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][0:]
    fig, axs = plt.subplots(1, 1, figsize=(1.6, 1.4), dpi=300) 
    h1 = axs.pcolormesh(cm_mean, vmin=.05, vmax=0.3) # , vmin=.15, vmax=0.35 , cmap = plt.cm.RdBu_r
    axs.invert_yaxis()
    axs.set_xticks(np.arange(8)+0.5, np.arange(1,9))
    axs.set_yticks(np.arange(8)+0.5, np.arange(1,9))
    axs.set_xlabel('Prediction')
    axs.set_ylabel('Data')
    fig.colorbar(h1)

with plt.style.context(style_path):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][0:]
    fig, axs = plt.subplots(1, 1, figsize=(1.2, 0.6), dpi=300) 
    
    seltask = 4
    acc = (result[seltask]['scores_mean_perm'][:].mean() + result[seltask+4]['scores_mean_perm'][:].mean())/2
    axs.vlines(acc, 0,200, color=colors[1])
    accsf = (result[seltask+1]['scores_mean_perm'][:]+result[seltask+1+4]['scores_mean_perm'][:])/2
    axs.hist(accsf, color='gray', alpha=0.5)
    # axs.hist(result[3]['scores_mean_perm'][:50], color=colors[0], alpha=0.5)
    
    axs.set_xlim([0.1, 0.35])
    axs.set_xlabel('Accuracy')
    axs.set_ylabel('Counts')
    # axs.set_title('L0')
    # axs.set_title('L1 & L2')
sum(accsf>acc)/1000

with plt.style.context(style_path):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][0:]
    fig, axs = plt.subplots(1, 1, figsize=(1.2, 0.6), dpi=300) 
    
    seltask = 4
    axs.vlines(result[seltask]['scores_mean_perm'][:].mean(), 0,10, color=colors[1])
    axs.hist(result[seltask+1]['scores_mean_perm'][:], color='gray', alpha=0.5)
    # axs.hist(result[3]['scores_mean_perm'][:50], color=colors[0], alpha=0.5)
    
    axs.set_xlim([0.1, 0.35])
    axs.set_xlabel('Accuracy')
    axs.set_ylabel('Counts')
    
np.sum(result[seltask+1]['scores_mean_perm']>result[seltask]['scores_mean_perm'].mean())/1000

#####
cm_ir = result[0]['cm']
cm_r4 = result[2]['cm']
cm_r2 = result[4]['cm']
# cm_r = cm

cm_ir[:,np.arange(8), np.arange(8)] = np.nan
cm_ir_within = (np.nanmean(cm_ir[:,:4,:4], -1).mean(-1) + np.nanmean(cm_ir[:,4:,4:], -1).mean(-1))/2
cm_ir_between = (cm_ir[:,:4,4:].mean(-1).mean(-1) + cm_ir[:,4:,:4].mean(-1).mean(-1))/2

cm_r4[:,np.arange(8), np.arange(8)] = np.nan
cm_r4_within = (np.nanmean(cm_r4[:,:4,:4], -1).mean(-1) + np.nanmean(cm_r4[:,4:,4:], -1).mean(-1))/2
cm_r4_between = (cm_r4[:,:4,4:].mean(-1).mean(-1) + cm_r4[:,4:,:4].mean(-1).mean(-1))/2

cm_r2[:,np.arange(8), np.arange(8)] = np.nan
cm_r2_within = (np.nanmean(cm_r2[:,:4,:4], -1).mean(-1) + np.nanmean(cm_r2[:,4:,4:], -1).mean(-1))/2
cm_r2_between = (cm_r2[:,:4,4:].mean(-1).mean(-1) + cm_r2[:,4:,:4].mean(-1).mean(-1))/2


df = pd.DataFrame({'cm_ir_within':cm_ir_within,'cm_ir_between':cm_ir_between,
                   'cm_r4_within':cm_r4_within, 'cm_r4_between':cm_r4_between,
                   'cm_r2_within':cm_r2_within, 'cm_r2_between':cm_r2_between,})

# df['wb_ir'] = df['cm_ir_between']/df['cm_ir_within']
# df['wb_r4'] = df['cm_r4_between']/df['cm_r4_within']
# df['wb_r2'] = df['cm_r2_between']/df['cm_r2_within']

df['wb_ir'] = df['cm_ir_between']/(df['cm_ir_within']+df['cm_ir_between'])
df['wb_r4'] = df['cm_r4_between']/(df['cm_r4_within']+df['cm_r4_between'])
df['wb_r2'] = df['cm_r2_between']/(df['cm_r2_within']+df['cm_r2_between'])

with plt.style.context(style_path):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][0:]
    fig, axs = plt.subplots(1, 1, figsize=(1, 1.2), dpi=300) 
    
    color1 = ['gray', colors[0], colors[1]]
    df_melted = df[['wb_ir', 'wb_r4','wb_r2']].melt(var_name='Variable', value_name='Value')
    sns.violinplot(x='Variable', y='Value', ax=axs, data=df_melted, color = 'gray',width=0.5)
    for patch, color in zip(axs.collections[::1], color1):  # 每个 violin 有 2 个 PolyCollection（轮廓和填充）
        patch.set_facecolor(color)
    axs.set_ylim([0, 0.8])
    axs.set_xticklabels(['L0', 'L1', 'L2'])    
    axs.set_xlabel(None)    
    axs.set_ylabel('Norm. cross error')    

df.mean()    
t,p = stats.ttest_ind(df['wb_ir'][:50], df['wb_r2'][:50])
print(p)
# p*3
# stats.ttest_1samp(df['wb_ir'][:50], df['wb_r2'].mean())

# cm_ir_between.mean()
# np.sum(df['wb_ir']>df['wb_r'].mean())

#%%
#### 前半段2+2
# cm_r4 = cm_r4[:50,:4,:4]
# cm_r2 = cm_r2[:50,:4,:4]
#### 后半段2+2
cm_r4 = cm_r4[:,4:,4:]
cm_r2 = cm_r2[:,4:,4:]

bb = 2
cm_r4_within = (np.nanmean(cm_r4[:,:bb,:bb], -1).mean(-1) + np.nanmean(cm_r4[:,bb:,bb:], -1).mean(-1))/2
cm_r4_between = (cm_r4[:,:bb,bb:].mean(-1).mean(-1) + cm_r4[:,bb:,:bb].mean(-1).mean(-1))/2

cm_r2_within = (np.nanmean(cm_r2[:,:bb,:bb], -1).mean(-1) + np.nanmean(cm_r2[:,bb:,bb:], -1).mean(-1))/2
cm_r2_between = (cm_r2[:,:bb,bb:].mean(-1).mean(-1) + cm_r2[:,bb:,:bb].mean(-1).mean(-1))/2

df = pd.DataFrame({'cm_r4_within':cm_r4_within, 'cm_r4_between':cm_r4_between,
                   'cm_r2_within':cm_r2_within, 'cm_r2_between':cm_r2_between,})

df['wb_r4'] = df['cm_r4_between']/(df['cm_r4_within']+df['cm_r4_between'])
df['wb_r2'] = df['cm_r2_between']/(df['cm_r2_within']+df['cm_r2_between'])

with plt.style.context(style_path):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axs = plt.subplots(1, 1, figsize=(1, 1.2), dpi=300) 
    
    color1 = [colors[0], colors[1]]
    # df_melted = df[['cm_ir_within', 'cm_r_within']].melt(var_name='Variable', value_name='Value')
    df_melted = df[['wb_r4','wb_r2']].melt(var_name='Variable', value_name='Value')
    sns.violinplot(x='Variable', y='Value', ax=axs, data=df_melted,width=0.5)
    for patch, color in zip(axs.collections[::1], color1):  # 每个 violin 有 2 个 PolyCollection（轮廓和填充）
        patch.set_facecolor(color)
    
    axs.set_ylim([-0.1, 1])
    axs.set_xlim([-0.8, 1.8])
    axs.set_xticklabels(['L1', 'L2'])    
    axs.set_xlabel(None)    
    axs.set_ylabel('Norm. cross error')    
    
stats.ttest_ind(df['wb_r4'], df['wb_r2'])
#%%
scores_mean = np.vstack(scores_mean_allsub)
scores_mean = scores_mean.mean(0)
scores_mean_sm = gaussian_filter1d(scores_mean, 1)

with plt.style.context(style_path):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][0:]
    fig, axs = plt.subplots(1, 1, figsize=(3, 2), dpi=300) 
    axs.plot(scores_mean_sm)
    axs.vlines(np.array([10, 50, 90, 130]) - 4, 0.5, 0.9, linestyle='--',linewidth=1.5, color='gray')  
    
    axs.set_xticks(np.array([10, 50, 90, 130]) - 4)
    axs.set_xticklabels(['0', '2', '4','Resp'])
    
    axs.set_ylabel('Decoding accuracy')
#%% decoding primitives
scores_mean_allsub = []
for s in range(30):
    # print(s)
    fr_allsub_alltask, trialInfo_allsub = mergeSub_noBoot(fr)
    info = trialInfo_allsub.copy()
    
    fr_allsub_rank = np.zeros([fr_allsub_alltask.shape[0], fr_allsub_alltask.shape[1], 16])
    for i in range(16):
        fr_allsub_rank[:,:,i] = fr_allsub_alltask[:,:,mk_ts[0]+i*5:mk_ts[0]+i*5+5].mean(2)
        
    fr_allsub_rank = fr_allsub_rank[:,:,::2]
    fr_allsub_rank = fr_allsub_rank.reshape(fr_allsub_rank.shape[0],-1)
    
    info = pd.DataFrame({'task': np.repeat(info, 8)})
    fff = np.array([3,2,1,3,3,3,3,3])
    info['label'] = np.tile(fff, int(len(info)/8))
    
    idx = info.query('(label==1 or label==2) and task.str.contains("G_Regular4")').index.values
    y = info[['label']].values[idx]
    x = fr_allsub_rank.copy()[:,idx]
    # scaler = MinMaxScaler()
    scaler = StandardScaler()
    x = scaler.fit_transform(x.T).T
    
    idx = info.query('(label==1 or label==2) and task.str.contains("M_Regular4")').index.values
    y2 = info[['label']].values[idx]
    x2 = fr_allsub_rank.copy()[:,idx]
    # scaler = MinMaxScaler()
    scaler = StandardScaler()
    x2 = scaler.fit_transform(x2.T).T

    C = 1
    gamma = 1
    n_fold = 3
    n_repeats = 10
    
    shuffle = 0
    if shuffle==1:
        n_repeats = 10
    # print([n_fold, n_repeats])
    
    clf = svm.SVC(kernel='linear', C=C, gamma=gamma)
    scores =  np.full((n_repeats, n_fold), np.nan)
    cm_all =  np.full((2,2,n_repeats, n_fold), np.nan)
    for r in range(n_repeats):
        # if np.mod(r,10) == 0:
        #     print([s,r])
        if shuffle == 1: 
            np.random.shuffle(y)
            
        rkf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=None)  # for within rank
        # rkf = KFold(n_splits=n_fold, shuffle=True, random_state=None)  # for cross rank
        for i, (train, test) in enumerate(rkf.split(x[:,:].T, y[:,0])):
            # print(train)
            y_train = y[train, 0]
            
            x_train = x[:,train].T
            clf.fit(x_train, y_train)
                
            # x_test = x[:,test].T
            x_test = x2[:,:].T
            # y_test = y[test, 0]
            y_test = y2[:, 0]
                    
            scores[r,i] = clf.score(x_test, y_test)
                
            predictions = clf.predict(x_test)
            tmp  = confusion_matrix(y_test, predictions, labels=clf.classes_)
            # cm_all[:,:,r,i] = tmp
    scores_mean = scores.mean(0).mean(0)
    scores_mean_allsub.append(scores_mean)


scores_mean = np.vstack(scores_mean_allsub)
scores_mean = scores_mean.mean(0)
print(scores_mean)
    
    
