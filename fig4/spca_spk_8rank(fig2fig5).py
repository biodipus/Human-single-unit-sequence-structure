# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 22:59:26 2022

@author: wen
"""

get_ipython().run_line_magic('reset', '-sf')
style_path = r'C:\Wen\OneDrive\CodeHub\Python\style_paper.mplstyle'
# get_ipython().run_line_magic('matplotlib', 'qt5')

import sys 
import numpy as np
import pandas as pd
from scipy import stats
import scipy.io as sio
from sklearn import preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import random
from scipy.sparse.linalg import eigsh as ssl_eigsh
from scipy import linalg
import scipy.io
import pickle as pkl
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import svm
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, RepeatedKFold, KFold,LeaveOneOut
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from scipy.ndimage import gaussian_filter1d
from numpy.lib.stride_tricks import sliding_window_view
import seaborn as sns
# from utils import fun_ct_decoding_target
# from utils import fun_loadData

sys.path.append("..") 
from spca.spca import My_supervised_PCA
#%% functions
def best_fit_plane(points):
    # 构造 A 矩阵和 b 向量
    A = np.c_[points[:, 0], points[:, 1], np.ones(points.shape[0])]
    b = points[:, 2]
    
    # 最小二乘法求解 [a, b, d]，即 z = ax + by + d
    coeff, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    a, b, d = coeff
    return a, b, d

def plot_dynamic_pc(x1, x2, x3=[], title='spca', seqL=3):
    tar_label = [None] * seqL
    tar_label[0] = info['Target_1']
    tar_label[1] = info['Target_2']
    if seqL == 3:
        tar_label[2] = info['Target_3']
    
    x = [x1, x2, x3]
    with plt.style.context(style_path):
        # colors = plt.cm.Set1.colors
        # colors = ['#ca0020', '#f4a582', '#0571b0', '#92c5de']
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][0:]
        fig = plt.figure(figsize=(2.2*seqL, 2), dpi=300)
        axs = fig.subplots(1,seqL)
        k = 0
        for ax in axs:
            for i in range(6):
                idx = tar_label[k] == i+1
                xx = x[k][:, idx, :]  # .mean(0)
                ax.plot(np.arange(n_time), xx[0, :, :].mean(0),  c=colors[i], label=str(i+1))
                
            ax.vlines(mkts_merge[0], -3, 2, color='gray', linestyle='--')
            ax.set_xticks(mkts_merge[0])
            ax.set_ylim([x[k][0, :, :].min(), x[k][0, :, :].max()])
            ax.set_xlabel('Time')
            ax.set_ylabel('Component')
            k += 1
        
        axs[-1].legend(bbox_to_anchor=(1, 1.1))
        fig.suptitle(title) # ,  Ch(' + rid + ')
        # fig.suptitle('Encoding subspace') # Ch(' + rid + ')
        plt.tight_layout(pad=0.2)
        plt.show()
        
def plot_2d_geo(x1, x2, x3=[], title='spca', seqL=3):
    x = [x1, x2, x3]
    
    tar_label = [None] * seqL
    tar_label[0] = info['Target_1']
    tar_label[1] = info['Target_2']
    if seqL == 3:
        tar_label[2] = info['Target_3']
    
    #### plot sessions
    ts = np.array([[15,21],[21,26],[26,31],[37,42],[42,47],[53,58]]) + 1
  
    with plt.style.context(style_path):
        # colors = plt.cm.Set1.colors
        # colors = ['#ca0020', '#f4a582', '#0571b0', '#92c5de']
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][0:]
        fig = plt.figure(figsize=(2.5*seqL, 1.5), dpi=300)
        axs = fig.subplots(seqL,ts.shape[0])
        plt.subplots_adjust(wspace=0.15, hspace=0.2)
        for axis in range(axs.shape[0]):
            for t in range(ts.shape[0]):
                ax = axs[axis,t]
                xx = x[axis][:, :,  ts[t,0]:ts[t,1]].mean(2)
                x_mean = np.zeros([6,2])
                for i in range(6):
                    idx = tar_label[axis] == i+1
                    x_mean[i,:] = [xx[0, idx].mean(), xx[1, idx].mean()]
                    ax.scatter(x_mean[i,0], x_mean[i,1], c=colors[i], label=str(i+1))
                
                x_mean = np.concatenate([x_mean, x_mean[0:1,:]], axis=0)
                ax.plot(x_mean[:,0], x_mean[:,1], c='gray')
                # ax.set_xlim([x_mean[:,0].min()-0.1, x_mean[:,0].max()+0.1])    
                # ax.set_ylim([x_mean[:,1].min()-0.1, x_mean[:,1].max()+0.1])    
                ax.set_xlim([-2.5, 2])    
                ax.set_ylim([-1.8, 4.5]) 
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                
                # ax.spines['bottom'].set_position(('axes', -0.1))
                # ax.spines['left'].set_position(('axes', -0.1))
                # ax.set_xlabel('PC1')
                # ax.set_ylabel('PC2')
        
        axs[0,-1].legend([str(i+1) for i in range(6)], bbox_to_anchor=(1, 1.1))
        # fig.suptitle(title) # ,  Ch(' + rid + ')
        # fig.suptitle('Encoding subspace') # Ch(' + rid + ')
        # plt.tight_layout() # pad=0.2
        plt.show()
        
def remove_outlier(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
    return filtered_data
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
neuNum = sum([len(i) for i in neu_info])

neu_info_all = pd.concat(neu_info)
neu_info_all['neuID_in_155'] = np.arange(neuNum)+1

aline_ts = int(window/step/2)
mk_ts = mk_ts #-aline_ts
mk_name = ['0', '2', '4','Resp']

#### 选择正确/错误trial
sel_trialtype = 1
if sel_trialtype ==1:
    trialInfo_correct = [None] * len(subs)
    fr_correct = [None] * len(subs)
    for sub in range(len(subs)):
        x = fr[sub].copy()
        info = trialInfo[sub].copy()
        
        idx = (info['acc'] == 1) | (info['acc'] == 3)
        # idx = (info['acc'] > 1)
        trialInfo_correct[sub] = info.loc[idx,:]
        fr_correct[sub] = x[:,idx,:]
        
    fr = fr_correct
    trialInfo = trialInfo_correct
    
#### bin 成8个点
fr_bin = [None]*len(fr)
for s in range(len(fr)):
    x = fr[s]
    tmp = np.zeros(x.shape)[:,:,:8]
    for i in range(8):
        win = [mk_ts[0]+i*10+5, mk_ts[0]+i*10+10]
        # print(win)
        tmp[:,:,i] = x[:,:,win[0]:win[1]].mean(2)
    fr_bin[s] = tmp

#### 滑窗取全部时间
fr_at = [None]*len(fr)
for s in range(len(fr)):
    tmp = fr[s][:,:,mk_ts[0]:mk_ts[2]+3] # 10:94
    tt = sliding_window_view(tmp, window_shape=5, axis=2)
    fr_at[s] = tt[:,:,::2].mean(axis=-1)
    
    # tmp = fr[s][:,:,:] 
    # for i in range(tmp.shape[0]):
    #     for j in range(tmp.shape[1]):
    #         tmp[i,j,:] = gaussian_filter1d(tmp[i,j,:], 2)
    # fr_at[s] = tmp[:,:,10:90]

ts_per_rank = fr_at[0].shape[2]/8
#%% select neuron
path = r'F:\Human_chunking_Huashan_intraoperative\Result\bd_neuron_PCA\fig2'
file = path + '\\PCA_155drift_AllTrials_TopPCgen_AL5_AllPerm_RespTrim_PCMax6.mat'
# file = path + '\\PCA_155drift_AllTrials_Stacked_TopPCgen_AL5_AllPerm_RespTrim.mat'

pcadata = sio.loadmat(file)
qt = 1
pc = 1
pctask = 1
plottask1 = pctask
ch_sel = pcadata['chan_idx_set'][0,qt][pctask,pc][:,0]-1

neu_info_all = neu_info_all.reset_index(drop=True)
neu_info_all['bd_neuron'] = 0
neu_info_all.loc[ch_sel, 'bd_neuron'] = 1
neuNum = sum(neu_info_all['bd_neuron']==0)

def remove_bd_neuron(s):
    selsub = subs[s]
    tmp = neu_info_all.query('subID==@selsub').reset_index(drop=True)
    neuidx = tmp['bd_neuron'] == 0
    return neuidx
#%% merge data, boot, spca, 3种GLM, 采用
def pc_label_explained_variance(Z, Y):
    """
    Z: (n_samples, n_components) PC scores
    Y: (n_samples,) categorical label

    return:
        var_explained: (n_components,)
    """
    n_pc = Z.shape[1]
    var_exp = np.zeros(n_pc)

    for k in range(n_pc):
        z = Z[:, k]
        mu_global = z.mean()

        for c in np.unique(Y):
            idx = (Y == c)
            pc = idx.mean()
            mu_c = z[idx[:,0]].mean()
            var_exp[k] += (pc * (mu_c - mu_global) ** 2)

    return var_exp

def pc_label_explained_std(Z, Y):
    """
    Z: (n_samples, n_components) PC scores
    Y: (n_samples,) categorical label

    return:
        var_explained: (n_components,)
    """
    n_pc = Z.shape[1]
    var_exp = np.zeros(n_pc)

    for k in range(n_pc):
        z = Z[:, k]
        mu_global = z.std()

        for c in np.unique(Y):
            idx = (Y == c)
            # pc = idx.mean()
            mu_c = z[idx[:,0]].std()
            var_exp[k] += mu_c/mu_global

    return var_exp

def spca_perm(dotask, projecttask):
    rank3d_value = np.zeros([repetition,timebin, 3])
    fea_imp = np.zeros([neuNum, repetition])
    exp_var = np.zeros([repetition,neuNum])
    trial_perm = [None] * repetition
    trial_reduce_perm = [None] * repetition
    for repeat in range(repetition):
        task = [dotask, projecttask]
        psudoTrial_num = 20
        
        fr_allsub1 = [None]*len(subs)
        fr_allsub2 = [None]*len(subs)
        for s in range(len(subs)):
            # neuidx = remove_bd_neuron(s)
            fr_at_sub = fr_at[s][:,:,:]
            idx = trialInfo[s]['task'] == dotask
            tmp1 = fr_at_sub[:,idx,:].copy()
            tmp2 = np.zeros([fr_at_sub.shape[0], psudoTrial_num, fr_at_sub.shape[2]])
            tmp3 = fr_bin[s][:,idx,:].copy()
            tmp4 = np.zeros([fr_bin[s].shape[0], psudoTrial_num, fr_bin[s].shape[2]])
            if tmp1.shape[1] > 0:
                for r in range(psudoTrial_num):
                    perm_idx = np.random.default_rng().choice(tmp1.shape[1], size=2, replace=True)
                    tmp2[:,r,:] = tmp1[:,perm_idx,:].mean(1)
                    tmp4[:,r,:] = tmp3[:,perm_idx,:].mean(1)
            fr_allsub1[s] = tmp2
            fr_allsub2[s] = tmp4
        fr_allsub_at = np.concatenate(fr_allsub1, axis=0)
        fr_allsub_bin = np.concatenate(fr_allsub2, axis=0)
        
        #### Norm
        # print(dotask)
        # sel_task = trialInfo_allsub.query('task.str.contains(@projecttask)').index.values
        # trialInfo_seltask = trialInfo_allsub.iloc[sel_task,:].reset_index()
        fr_allsub_seltask = fr_allsub_at
        
        scaler = StandardScaler()
        x = fr_allsub_seltask[:,:,:].copy()
        shape_3d = x.shape
        xx = x.reshape(shape_3d[0], -1)
        x_norm = scaler.fit_transform(xx.T).T
        # x_norm_3d = x_norm.reshape(shape_3d)
        
        #### Norm project task
        # sel_task = trialInfo_allsub.query('task.str.contains(@dotask)').index.values
        # trialInfo_seltask = trialInfo_allsub.iloc[sel_task,:].reset_index()
        fr_allsub_seltask = fr_allsub_at
        
        scaler = StandardScaler()
        x = fr_allsub_seltask[:,:,:].copy()
        shape_3d = x.shape
        xx = x.reshape(shape_3d[0], -1)
        xtest_norm = scaler.fit_transform(xx.T).T

        #### spca model fit
        y1 = np.tile(np.repeat([1,2,3,4,5,6,7,8], ts_per_rank), int(x.shape[1]))
        # y1 = np.tile(np.repeat([1,1,1,1,2,2,2,2], ts_per_rank), int(x.shape[1]))
        # spca of T1
        model = My_supervised_PCA()
        model.fit(x_norm, y1.T)
        eig_val = model.V
        eig_vec = model.U
        x_reduce = model.transform(xtest_norm, y1.T)
        
        # feature projection weight
        fea_imp[:,repeat] = np.linalg.norm(eig_vec[:, :3], axis=1)
        
        # reshape back to f*time*sample
        x_reduce3d = x_reduce.reshape(x_reduce.shape[0], shape_3d[1], shape_3d[2])
        # exp_var[repeat,:] = [eig_val[i]/eig_val.sum() for i in range(0,10)]
        
        for jj in range(neuNum):
            pc1_scores = xtest_norm.T @ eig_vec[:, jj]  # 投影到 PC1
            exp_var[repeat,jj] = np.var(pc1_scores, ddof=1)
            
        # exp_var[repeat,:] = pc_label_explained_std(x_reduce.T, y1.reshape(-1,1))
        exp_var[repeat,:] = exp_var[repeat,:]/exp_var[repeat,:].sum()
        
        y = x_reduce3d
        y_mean = y[:, :, :].mean(1)[:3,:].T
        
        # if repeat==0:
        #     eig_vec_fix = eig_vec
        # else:
        #     for pc in range(3):
        #         print(np.dot(eig_vec[pc], eig_vec_fix[pc]))
        #         if np.dot(eig_vec[pc], eig_vec_fix[pc]) < 0:
        #             y_mean[:,pc] = -y_mean[:,pc]
                
        rank3d_value[repeat,:,:] = y_mean
        trial_perm[repeat] = fr_allsub_seltask
        trial_reduce_perm[repeat] = x_reduce3d
    return rank3d_value, fea_imp, exp_var,trial_perm,trial_reduce_perm

timebin = fr_at[0].shape[2]
repetition = 50

# dotask, projecttask = 'geo_ir','geo_ir'
rank3d_value_gir, fea_imp_ir, exp_var_ir,fr_allsub_ir,fr_reduce_ir = spca_perm('geo_ir','geo_ir')
rank3d_value_gr4, fea_imp_r4, exp_var_r4,fr_allsub_r4,fr_reduce_r4 = spca_perm('geo_r4','geo_r4')
rank3d_value_gr2, fea_imp_r2, exp_var_r2,fr_allsub_r2,fr_reduce_r2 = spca_perm('geo_r2','geo_r2')

# rank3d_value_gir, fea_imp_ir, exp_var_ir,fr_allsub_ir,fr_reduce_ir = spca_perm('lan_ir','lan_ir')
# rank3d_value_gr4, fea_imp_r4, exp_var_r4,fr_allsub_r4,fr_reduce_r4 = spca_perm('lan_r','lan_r')
# rank3d_value_gr2, fea_imp_r2, exp_var_r2,fr_allsub_r2,fr_reduce_r2 = spca_perm('lan_r','lan_r')

# rank3d_value_gir, fea_imp_ir, exp_var_ir,fr_allsub_ir,fr_reduce_ir  = spca_perm('mus_ir','mus_ir')
# rank3d_value_gr4, fea_imp_r4, exp_var_r4,fr_allsub_r4,fr_reduce_r4 = spca_perm('mus_r4','mus_r4')
# rank3d_value_gr2, fea_imp_r2, exp_var_r2,fr_allsub_r2,fr_reduce_r2 = spca_perm('mus_r2','mus_r2')

exp_var = [exp_var_ir, exp_var_r4, exp_var_r2]
exp_var_ir[:,:3].mean(0)
exp_var_r4[:,:3].mean(0)
exp_var_r2[:,:3].mean(0)

pkl.dump({'fr_allsub_ir':fr_allsub_ir, 'fr_allsub_r4':fr_allsub_r4,'fr_allsub_r2':fr_allsub_r2}, 
         open(r'F:\Human_chunking_Huashan_intraoperative\Result\spca_3Hyp_ExpVar\geo_perm_correct(1_3).pkl','wb'))
#%% merge data, boot, PCA
def perm(dotask, projecttask):
    fr_allsub_at = [None] * repetition
    fr_allsub_bin = [None] * repetition
    for repeat in range(repetition):
        task = [dotask, projecttask]
        psudoTrial_num = 20
        
        fr_allsub1 = [None]*len(subs)
        fr_allsub2 = [None]*len(subs)
        for s in range(len(subs)):
            idx = trialInfo[s]['task'] == dotask
            tmp1 = fr_at[s][:,idx,:].copy()
            tmp2 = np.zeros([fr_at[s].shape[0], psudoTrial_num, fr_at[s].shape[2]])
            tmp3 = fr_bin[s][:,idx,:].copy()
            tmp4 = np.zeros([fr_bin[s].shape[0], psudoTrial_num, fr_bin[s].shape[2]])
            if tmp1.shape[1] > 0:
                for r in range(psudoTrial_num):
                    perm_idx = np.random.default_rng().choice(tmp1.shape[1], size=2, replace=True)
                    tmp2[:,r,:] = tmp1[:,perm_idx,:].mean(1)
                    tmp4[:,r,:] = tmp3[:,perm_idx,:].mean(1)
            fr_allsub1[s] = tmp2
            fr_allsub2[s] = tmp4
        fr_allsub_at[repeat] = np.concatenate(fr_allsub1, axis=0)
        fr_allsub_bin[repeat] = np.concatenate(fr_allsub2, axis=0)
    
    return fr_allsub_at, fr_allsub_at

def spca_perm(x_fit, x_test):
    exp_var = np.zeros([repetition,3])
    rank3d_value = np.zeros([repetition,timebin, 3])
    fea_imp = np.zeros([neuNum, repetition])
    for repeat in range(repetition):
        #### Norm
        # print(dotask)
        # sel_task = trialInfo_allsub.query('task.str.contains(@projecttask)').index.values
        # trialInfo_seltask = trialInfo_allsub.iloc[sel_task,:].reset_index()
        fr_allsub_seltask = x_fit[repeat]
        
        scaler = StandardScaler()
        x = fr_allsub_seltask[:,:,:].copy()
        shape_3d = x.shape
        xx = x.reshape(shape_3d[0], -1)
        x_norm = scaler.fit_transform(xx.T)
        # x_norm_3d = x_norm.reshape(shape_3d)
        
        #### Norm project task
        # sel_task = trialInfo_allsub.query('task.str.contains(@dotask)').index.values
        # trialInfo_seltask = trialInfo_allsub.iloc[sel_task,:].reset_index()
        fr_allsub_seltask = x_test[repeat]
        
        scaler = StandardScaler()
        x = fr_allsub_seltask[:,:,:].copy()
        shape_3d = x.shape
        xx = x.reshape(shape_3d[0], -1)
        xtest_norm = scaler.fit_transform(xx.T)
        
        #### PCA model fit
        pca = PCA(n_components=3)
        pca.fit(x_norm)
        x_reduce = pca.transform(x_norm)
        x_reduce = x_reduce.T
        x_reduce3d = x_reduce.reshape(3, shape_3d[1], shape_3d[2])
        exp_var[repeat,:] = pca.explained_variance_ratio_
        
        y = x_reduce3d
        
        y_mean = y[:, :, :].mean(1).T
        
        rank3d_value[repeat,:,:] = y_mean
    return rank3d_value, fea_imp, exp_var

repetition = 20
# dotask, projecttask = 'geo_ir','geo_ir'

x_fit, x_test = perm('geo_ir','geo_ir')
timebin = x_test[0].shape[2]
rank3d_value_gir, fea_imp_ir, exp_var_ir = spca_perm(x_fit, x_test)
x_fit, x_test = perm('geo_r4','geo_r4')
rank3d_value_gr4, fea_imp_r4, exp_var_r4 = spca_perm(x_fit, x_test)
x_fit, x_test = perm('geo_r2','geo_r2')
rank3d_value_gr2, fea_imp_r2, exp_var_r2 = spca_perm(x_fit, x_test)

# x_fit, x_test = perm('lan_ir','lan_ir')
# timebin = x_test[0].shape[2]
# rank3d_value_gir, fea_imp_ir, exp_var_ir = spca_perm(x_fit, x_test)
# x_fit, x_test = perm('lan_r','lan_r')
# rank3d_value_gr4, fea_imp_r4, exp_var_r4 = spca_perm(x_fit, x_test)
# x_fit, x_test = perm('lan_r','lan_r')
# rank3d_value_gr2, fea_imp_r2, exp_var_r2 = spca_perm(x_fit, x_test)

# x_fit, x_test = perm('lan_ir','lan_ir')
# timebin = x_test[0].shape[2]
# rank3d_value_gir, fea_imp_ir, exp_var_ir = spca_perm(x_fit, x_test)
# x_fit, x_test = perm('lan_r','lan_r')
# rank3d_value_gr4, fea_imp_r4, exp_var_r4 = spca_perm(x_fit, x_test)
# x_fit, x_test = perm('lan_r','lan_r')
# rank3d_value_gr2, fea_imp_r2, exp_var_r2 = spca_perm(x_fit, x_test)

# x_fit, x_test = perm('geo_ir','geo_ir')
# timebin = x_test[0].shape[2]
# rank3d_value_gir, fea_imp_ir, exp_var_ir = spca_perm(x_fit, x_test)
# x_fit, x_test = perm('geo_r4','geo_r4')
# rank3d_value_gr4, fea_imp_r4, exp_var_r4 = spca_perm(x_fit, x_test)
# x_fit, x_test = perm('geo_r2','geo_r2')
# rank3d_value_gr2, fea_imp_r2, exp_var_r2 = spca_perm(x_fit, x_test)
#%% merge data, boot, spca
def spca_perm(dotask, projecttask):
    rank3d_value = np.zeros([repetition,timebin, 3])
    fea_imp = np.zeros([neuNum, repetition])
    exp_var = np.zeros([repetition,10])
    for repeat in range(repetition):
        task = [dotask, projecttask]
        psudoTrial_num = 20
        
        task_allsub = []
        fr_allsub_alltask = []
        for i in range(len(task)):
            task_allsub.append(task[i])
            fr_allsub = [None]*len(subs)
            for s in range(len(subs)):
                idx = trialInfo[s]['task'] == task[i]
                tmp = fr[s][:,idx,:].copy()
                tmp2 = np.zeros([fr[s].shape[0], psudoTrial_num, fr[s].shape[2]])
                if tmp.shape[1]>0:
                    for r in range(psudoTrial_num):
                        tmp2[:,r,:] = tmp[:,np.random.default_rng().choice(tmp.shape[1], size=3, replace=True),:].mean(1)
                fr_allsub[s] = tmp2
            fr_allsub = np.concatenate(fr_allsub, axis=0)
            fr_allsub_alltask.append(fr_allsub)
        
        fr_allsub_alltask = np.concatenate(fr_allsub_alltask, axis=1)
        trialInfo_allsub = pd.DataFrame({'task':np.repeat(task_allsub,psudoTrial_num)})
        
        fr_allsub_alltask = fr_allsub_alltask[:,:,:]
        chNum, trialNum, timeNum = fr_allsub_alltask.shape
        
        #### Norm
        # print(dotask)
        sel_task = trialInfo_allsub.query('task.str.contains(@projecttask)').index.values
        trialInfo_seltask = trialInfo_allsub.iloc[sel_task,:].reset_index()
        fr_allsub_seltask = fr_allsub_alltask[:,sel_task,:]
        
        scaler = StandardScaler()
        x = fr_allsub_seltask[:,:,:].copy()
        shape_3d = x.shape
        xx = x.reshape(shape_3d[0], -1)
        x_norm = scaler.fit_transform(xx.T).T
        # x_norm_3d = x_norm.reshape(shape_3d)
        
        #### Norm project task
        sel_task = trialInfo_allsub.query('task.str.contains(@dotask)').index.values
        trialInfo_seltask = trialInfo_allsub.iloc[sel_task,:].reset_index()
        fr_allsub_seltask = fr_allsub_alltask[:,sel_task,:]
        
        scaler = StandardScaler()
        x = fr_allsub_seltask[:,:,:].copy()
        shape_3d = x.shape
        xx = x.reshape(shape_3d[0], -1)
        xtest_norm = scaler.fit_transform(xx.T).T

        #### spca model fit
        y1 = np.tile(np.repeat([1,2,3,4,5,6,7,8], ts_per_rank), int(x.shape[1]))
        # spca of T1
        model = My_supervised_PCA()
        model.fit(x_norm, y1.T)
        eig_val = model.V
        eig_vec = model.U
        x_reduce = model.transform(xtest_norm, y1.T)
        
        # feature projection weight
        fea_imp[:,repeat] = np.linalg.norm(eig_vec[:, :3], axis=1)
        
        # reshape back to f*time*sample
        x_reduce3d = x_reduce.reshape(x_reduce.shape[0], shape_3d[1], shape_3d[2])
        exp_var[repeat,:] = [eig_val[i]/eig_val.sum() for i in range(0,10)]
        
        #### 计算每个点到平面的距离
        y = x_reduce3d
        y_mean = y[:, :, :].mean(1)
    
        rank3d_value[repeat,:,:] = y_mean[:3,:].T
    return rank3d_value, fea_imp, exp_var

timebin = fr[0].shape[2]
repetition = 100

# rank3d_value_gir, fea_imp_ir, _ = spca_perm('geo_ir','geo_ir')
# rank3d_value_gr4, fea_imp_r4, _ = spca_perm('geo_r4','geo_r4')
# rank3d_value_gr2, fea_imp_r2, _ = spca_perm('geo_r2','geo_r2')

rank3d_value_gir, fea_imp_ir, _ = spca_perm('lan_ir','lan_ir')
rank3d_value_gr4, fea_imp_r4, _ = spca_perm('lan_r','lan_r')
rank3d_value_gr2, fea_imp_r2, = rank3d_value_gr4, fea_imp_r4,

# rank3d_value_gir, fea_imp_ir, _ = spca_perm('mus_ir','mus_ir')
# rank3d_value_gr4, fea_imp_r4, _ = spca_perm('mus_r2','mus_r2')
# rank3d_value_gr2, fea_imp_r2, = rank3d_value_gr4, fea_imp_r4,

# savefile = r'F:\Human_chunking_Huashan_intraoperative\Result\spca\geo_correct.pkl'
# pkl.dump({'rank3d_value_gir':rank3d_value_gir, 'rank3d_value_gr4':rank3d_value_gr4,'rank3d_value_gr2':rank3d_value_gr2}, 
#          open(savefile,'wb'))
#%% merge data, boot, spca, 兼容跨任务
def spca_perm(dotask, projecttask):
    rank3d_value = np.zeros([repetition,timebin, 3])
    fea_imp = np.zeros([neuNum, repetition])
    exp_var = np.zeros([repetition,10])
    for repeat in range(repetition):
        task = [dotask, projecttask]
        psudoTrial_num = 20
        
        task_allsub = []
        fr_allsub_alltask = []
        for i in range(len(task)):
            task_allsub.append(task[i])
            fr_allsub = [None]*len(subs)
            for s in range(len(subs)):
                idx = trialInfo[s]['task'] == task[i]
                tmp = fr[s][:,idx,:].copy()
                tmp2 = np.zeros([fr[s].shape[0], psudoTrial_num, fr[s].shape[2]])
                if tmp.shape[1] > 0:
                    for r in range(psudoTrial_num):
                        tmp2[:,r,:] = tmp[:,np.random.default_rng().choice(tmp.shape[1], size=3, replace=True),:].mean(1)
                fr_allsub[s] = tmp2
            fr_allsub = np.concatenate(fr_allsub, axis=0)
            fr_allsub_alltask.append(fr_allsub)
        
        # fr_allsub_alltask = np.concatenate(fr_allsub_alltask, axis=1)
        # trialInfo_allsub = pd.DataFrame({'task':np.repeat(task_allsub,psudoTrial_num)})
        
        # fr_allsub_alltask = fr_allsub_alltask[:,:,:]
        # chNum, trialNum, timeNum = fr_allsub_alltask.shape
        
        #### Norm
        # print(dotask)
        # sel_task = trialInfo_allsub.query('task.str.contains(@projecttask)').index.values
        # trialInfo_seltask = trialInfo_allsub.iloc[sel_task,:].reset_index()
        fr_allsub_seltask = fr_allsub_alltask[0]
        
        scaler = StandardScaler()
        x = fr_allsub_seltask[:,:,:].copy()
        shape_3d = x.shape
        xx = x.reshape(shape_3d[0], -1)
        x_norm = scaler.fit_transform(xx.T).T
        # x_norm_3d = x_norm.reshape(shape_3d)
        
        #### Norm project task
        # sel_task = trialInfo_allsub.query('task.str.contains(@dotask)').index.values
        # trialInfo_seltask = trialInfo_allsub.iloc[sel_task,:].reset_index()
        fr_allsub_seltask = fr_allsub_alltask[1]
        
        scaler = StandardScaler()
        x = fr_allsub_seltask[:,:,:].copy()
        shape_3d = x.shape
        xx = x.reshape(shape_3d[0], -1)
        xtest_norm = scaler.fit_transform(xx.T).T

        #### spca model fit
        y1 = np.tile(np.repeat([1,2,3,4,5,6,7,8], ts_per_rank), int(x.shape[1]))
        # spca of T1
        model = My_supervised_PCA()
        model.fit(x_norm, y1.T)
        eig_val = model.V
        eig_vec = model.U
        x_reduce = model.transform(xtest_norm, y1.T)
        
        # feature projection weight
        fea_imp[:,repeat] = np.linalg.norm(eig_vec[:, :3], axis=1)
        
        # reshape back to f*time*sample
        x_reduce3d = x_reduce.reshape(x_reduce.shape[0], shape_3d[1], shape_3d[2])
        exp_var[repeat,:] = [eig_val[i]/eig_val.sum() for i in range(0,10)]
        
        #### 计算每个点到平面的距离
        y = x_reduce3d
        y_mean = y[:, :, :].mean(1)[:3,:].T
        
        # if repeat==0:
        #     eig_vec_fix = eig_vec
        # else:
        #     for pc in range(3):
        #         print(np.dot(eig_vec[pc], eig_vec_fix[pc]))
        #         if np.dot(eig_vec[pc], eig_vec_fix[pc]) < 0:
        #             y_mean[:,pc] = -y_mean[:,pc]
                
        rank3d_value[repeat,:,:] = y_mean
    return rank3d_value, fea_imp, exp_var

timebin = fr[0].shape[2]
repetition = 10

rank3d_value_gir, fea_imp_ir, _ = spca_perm('geo_ir','geo_ir')
rank3d_value_gr4, fea_imp_r4, _ = spca_perm('geo_r4','geo_r4')
rank3d_value_gr2, fea_imp_r2, _ = spca_perm('geo_r2','geo_r2')

# rank3d_value_gir, fea_imp_ir, _ = spca_perm('lan_ir','lan_ir')
# rank3d_value_gr4, fea_imp_r4, _ = spca_perm('lan_r','lan_r')
# rank3d_value_gr2, fea_imp_r2, = rank3d_value_gr4, fea_imp_r4,

# rank3d_value_gir, fea_imp_ir, _ = spca_perm('mus_ir','mus_ir')
# rank3d_value_gr4, fea_imp_r4, _ = spca_perm('mus_r4','mus_r4')
# rank3d_value_gr2, fea_imp_r2, _ = spca_perm('mus_r2','mus_r2')

# savefile = r'F:\Human_chunking_Huashan_intraoperative\Result\spca\geo_correct.pkl'
# pkl.dump({'rank3d_value_gir':rank3d_value_gir, 'rank3d_value_gr4':rank3d_value_gr4,'rank3d_value_gr2':rank3d_value_gr2}, 
#          open(savefile,'wb'))
#%% 计算到平面的距离， 所有permutation
def distance_index(xx):
    seg = timebin//2
    dis_perm = np.zeros([repetition, 1])
    for repeat in range(repetition):
        pp = xx[repeat,:,:]
        plane1 = np.array(best_fit_plane(pp[:seg,:]))
        plane2 = np.array(best_fit_plane(pp[seg:,:]))
        plane12 = np.array(best_fit_plane(pp[:,:]))
        
        distances = np.zeros([3,timebin])
        a,b,d = plane1
        numerators = np.abs(a * pp[:, 0] + b * pp[:, 1] - pp[:, 2] + d)
        denominator = np.sqrt(a**2 + b**2 + 1)
        distances[0,:] = numerators / denominator
        
        a,b,d = plane2
        numerators = np.abs(a * pp[:, 0] + b * pp[:, 1] - pp[:, 2] + d)
        denominator = np.sqrt(a**2 + b**2 + 1)
        distances[1,:] = numerators / denominator
        
        a,b,d = plane12
        numerators = np.abs(a * pp[:, 0] + b * pp[:, 1] - pp[:, 2] + d)
        denominator = np.sqrt(a**2 + b**2 + 1)
        distances[2,:] = numerators / denominator
        
        # get_ipython().run_line_magic('matplotlib', 'inline')
        # with plt.style.context(style_path):
        #     colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        #     fig, ax = plt.subplots(1,1, figsize=(2.5, 2), dpi=300) 
        #     ax.plot(np.arange(1,9), distances[0,:], marker='o')
        #     ax.plot(np.arange(1,9), distances[1,:], marker='o')
        #     ax.plot(np.arange(1,9), distances[2,:], marker='o')
        
        dis_perm[repeat, 0] = ((distances[0,seg:].mean()/distances[0,:seg].mean()) + 
              (distances[1,:seg].mean()/distances[1,seg:].mean()))
    return dis_perm
    
dis_ir = distance_index(rank3d_value_gir)
dis_ir = remove_outlier(dis_ir)
print(dis_ir.mean())

dis_r4 = distance_index(rank3d_value_gr4)
dis_r4 = remove_outlier(dis_r4)
print(dis_r4.mean())


plt.hist(dis_ir, alpha=0.5)
plt.hist(dis_r4, alpha=0.5)
# plt.hist(dis_perm_removeOut)
stats.ttest_ind(dis_ir, dis_r4)

#%% 构造整体和分段两个模型，用BIC比较最优模型
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import euclidean
from math import acos, degrees

def fit_plane_least_squares(points):
    X = points[:, :2]  # x, y
    Z = points[:, 2]   # z
    A = np.c_[X, np.ones(X.shape[0])]
    coeff, _, _, _ = np.linalg.lstsq(A, Z, rcond=None)
    a, b, c = coeff
    normal = np.array([a, b, -1.0])
    normal = normal / np.linalg.norm(normal)
    arbitrary = np.array([1, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1, 0])
    v1 = np.cross(normal, arbitrary)
    v1 /= np.linalg.norm(v1)
    v2 = np.cross(normal, v1)
    v2 /= np.linalg.norm(v2)
    axes = np.vstack([v1, v2])
    origin = points.mean(axis=0)
    return origin, axes, normal

def project_to_plane(points, origin, axes):
    centered = points - origin
    return centered @ axes.T

def reconstruction_error(points, origin, axes):
    proj = project_to_plane(points, origin, axes)
    recon = proj @ axes + origin
    return np.mean(np.linalg.norm(points - recon, axis=1) ** 2)

def path_efficiency(points, axes):
    proj = project_to_plane(points, points.mean(axis=0), axes)
    direct = np.linalg.norm(proj[-1] - proj[0])
    path = np.sum(np.linalg.norm(np.diff(proj, axis=0), axis=1))
    return 1 - (direct / path)

def bic(error, n_points, n_params):
    return n_points * np.log(error) + n_params * np.log(n_points)

def angle_between_planes(n1, n2):
    cos_angle = np.clip(np.abs(np.dot(n1, n2)), 0, 1)
    return np.degrees(np.arccos(cos_angle))

def analyze_curve(curve):
    N = len(curve)
    # 模型 1：整体
    o1, ax1, n1 = fit_plane_least_squares(curve)
    err1 = reconstruction_error(curve, o1, ax1)
    pe1 = path_efficiency(curve, ax1)
    bic1 = bic(err1 + pe1 * 1.0, N, 6)

    # 模型 2：前后两段
    mid = N // 2
    part1, part2 = curve[:mid], curve[mid:]

    o2a, ax2a, n2a = fit_plane_least_squares(part1)
    o2b, ax2b, n2b = fit_plane_least_squares(part2)

    err2a = reconstruction_error(part1, o2a, ax2a)
    err2b = reconstruction_error(part2, o2b, ax2b)
    pe2a = path_efficiency(part1, ax2a)
    pe2b = path_efficiency(part2, ax2b)
    total_error = (err2a + err2b) / 2 + (pe2a + pe2b) * 0.5
    bic2 = bic(total_error, N, 12)

    angle = angle_between_planes(n2a, n2b)

    return {
        'bic1': bic1,
        'bic2': bic2,
        'angle': angle,
        'best_model': 'segment' if bic2 < bic1 else 'whole',
        'n1': n2a,
        'n2': n2b,
        'origin1': o2a,
        'origin2': o2b,
        'whole_normal': n1,
        'whole_origin': o1,
        'path_efficiency': pe2a + pe2b
    }

def plot_curve_with_planes(curve, analysis, title=''):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(*curve.T, label='curve', color='k')

    # 网格范围
    xx, yy = np.meshgrid(
        np.linspace(np.min(curve[:, 0]), np.max(curve[:, 0]), 10),
        np.linspace(np.min(curve[:, 1]), np.max(curve[:, 1]), 10)
    )

    # 整体模型平面（灰色）
    n = analysis['whole_normal']
    o = analysis['whole_origin']
    d = -np.dot(n, o)
    zz = (-n[0] * xx - n[1] * yy - d) / n[2]
    ax.plot_surface(xx, yy, zz, alpha=0.3, color='gray')

    # 分段模型两个平面（红色和蓝色）
    for normal, origin, color in zip([analysis['n1'], analysis['n2']],
                                     [analysis['origin1'], analysis['origin2']],
                                     ['r', 'b']):
        d = -np.dot(normal, origin)
        zz = (-normal[0] * xx - normal[1] * yy - d) / normal[2]
        ax.plot_surface(xx, yy, zz, alpha=0.4, color=color)

    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()

bic_geo = np.zeros([3,3,rank3d_value_gir.shape[0]]) # task * model * repeat
for i in range(rank3d_value_gir.shape[0]):
    curve_a = rank3d_value_gir[i,:,:]
    curve_b = rank3d_value_gr4[i,:,:]
    curve_c = rank3d_value_gr2[i,:,:]
    # 分析
    analysis_a = analyze_curve(curve_a)
    analysis_b = analyze_curve(curve_b)
    analysis_c = analyze_curve(curve_c)
    
    bic_geo[0,:,i] = analysis_a['bic1'], analysis_a['bic2'], analysis_a['path_efficiency']
    bic_geo[1,:,i] = analysis_b['bic1'], analysis_b['bic2'], analysis_b['path_efficiency']
    bic_geo[2,:,i] = analysis_c['bic1'], analysis_c['bic2'], analysis_c['path_efficiency']
    

# 输出结果
print("=== Curve A ===")
print(f"BIC (whole): {analysis_a['bic1']:.2f}")
print(f"BIC (segmented): {analysis_a['bic2']:.2f}")
print(f"Plane angle: {analysis_a['angle']:.2f} degrees")
print(f"Best model: {analysis_a['best_model']}")

print("\n=== Curve B ===")
print(f"BIC (whole): {analysis_b['bic1']:.2f}")
print(f"BIC (segmented): {analysis_b['bic2']:.2f}")
print(f"Plane angle: {analysis_b['angle']:.2f} degrees")
print(f"Best model: {analysis_b['best_model']}")

print("\n=== Curve C ===")
print(f"BIC (whole): {analysis_c['bic1']:.2f}")
print(f"BIC (segmented): {analysis_c['bic2']:.2f}")
print(f"Plane angle: {analysis_c['angle']:.2f} degrees")
print(f"Best model: {analysis_c['best_model']}")

# 可视化
plot_curve_with_planes(curve_a, analysis_a, title='Curve A')
plot_curve_with_planes(curve_b, analysis_b, title='Curve B')

#%% 构造整体、分段（4+4）、投影到平面上曲线分段（2+2），3个模型，BIC比较
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def fit_plane_least_squares(points):
    X = points[:, :2]
    Z = points[:, 2]
    A = np.c_[X, np.ones(X.shape[0])]
    coeff, _, _, _ = np.linalg.lstsq(A, Z, rcond=None)
    a, b, c = coeff
    normal = np.array([a, b, -1.0])
    normal = normal / np.linalg.norm(normal)
    arbitrary = np.array([1, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1, 0])
    v1 = np.cross(normal, arbitrary)
    v1 /= np.linalg.norm(v1)
    v2 = np.cross(normal, v1)
    v2 /= np.linalg.norm(v2)
    axes = np.vstack([v1, v2])
    origin = points.mean(axis=0)
    return origin, axes, normal

def project_to_plane(points, origin, axes):
    centered = points - origin
    return centered @ axes.T

def reconstruction_error(points, origin, axes):
    proj = project_to_plane(points, origin, axes)
    recon = proj @ axes + origin
    return np.mean(np.linalg.norm(points - recon, axis=1) ** 2)

def path_efficiency(points, axes):
    proj = project_to_plane(points, points.mean(axis=0), axes)
    direct = np.linalg.norm(proj[-1] - proj[0])
    path = np.sum(np.linalg.norm(np.diff(proj, axis=0), axis=1))
    return 1 - (direct / path)

def bic(error, n_points, n_params):
    return n_points * np.log(error + 1e-8) + n_params * np.log(n_points)

def angle_between_planes(n1, n2):
    cos_angle = np.clip(np.abs(np.dot(n1, n2)), 0, 1)
    return np.degrees(np.arccos(cos_angle))

def analyze_within_plane_segments(points_2d,err):
    N = len(points_2d)
    if N < 4:
        return {'bic1': np.inf, 'bic2': np.inf, 'prefer_split': False}
    direct = np.linalg.norm(points_2d[-1] - points_2d[0])
    path = np.sum(np.linalg.norm(np.diff(points_2d, axis=0), axis=1))
    err1 = path - direct
    bic1 = bic(err1, N, 2)
    
    mid = N // 2
    seg1, seg2 = points_2d[:mid], points_2d[mid:]
    if len(seg1) < 2 or len(seg2) < 2:
        return {'bic1': bic1, 'bic2': np.inf, 'prefer_split': False}
    dir1 = np.linalg.norm(seg1[-1] - seg1[0])
    path1 = np.sum(np.linalg.norm(np.diff(seg1, axis=0), axis=1))
    err_seg1 = path1 - dir1
    dir2 = np.linalg.norm(seg2[-1] - seg2[0])
    path2 = np.sum(np.linalg.norm(np.diff(seg2, axis=0), axis=1))
    err_seg2 = path2 - dir2
    err2 = (err_seg1 + err_seg2)
    bic2 = bic(err2, N, 4)
    return {
        'bic1': bic1,
        'bic2': bic2,
        'prefer_split': bic2 < bic1
    }

def analyze_curve(curve):
    N = len(curve)

    # 模型 1：整体
    o1, ax1, n1 = fit_plane_least_squares(curve)
    err1 = reconstruction_error(curve, o1, ax1)
    pe1 = path_efficiency(curve, ax1)
    bic1 = bic(err1, N, 3)

    # 模型 2：前后两段
    mid = N // 2
    part1, part2 = curve[:mid], curve[mid:]
    o2a, ax2a, n2a = fit_plane_least_squares(part1)
    o2b, ax2b, n2b = fit_plane_least_squares(part2)

    err2a = reconstruction_error(part1, o2a, ax2a)
    err2b = reconstruction_error(part2, o2b, ax2b)
    pe2a = path_efficiency(part1, ax2a)
    pe2b = path_efficiency(part2, ax2b)
    total_error2 = (err2a + err2b)
    bic2 = bic(total_error2, N, 6)

    # 模型 3：在每段内部判断是否还可以细分 （4个平面）
    # part1, part2,part3, part4 = curve[:10], curve[10:20], curve[20:30], curve[30:40]
    # o2a, ax2a, _ = fit_plane_least_squares(part1)
    # o2b, ax2b, _ = fit_plane_least_squares(part2)
    # o2c, ax2c, _ = fit_plane_least_squares(part3)
    # o2d, ax2d, _ = fit_plane_least_squares(part4)

    # err2a = reconstruction_error(part1, o2a, ax2a)
    # err2b = reconstruction_error(part2, o2b, ax2b)
    # err2c = reconstruction_error(part3, o2c, ax2c)
    # err2d = reconstruction_error(part4, o2d, ax2d)
    # total_error2 = (err2a + err2b + err2c + err2d)
    # bic3 = bic(total_error2, N, 12)
    

    # 模型 3：在每段内部判断是否还可以细分 （两个平面内，平面内两条线）
    proj1 = project_to_plane(part1, o2a, ax2a)
    proj2 = project_to_plane(part2, o2b, ax2b)
    ana1 = analyze_within_plane_segments(proj1,err2a)
    ana2 = analyze_within_plane_segments(proj2,err2b)
    prefer_nested = ana1['prefer_split'] and ana2['prefer_split']
    if prefer_nested:
        bic3 = (ana1['bic2'] + ana2['bic2']) / 2
    else:
        bic3 = (ana1['bic2'] + ana2['bic2']) / 2
    # bic3 = (ana1['bic2'] + ana2['bic2']) / 2 if prefer_nested else np.inf
    bic3 = (bic3+bic2)/2
    
    angle = angle_between_planes(n2a, n2b)
    best_model = (
        'nested' if bic3 < min(bic1, bic2)
        else 'segment' if bic2 < bic1
        else 'whole'
    )

    return {
        'bic1': bic1,
        'bic2': bic2,
        'bic3': bic3,
        # 'angle': angle,
        # 'best_model': best_model,
        'n1': n2a,
        'n2': n2b,
        'origin1': o2a,
        'origin2': o2b,
        'whole_normal': n1,
        'whole_origin': o1
    }

def plot_curve_with_planes(curve, analysis, title=''):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(*curve.T, label='curve', color='k')

    xx, yy = np.meshgrid(
        np.linspace(np.min(curve[:, 0]), np.max(curve[:, 0]), 10),
        np.linspace(np.min(curve[:, 1]), np.max(curve[:, 1]), 10)
    )

    # 整体模型平面（灰色）
    n = analysis['whole_normal']
    o = analysis['whole_origin']
    d = -np.dot(n, o)
    zz = (-n[0] * xx - n[1] * yy - d) / n[2]
    ax.plot_surface(xx, yy, zz, alpha=0.3, color='gray')

    # 分段模型两个平面（红蓝）
    for normal, origin, color in zip([analysis['n1'], analysis['n2']],
                                     [analysis['origin1'], analysis['origin2']],
                                     ['r', 'b']):
        d = -np.dot(normal, origin)
        zz = (-normal[0] * xx - normal[1] * yy - d) / normal[2]
        ax.plot_surface(xx, yy, zz, alpha=0.4, color=color)

    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()

#### geo
# geo_spca = pkl.load(open(r'F:\Human_chunking_Huashan_intraoperative\Result\spca\geo_fit.pkl','rb'))
# rank3d_value_gir = geo_spca['rank3d_value_gir']
# rank3d_value_gr4 = geo_spca['rank3d_value_gr4']
# rank3d_value_gr2 = geo_spca['rank3d_value_gr2']

#### lan
# geo_spca = pkl.load(open(r'F:\Human_chunking_Huashan_intraoperative\Result\spca\lan_fit.pkl','rb'))
# rank3d_value_gir = geo_spca['rank3d_value_gr4']
# rank3d_value_gr4 = geo_spca['rank3d_value_gr4']
# rank3d_value_gr2 = geo_spca['rank3d_value_gr4']

bic_geo = np.zeros([3,3,rank3d_value_gr4.shape[0]]) # task * model * repeat
for i in range(rank3d_value_gir.shape[0]):
    curve_a = rank3d_value_gir[i,:,:]
    curve_b = rank3d_value_gr4[i,:,:]
    curve_c = rank3d_value_gr2[i,:,:]
    # 分析
    analysis_a = analyze_curve(curve_a)
    analysis_b = analyze_curve(curve_b)
    analysis_c = analyze_curve(curve_c)
    
    bic_geo[0,:,i] = analysis_a['bic1'], analysis_a['bic2'], analysis_a['bic3']
    bic_geo[1,:,i] = analysis_b['bic1'], analysis_b['bic2'], analysis_b['bic3']
    bic_geo[2,:,i] = analysis_c['bic1'], analysis_c['bic2'], analysis_c['bic3']
    
# 输出结果
# print("=== Curve A ===")
# print(f"BIC (whole):     {analysis_a['bic1']:.2f}")
# print(f"BIC (2 segments):{analysis_a['bic2']:.2f}")
# print(f"BIC (4 nested):  {analysis_a['bic3']:.2f}")
# print(f"Plane angle:     {analysis_a['angle']:.2f} degrees")
# print(f"Best model:      {analysis_a['best_model']}")

# print("\n=== Curve B ===")
# print(f"BIC (whole):     {analysis_b['bic1']:.2f}")
# print(f"BIC (2 segments):{analysis_b['bic2']:.2f}")
# print(f"BIC (4 nested):  {analysis_b['bic3']:.2f}")
# print(f"Plane angle:     {analysis_b['angle']:.2f} degrees")
# print(f"Best model:      {analysis_b['best_model']}")

# print("\n=== Curve C ===")
# print(f"BIC (whole):     {analysis_c['bic1']:.2f}")
# print(f"BIC (2 segments):{analysis_c['bic2']:.2f}")
# print(f"BIC (4 nested):  {analysis_c['bic3']:.2f}")
# print(f"Plane angle:     {analysis_c['angle']:.2f} degrees")
# print(f"Best model:      {analysis_c['best_model']}")

# 可视化
# plot_curve_with_planes(curve_a, analysis_a, title='Curve A')
# plot_curve_with_planes(curve_b, analysis_b, title='Curve B')
# pkl.dump({'rank3d_value_gir':rank3d_value_gir,'rank3d_value_gr4':rank3d_value_gr4,'rank3d_value_gr2':rank3d_value_gr2,
#           'bic_geo':bic_geo}, open(r'F:\Human_chunking_Huashan_intraoperative\Result\spca\geo_fit.pkl','wb'))


# pkl.dump({'rank3d_value_gr4':rank3d_value_gr4,
#           'bic_geo':bic_geo}, open(r'F:\Human_chunking_Huashan_intraoperative\Result\spca\lan_fit.pkl','wb'))

df = pd.DataFrame({'ir_m1':bic_geo[0,0,:], 'ir_m2':bic_geo[0,1,:],'ir_m3':bic_geo[0,2,:],
                   'r4_m1':bic_geo[1,0,:], 'r4_m2':bic_geo[1,1,:],'r4_m3':bic_geo[1,2,:],
                   'r2_m1':bic_geo[2,0,:], 'r2_m2':bic_geo[2,1,:],'r2_m3':bic_geo[2,2,:],})
print(df.mean(0))
#%% 构造整体、分段（4+4）、投影到平面上曲线分段（2+2），3个模型，BIC比较, 修改model3 (0822采用)
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def fit_plane_least_squares(points):
    X = points[:, :2]
    Z = points[:, 2]
    A = np.c_[X, np.ones(X.shape[0])]
    coeff, _, _, _ = np.linalg.lstsq(A, Z, rcond=None)
    a, b, c = coeff
    normal = np.array([a, b, -1.0])
    normal = normal / np.linalg.norm(normal)
    arbitrary = np.array([1, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1, 0])
    v1 = np.cross(normal, arbitrary)
    v1 /= np.linalg.norm(v1)
    v2 = np.cross(normal, v1)
    v2 /= np.linalg.norm(v2)
    axes = np.vstack([v1, v2])
    origin = points.mean(axis=0)
    return origin, axes, normal

def project_to_plane(points, origin, axes):
    centered = points - origin
    return centered @ axes.T

def reconstruction_error(points, origin, axes):
    proj = project_to_plane(points, origin, axes)
    recon = proj @ axes + origin
    return np.mean(np.linalg.norm(points - recon, axis=1) ** 2)

def reconstruction_error_distance(pp):
    plane1 = np.array(best_fit_plane(pp))
    distances = np.zeros([3,timebin])
    a,b,d = plane1
    numerators = np.abs(a * pp[:, 0] + b * pp[:, 1] - pp[:, 2] + d)
    denominator = np.sqrt(a**2 + b**2 + 1)
    distances = numerators / denominator
    
    
def path_efficiency(points, axes):
    proj = project_to_plane(points, points.mean(axis=0), axes)
    direct = np.linalg.norm(proj[-1] - proj[0])
    path = np.sum(np.linalg.norm(np.diff(proj, axis=0), axis=1))
    return 1 - (direct / path)

def bic(error, n_points, n_params):
    return n_points * np.log((error + 1e-8)) + n_params * np.log(n_points)

def angle_between_planes(n1, n2):
    cos_angle = np.clip(np.abs(np.dot(n1, n2)), 0, 1)
    return np.degrees(np.arccos(cos_angle))

def analyze_within_plane_segments(points_2d):
    N = len(points_2d)
    if N < 4:
        return {'bic1': np.inf, 'bic2': np.inf, 'prefer_split': False}
    direct = np.linalg.norm(points_2d[-1] - points_2d[0])
    path = np.sum(np.linalg.norm(np.diff(points_2d, axis=0), axis=1))
    err1 = path - direct
    bic1 = bic(err1, N, 2)
    
    mid = N // 2
    seg1, seg2 = points_2d[:mid], points_2d[mid:]
    if len(seg1) < 2 or len(seg2) < 2:
        return {'bic1': bic1, 'bic2': np.inf, 'prefer_split': False}
    dir1 = np.linalg.norm(seg1[-1] - seg1[0])
    path1 = np.sum(np.linalg.norm(np.diff(seg1, axis=0), axis=1))
    err_seg1 = path1 - dir1
    dir2 = np.linalg.norm(seg2[-1] - seg2[0])
    path2 = np.sum(np.linalg.norm(np.diff(seg2, axis=0), axis=1))
    err_seg2 = path2 - dir2
    err2 = (err_seg1 + err_seg2)
    bic2 = bic(err2, N, 4)
    return {
        'bic1': bic1,
        'bic2': bic2,
        'prefer_split': bic2 < bic1
    }

def analyze_curve(curve):
    from scipy.interpolate import interp1d
    N = len(curve)
    # 模型 1：整体
    o1, ax1, n1 = fit_plane_least_squares(curve[:,:])
    err1 = reconstruction_error(curve, o1, ax1)
    # pe1 = path_efficiency(curve, ax1)
    bic1 = bic(err1, N, 3)

    # 模型 2：前后两段
    # n_points = curve.shape[0]
    # t = np.linspace(0, 1, n_points)       # 原始参数
    # t_new = np.linspace(0, 1, 80)        # 新参数
    # interp_funcs = [interp1d(t, curve[:, i], kind='linear') for i in range(3)]
    # curve_new = np.vstack([f(t_new) for f in interp_funcs]).T  # (40,3)

    mid = N//2
    part1, part2 = curve[:mid], curve[mid:]
    o2a, ax2a, n2a = fit_plane_least_squares(part1[:,:])
    o2b, ax2b, n2b = fit_plane_least_squares(part2[:,:])
    angle = angle_between_planes(n2a, n2b)
    
    err2a = reconstruction_error(part1, o2a, ax2a)
    err2b = reconstruction_error(part2, o2b, ax2b)
    # pe2a = path_efficiency(part1, ax2a)
    # pe2b = path_efficiency(part2, ax2b)
    total_error2 = (err2a + err2b)
    bic2 = bic(total_error2, N, 6)
    # print([err2a,err2b,bic2])
    # if bic2>0:
    #     print([err2a,err2b])

    # 模型 3：在每段内部判断是否还可以细分 （两个平面内，平面内两条线）
    proj1 = project_to_plane(part1, o2a, ax2a)
    proj2 = project_to_plane(part2, o2b, ax2b)
    ana1 = analyze_within_plane_segments(proj1)
    ana2 = analyze_within_plane_segments(proj2)
    bic31 = (ana1['bic1'] + ana2['bic1']) / 2 
    bic32= (ana1['bic2'] + ana2['bic2']) / 2 


    return {
        'bic1': bic1,
        'bic2': bic2,
        'bic31': bic31,
        'bic32': bic32,
        'angle': angle,
        # 'best_model': best_model,
        'n1': n2a,
        'n2': n2b,
        'origin1': o2a,
        'origin2': o2b,
        'whole_normal': n1,
        'whole_origin': o1
    }

def plot_curve_with_planes(curve, analysis, title=''):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(*curve.T, label='curve', color='k')

    xx, yy = np.meshgrid(
        np.linspace(np.min(curve[:, 0]), np.max(curve[:, 0]), 10),
        np.linspace(np.min(curve[:, 1]), np.max(curve[:, 1]), 10)
    )

    # 整体模型平面（灰色）
    n = analysis['whole_normal']
    o = analysis['whole_origin']
    d = -np.dot(n, o)
    zz = (-n[0] * xx - n[1] * yy - d) / n[2]
    ax.plot_surface(xx, yy, zz, alpha=0.3, color='gray')

    # 分段模型两个平面（红蓝）
    for normal, origin, color in zip([analysis['n1'], analysis['n2']],
                                     [analysis['origin1'], analysis['origin2']],
                                     ['r', 'b']):
        d = -np.dot(normal, origin)
        zz = (-normal[0] * xx - normal[1] * yy - d) / normal[2]
        ax.plot_surface(xx, yy, zz, alpha=0.4, color=color)

    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()

#### geo
geo_spca = pkl.load(open(r'F:\Human_chunking_Huashan_intraoperative\Result\spca\geo_fit.pkl','rb'))
rank3d_value_gir = geo_spca['rank3d_value_gir']
rank3d_value_gr4 = geo_spca['rank3d_value_gr4']
rank3d_value_gr2 = geo_spca['rank3d_value_gr2']

### lan
# geo_spca = pkl.load(open(r'F:\Human_chunking_Huashan_intraoperative\Result\spca\lan_fit(0822model3).pkl','rb'))
# rank3d_value_gir = geo_spca['rank3d_value_gir']
# rank3d_value_gr4 = geo_spca['rank3d_value_gr4']
# rank3d_value_gr2 = geo_spca['rank3d_value_gr4']

#### mus
# geo_spca = pkl.load(open(r'F:\Human_chunking_Huashan_intraoperative\Result\spca\mus_fit(0822model3).pkl','rb'))
# rank3d_value_gir = geo_spca['rank3d_value_gir']
# rank3d_value_gr4 = geo_spca['rank3d_value_gr4']
# rank3d_value_gr2 = geo_spca['rank3d_value_gr2']

for i in range(1, rank3d_value_gr4.shape[0]):
    for j in range(3):
        r, p = stats.pearsonr(rank3d_value_gir[i,:,j], rank3d_value_gir[:,:,j].mean(0))
        if r <0:
            rank3d_value_gir[i,:,j] = rank3d_value_gir[i,:,j]*-1
            
        r, p = stats.pearsonr(rank3d_value_gr4[i,:,j], rank3d_value_gr4[:,:,j].mean(0))
        if r <0:
            rank3d_value_gr4[i,:,j] = rank3d_value_gr4[i,:,j]*-1
        
        r, p = stats.pearsonr(rank3d_value_gr2[i,:,j], rank3d_value_gr2[:,:,j].mean(0))
        if r <0:
            rank3d_value_gr2[i,:,j] = rank3d_value_gr2[i,:,j]*-1

bic_geo = np.zeros([3, 4, rank3d_value_gr4.shape[0]]) # task * model * repeat
angle_m2 = np.zeros([3, rank3d_value_gr4.shape[0]]) # task * model * repeat
for i in range(rank3d_value_gr4.shape[0]):
    curve_a = rank3d_value_gir[i,:,:]
    curve_b = rank3d_value_gr4[i,:,:]
    curve_c = rank3d_value_gr2[i,:,:]
    # 分析
    analysis_a = analyze_curve(curve_a)
    analysis_b = analyze_curve(curve_b)
    analysis_c = analyze_curve(curve_c)
    
    bic_geo[0,:,i] = analysis_a['bic1'], analysis_a['bic2'], analysis_a['bic31'], analysis_a['bic32']
    bic_geo[1,:,i] = analysis_b['bic1'], analysis_b['bic2'], analysis_b['bic31'], analysis_b['bic32']
    bic_geo[2,:,i] = analysis_c['bic1'], analysis_c['bic2'], analysis_c['bic31'], analysis_c['bic32']
    
    angle_m2[0,i] = analysis_a['angle']
    angle_m2[1,i] = analysis_b['angle']
    angle_m2[2,i] = analysis_c['angle']

# pkl.dump({'rank3d_value_gir':rank3d_value_gir,'rank3d_value_gr4':rank3d_value_gr4,'rank3d_value_gr2':rank3d_value_gr2,
#           'bic_geo':bic_geo}, open(r'F:\Human_chunking_Huashan_intraoperative\Result\spca\mus_fit(0822model3).pkl','wb'))

df = pd.DataFrame({'ir_m1':bic_geo[0,0,:], 'ir_m2':bic_geo[0,1,:],'ir_m31':bic_geo[0,2,:],'ir_m32':bic_geo[0,3,:],
                   'r4_m1':bic_geo[1,0,:], 'r4_m2':bic_geo[1,1,:],'r4_m31':bic_geo[1,2,:],'r4_m32':bic_geo[1,3,:],
                   'r2_m1':bic_geo[2,0,:], 'r2_m2':bic_geo[2,1,:],'r2_m31':bic_geo[2,2,:],'r2_m32':bic_geo[2,3,:]})
print(df.mean(0))
# print(df.std(0))

with plt.style.context(style_path):
    get_ipython().run_line_magic('matplotlib', 'inline')
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(1,3, figsize=(3, 1.2), dpi=300)
    plotx = rank3d_value_gr2
    for i in range(20):
        ax[0].plot(plotx[i,:,0])
        ax[1].plot(plotx[i,:,1])
        ax[2].plot(plotx[i,:,2])

exp_var_ir[:,:3].mean(0)
exp_var_r4[:,:3].mean(0)
exp_var_r2[:,:3].mean(0)

# angle_m2.mean(1)
#%% 构造整体、分段（4+4）、投影到平面上曲线分段（2+2），3个模型，pca求平面，BIC比较 (0918采用)
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import f

def curve_length(points):
    pts = np.asarray(points)  # 替换为你的 Nx3 数据
    diffs = np.diff(pts, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    # L_open = segment_lengths.sum()
    return segment_lengths

# def fit_plane_least_squares(points):
#     X = points[:, :2]
#     Z = points[:, 2]
#     A = np.c_[X, np.ones(X.shape[0])]
#     coeff, _, _, _ = np.linalg.lstsq(A, Z, rcond=None)
#     a, b, c = coeff
#     normal = np.array([a, b, -1.0])
#     normal = normal / np.linalg.norm(normal)
#     arbitrary = np.array([1, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1, 0])
#     v1 = np.cross(normal, arbitrary)
#     v1 /= np.linalg.norm(v1)
#     v2 = np.cross(normal, v1)
#     v2 /= np.linalg.norm(v2)
#     axes = np.vstack([v1, v2])
#     origin = points.mean(axis=0)
#     return origin, axes, normal, coeff

def plane_angle(normal1, normal2):
    """
    计算两个平面的夹角 (单位: 弧度)
    normal1, normal2: 平面法向量 (长度为3的numpy数组)
    """
    n1 = normal1 / np.linalg.norm(normal1)
    n2 = normal2 / np.linalg.norm(normal2)
    cos_theta = np.abs(np.dot(n1, n2))  # 取绝对值，避免负号
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # 数值稳定性
    return np.arccos(cos_theta)

def fit_plane_pca(points):
    """
    用 PCA 拟合平面
    返回：平面上一点（均值点）、法向量、平面内的两个正交方向
    """
    centroid = points.mean(axis=0)  # 平面上一点
    centered = points - centroid
    U, S, Vt = np.linalg.svd(centered)
    
    normal = Vt[-1]        # 法向量 = 最小特征值对应方向
    axis1 = Vt[0]          # 平面内第一主方向
    axis2 = Vt[1]          # 平面内第二主方向
    
    residuals = np.abs(centered @ normal)   # 投影长度
    residuals /= np.linalg.norm(normal)
    residual_variance = np.mean(residuals**2)
    # n=80
    # log_likelihood = -n/2 * np.log(2 * np.pi * residual_variance) - np.sum(residuals**2) / (2 * residual_variance)
    return centroid, normal, axis1, axis2, np.mean(residuals**2)#,log_likelihood

def project_points_to_plane(points, plane_point, normal):
    """
    将点投影到平面上
    """
    normal = normal / np.linalg.norm(normal)
    diff = points - plane_point
    dist = np.dot(diff, normal)[:, np.newaxis]
    projected_points = points - dist * normal
    return projected_points

def to_plane_coordinates(projected_points, plane_point, axis1, axis2):
    """
    把投影点转换到平面坐标系 (axis1, axis2)
    """
    diff = projected_points - plane_point
    x_coords = np.dot(diff, axis1)
    y_coords = np.dot(diff, axis2)
    return np.vstack((x_coords, y_coords)).T

# def project_to_plane(points, origin, axes):
#     centered = points - origin
#     return centered @ axes.T

def reconstruction_error(points, origin, axes):
    proj = project_to_plane(points, origin, axes)
    recon = proj @ axes + origin
    return np.mean(np.linalg.norm(points - recon, axis=1) ** 2)

def reconstruction_error_distance(pp, coeff):
    a,b,d = coeff
    numerators = np.abs(a * pp[:, 0] + b * pp[:, 1] - pp[:, 2] + d)
    denominator = np.sqrt(a**2 + b**2 + 1)
    distances = numerators / denominator
    return distances.mean()
    
def path_efficiency(points, axes):
    proj = project_to_plane(points, points.mean(axis=0), axes)
    direct = np.linalg.norm(proj[-1] - proj[0])
    path = np.sum(np.linalg.norm(np.diff(proj, axis=0), axis=1))
    return 1 - (direct / path)

def bic(error, n_points, n_params):
    return n_points * np.log((error + 1e-8)) + n_params * np.log(n_points)

def angle_between_planes(n1, n2):
    cos_angle = np.clip(np.abs(np.dot(n1, n2)), 0, 1)
    return np.degrees(np.arccos(cos_angle))

def analyze_within_plane_segments(points_2d):
    N = len(points_2d)
    if N < 4:
        return {'bic1': np.inf, 'bic2': np.inf, 'prefer_split': False}
    direct = np.linalg.norm(points_2d[-1] - points_2d[0])
    path = np.sum(np.linalg.norm(np.diff(points_2d, axis=0), axis=1))
    err1 = path - direct
    bic1 = bic(err1, N, 2)
    
    mid = N // 2
    seg1, seg2 = points_2d[:mid], points_2d[mid:]
    if len(seg1) < 2 or len(seg2) < 2:
        return {'bic1': bic1, 'bic2': np.inf, 'prefer_split': False}
    dir1 = np.linalg.norm(seg1[-1] - seg1[0])
    path1 = np.sum(np.linalg.norm(np.diff(seg1, axis=0), axis=1))
    err_seg1 = path1 - dir1
    dir2 = np.linalg.norm(seg2[-1] - seg2[0])
    path2 = np.sum(np.linalg.norm(np.diff(seg2, axis=0), axis=1))
    err_seg2 = path2 - dir2
    err2 = (err_seg1 + err_seg2)
    bic2 = bic(err2, N, 4)
    #### bic差值
    bic_delta = bic((err1-err2)/(4-2),N,4)
    return {
        'err1':err1,
        'err2':err2,
        'bic1': bic1,
        'bic2': bic2,
        'bic_delta':bic_delta,
        'prefer_split': bic2 < bic1
    }

def analyze_curve(curve):
    from scipy.interpolate import interp1d
    N = len(curve)
    cur_len = curve_length(curve)
    cur_len = cur_len.sum()
    # 模型 1：整体
    # o1, ax1, n1, coeff = fit_plane_least_squares(curve[:,:])
    # coeff = np.array(best_fit_plane(curve))
    # err1 = reconstruction_error_distance(curve, coeff)
    # pe1 = path_efficiency(curve, ax1)
    output = {}
    centroid, normal,axis1, axis2, err1 = fit_plane_pca(curve)
    output['m1_centroid'] = centroid
    output['m1'] = [axis1, axis2]
    output['m1_normal'] = normal
    bic1 = bic(err1, N, 3)

    # 模型 2：前后两段
    # n_points = curve.shape[0]
    # t = np.linspace(0, 1, n_points)       # 原始参数
    # t_new = np.linspace(0, 1, 80)        # 新参数
    # interp_funcs = [interp1d(t, curve[:, i], kind='linear') for i in range(3)]
    # curve_new = np.vstack([f(t_new) for f in interp_funcs]).T  # (40,3)

    mid = N//2
    part1, part2 = curve[0:mid,:], curve[mid+3:-5,:]
    centroid, normal1, axis1, axis2, err2a = fit_plane_pca(part1)
    output['m2_1_centroid'] = centroid
    output['m2_1'] = [axis1, axis2]
    output['m2_normal1'] = normal1
    # proj_points1 = project_points_to_plane(part1, centroid, normal1)
    # coords_2d1 = to_plane_coordinates(proj_points1, centroid, axis1, axis2)
    # var1 = np.var(coords_2d1, axis=0, ddof=0).sum()
    
    centroid, normal2, axis1, axis2, err2b = fit_plane_pca(part2)
    output['m2_2_centroid'] = centroid
    output['m2_2'] = [axis1, axis2]
    output['m2_normal2'] = normal2
    # proj_points1 = project_points_to_plane(part2, centroid, normal2)
    # coords_2d1 = to_plane_coordinates(proj_points1, centroid, axis1, axis2)
    # var2 = np.var(coords_2d1, axis=0, ddof=0).sum()
    
    angle = plane_angle(normal1, normal2)
    total_error2 = (err2a + err2b)#/angle**2
    bic2 = bic(total_error2, N, 6)
    
    delta_error2 = (err1 - (err2a + err2b))#/(6 - 3)
    # bic2_delta = bic(delta_error2, N, 6) * -1
    bic2_delta = bic2 - bic1
    
    # F 统计量
    numerator = (err1 - (err2a + err2b)) / (6 - 3)
    denominator =  (err2a + err2b) / (80 - 6)
    f_statistic = numerator / denominator
    # 计算 p-value
    df1 = 6 - 3
    df2 = 80 - 6
    p_value = 1 - f.cdf(f_statistic, df1, df2)
    # print([err2a,err2b,bic2])
    # if bic2>0:
    #     print([err2a,err2b])


    # 模型 3：在每段内部判断是否还可以细分 （两个平面内，平面内两条线）
    # part1, part2 = curve[0:mid,:], curve[mid:,:]
    centroid, normal1, axis1, axis2, err3a = fit_plane_pca(part1)
    proj_points1 = project_points_to_plane(part1, centroid, normal1)
    coords_2d1 = to_plane_coordinates(proj_points1, centroid, axis1, axis2)
    
    centroid, normal2, axis1, axis2, err3b = fit_plane_pca(part2)
    proj_points2 = project_points_to_plane(part2, centroid, normal2)
    coords_2d2 = to_plane_coordinates(proj_points2, centroid, axis1, axis2)
    # proj1 = project_to_plane(part1, o2a, ax2a)
    # proj2 = project_to_plane(part2, o2b, ax2b)
    ana1 = analyze_within_plane_segments(coords_2d1)
    ana2 = analyze_within_plane_segments(coords_2d1)
    bic31 = (ana1['bic1'] + ana2['bic1'])# / 2 
    bic32= (ana1['bic2'] + ana2['bic2'])# / 2 
    # bic_delta = (ana1['bic_delta'] + ana2['bic_delta']) / 2 
    delta_error3 = ((ana1['err1'] + ana2['err1']) - (ana1['err2'] + ana2['err2']))
    # bic3_delta = bic(delta_error3, N, 8) * -1
    bic3_delta = (bic32 - bic31)

    return {
        'bic1': bic1,
        'bic2': bic2,
        'bic31': bic31,
        'bic32': bic32,
        'bic2_delta': bic2_delta,
        'bic3_delta': bic3_delta,
        'total_error2':delta_error2,
        'total_error3':delta_error3,
        'angle': angle,
        'f_statistic':f_statistic,
        'output':output
    }

def plot_curve_with_planes(curve, analysis, title=''):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(*curve.T, label='curve', color='k')

    xx, yy = np.meshgrid(
        np.linspace(np.min(curve[:, 0]), np.max(curve[:, 0]), 10),
        np.linspace(np.min(curve[:, 1]), np.max(curve[:, 1]), 10)
    )

    # 整体模型平面（灰色）
    n = analysis['whole_normal']
    o = analysis['whole_origin']
    d = -np.dot(n, o)
    zz = (-n[0] * xx - n[1] * yy - d) / n[2]
    ax.plot_surface(xx, yy, zz, alpha=0.3, color='gray')

    # 分段模型两个平面（红蓝）
    for normal, origin, color in zip([analysis['n1'], analysis['n2']],
                                     [analysis['origin1'], analysis['origin2']],
                                     ['r', 'b']):
        d = -np.dot(normal, origin)
        zz = (-normal[0] * xx - normal[1] * yy - d) / normal[2]
        ax.plot_surface(xx, yy, zz, alpha=0.4, color=color)

    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()

#### geo
# geo_spca = pkl.load(open(r'F:\Human_chunking_Huashan_intraoperative\Result\spca\geo_fit(time40)_correct(1).pkl','rb'))
# geo_spca = pkl.load(open(r'F:\Human_chunking_Huashan_intraoperative\Result\spca\geo_fit(time40)_correct(1_3)_PermData.pkl','rb'))
# rank3d_value_gir = geo_spca['rank3d_value_gir']
# rank3d_value_gr4 = geo_spca['rank3d_value_gr4']
# rank3d_value_gr2 = geo_spca['rank3d_value_gr2']

### lan
# geo_spca = pkl.load(open(r'F:\Human_chunking_Huashan_intraoperative\Result\spca\lan_fit(time40)_correct(1_3)_PermData.pkl','rb'))
# rank3d_value_gir = geo_spca['rank3d_value_gir']
# rank3d_value_gr4 = geo_spca['rank3d_value_gr4']
# rank3d_value_gr2 = geo_spca['rank3d_value_gr4']

#### mus
# geo_spca = pkl.load(open(r'F:\Human_chunking_Huashan_intraoperative\Result\spca\mus_fit(time40)_correct(1_3).pkl','rb'))
# rank3d_value_gir = geo_spca['rank3d_value_gir']
# rank3d_value_gr4 = geo_spca['rank3d_value_gr4']
# rank3d_value_gr2 = geo_spca['rank3d_value_gr2']

for i in range(1, rank3d_value_gr4.shape[0]):
    for j in range(3):
        r, p = stats.pearsonr(rank3d_value_gir[i,:,j], rank3d_value_gir[:,:,j].mean(0))
        if r <0:
            rank3d_value_gir[i,:,j] = rank3d_value_gir[i,:,j]*-1
            
        r, p = stats.pearsonr(rank3d_value_gr4[i,:,j], rank3d_value_gr4[:,:,j].mean(0))
        if r <0:
            rank3d_value_gr4[i,:,j] = rank3d_value_gr4[i,:,j]*-1
        
        r, p = stats.pearsonr(rank3d_value_gr2[i,:,j], rank3d_value_gr2[:,:,j].mean(0))
        if r <0:
            rank3d_value_gr2[i,:,j] = rank3d_value_gr2[i,:,j]*-1

bic_geo = np.zeros([3, 8, rank3d_value_gr4.shape[0]]) # task * model * repeat
angle_m2 = np.zeros([3, rank3d_value_gr4.shape[0]]) # task * model * repeat
planParam = np.zeros([3, rank3d_value_gr4.shape[0]]) # task * model * repeat
cur_len = np.zeros([rank3d_value_gr4.shape[0],rank3d_value_gr4.shape[1]-1,3])
fit_result_a = []
fit_result_b = []
fit_result_c = []

for i in range(rank3d_value_gr4.shape[0]):
    curve_a = rank3d_value_gir[i,:,:]
    curve_b = rank3d_value_gr4[i,:,:]
    curve_c = rank3d_value_gr2[i,:,:]
    # 分析
    analysis_a = analyze_curve(curve_a)
    analysis_b = analyze_curve(curve_b)
    analysis_c = analyze_curve(curve_c)
    fit_result_a.append(analysis_a)
    fit_result_b.append(analysis_b)
    fit_result_c.append(analysis_c)
    
    cur_len[i,:,0] = curve_length(curve_a)
    cur_len[i,:,1] = curve_length(curve_b)
    cur_len[i,:,2] = curve_length(curve_c)
    
    bic_geo[0,:,i] = analysis_a['bic1'], analysis_a['bic2'], analysis_a['bic31'], analysis_a['bic32'],analysis_a['bic2_delta'], analysis_a['bic3_delta'], analysis_a['total_error2'], analysis_a['total_error3']
    bic_geo[1,:,i] = analysis_b['bic1'], analysis_b['bic2'], analysis_b['bic31'], analysis_b['bic32'],analysis_b['bic2_delta'], analysis_b['bic3_delta'], analysis_b['total_error2'], analysis_b['total_error3']
    bic_geo[2,:,i] = analysis_c['bic1'], analysis_c['bic2'], analysis_c['bic31'], analysis_c['bic32'],analysis_c['bic2_delta'], analysis_c['bic3_delta'], analysis_c['total_error2'], analysis_c['total_error3']
    
    angle_m2[0,i] = analysis_a['angle']
    angle_m2[1,i] = analysis_b['angle']
    angle_m2[2,i] = analysis_c['angle']
    

df = pd.DataFrame({'ir_m1':bic_geo[0,0,:], 'ir_m2':bic_geo[0,1,:],'ir_m31':bic_geo[0,2,:],'ir_m32':bic_geo[0,3,:],'ir_bic2_delta':bic_geo[0,4,:],'ir_bic3_delta':bic_geo[0,5,:],
                   'r4_m1':bic_geo[1,0,:], 'r4_m2':bic_geo[1,1,:],'r4_m31':bic_geo[1,2,:],'r4_m32':bic_geo[1,3,:],'r4_bic2_delta':bic_geo[1,4,:],'r4_bic3_delta':bic_geo[1,5,:],
                   'r2_m1':bic_geo[2,0,:], 'r2_m2':bic_geo[2,1,:],'r2_m31':bic_geo[2,2,:],'r2_m32':bic_geo[2,3,:],'r2_bic2_delta':bic_geo[2,4,:],'r2_bic3_delta':bic_geo[2,5,:]})
print(df.mean(0))
# print(df.std(0))
# cur_len[:,25:,:].mean(0).mean(0)

angle_m2.mean(1)
# stats.ttest_ind(angle_m2[0,:], angle_m2[1,:])

stats.ttest_ind(df['ir_bic2_delta'][:50], df['r4_bic2_delta'][:50]) 
stats.ttest_ind(df['ir_bic3_delta'][:50], df['r4_bic3_delta'][:50]) 
# stats.ttest_ind(df['ir_m32'][:50], df['r4_m32'][:50]) 

#### save result
# pkl.dump({'rank3d_value_gir':rank3d_value_gir,'rank3d_value_gr4':rank3d_value_gr4,'rank3d_value_gr2':rank3d_value_gr2,
#           'angle_m2':angle_m2,'bic_df':df,'fit_result_a':fit_result_a,'fit_result_b':fit_result_b,
#           'fr_allsub_ir':fr_allsub_ir, 'fr_allsub_r4':fr_allsub_r4,'fr_allsub_r2':fr_allsub_r2,
#           'fr_reduce_ir':fr_reduce_ir, 'fr_reduce_r4':fr_reduce_r4,'fr_reduce_r2':fr_reduce_r2,
#           'fit_result_c':fit_result_c,'exp_var':exp_var}, 
#          open(r'F:\Human_chunking_Huashan_intraoperative\Result\spca\10sub_mus_fit(time40)_correct(1_3)_PermData.pkl','wb'))

#%% 构造整体、分段（4+4）、投影到平面上曲线分段（2+2），3个模型，BIC比较, box plot
# bic_geo[2,2,:].mean()
# stats.ttest_ind(bic_geo[1,2,:], bic_geo[2,2,:])

# #### plot 3个序列的m1和m2
# with plt.style.context(style_path):
#     import seaborn as sns
#     get_ipython().run_line_magic('matplotlib', 'inline')
#     colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
#     fig, ax = plt.subplots(1,1, figsize=(4, 2), dpi=300)
    
#     df_melted = df[['ir_m1','ir_m2','ir_m3','r4_m1','r4_m2','r4_m3','r2_m1','r2_m2','r2_m3']].melt(
#         var_name='Variable', value_name='Value')
#     # 画小提琴图
#     # sns.violinplot(x='Variable', y='Value', ax=ax, data=df_melted, color=colors[1])
#     # sns.boxplot(x='Variable', y='Value', ax=ax, data=df_melted, color=colors[0])
#     # 自定义 x 位置
#     positions = [1, 2, 3, 4, 5, 6]
#     # 自定义 outlier 的样式
#     flierprops = dict(marker='.', color='gray', markersize=5)
#     # 自定义离群值的判断阈值，例如设为1.0（默认是1.5）
#     whis = 2.0
#     # 绘制 boxplot
#     bp = ax.boxplot(df.loc[:,['ir_m1','r4_m1','r2_m1', 'ir_m2', 'r4_m2','r2_m2']],
#                     positions=positions, widths=0.5,
#                      flierprops=flierprops, whis=whis, patch_artist=True)
#     color1 = ['gray', colors[0], colors[1],'gray', colors[0], colors[1]]
#     for patch, color in zip(bp['boxes'], color1):
#         patch.set_facecolor(color)
#     # 修改中位数线的颜色
#     for median in bp['medians']:
#         median.set_color('k')
#         median.set_linewidth(1)
#     # 去掉边框线
#     for item in bp['boxes']:
#         item.set_linewidth(0)
#     for item in bp['caps']:
#         item.set_linewidth(0)
        
#     ax.set_xticklabels(['L0','L1','L2','L0','L1','L2'])
#     ax.set_ylabel('BIC')
#     ax.set_ylim([30, 120])
    
#### plot 3个序列的m1
with plt.style.context(style_path):
    get_ipython().run_line_magic('matplotlib', 'inline')
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(1,1, figsize=(1.5, 1.5), dpi=300)
    
    positions = [1, 2, 3]
    flierprops = dict(marker='.', color='gray', markersize=5)
    whis = 2.0
    bp = ax.boxplot(df.loc[:,['ir_m1', 'r4_m1','r2_m1']],
                    positions=positions, widths=0.5,
                     flierprops=flierprops, whis=whis, patch_artist=True)
    color1 = ['gray', colors[0], colors[1]]
    for patch, color in zip(bp['boxes'], color1):
        patch.set_facecolor(color)
    # 修改中位数线的颜色
    for median in bp['medians']:
        median.set_color('k')
        median.set_linewidth(1)
    # 去掉边框线
    for item in bp['boxes']:
        item.set_linewidth(0)
    for item in bp['caps']:
        item.set_linewidth(0)
    ax.set_xticklabels(['L0','L1','L2'])
    # ax.set_ylabel('BIC')
    # ax.set_ylim([40, 100])

#### plot 3个序列的m2
with plt.style.context(style_path):
    get_ipython().run_line_magic('matplotlib', 'inline')
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(1,1, figsize=(1.5, 1.5), dpi=300)
    
    positions = [1, 2, 3]
    flierprops = dict(marker='.', color='gray', markersize=5)
    whis = 2.0
    bp = ax.boxplot(df.loc[:,['ir_bic2_delta','r4_bic2_delta','r2_bic2_delta']],
                    positions=positions, widths=0.5,
                     flierprops=flierprops, whis=whis, patch_artist=True)
    color1 = ['gray', colors[0], colors[1]]
    for patch, color in zip(bp['boxes'], color1):
        patch.set_facecolor(color)
    # 修改中位数线的颜色
    for median in bp['medians']:
        median.set_color('k')
        median.set_linewidth(1)
    # 去掉边框线
    for item in bp['boxes']:
        item.set_linewidth(0)
    for item in bp['caps']:
        item.set_linewidth(0)
    ax.set_xticklabels(['L0','L1','L2'])
    # ax.set_ylabel('BIC')
    # ax.set_ylim([10, 110])
    
#### plot 3个序列的m3
with plt.style.context(style_path):
    get_ipython().run_line_magic('matplotlib', 'inline')
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(1,1, figsize=(1.5, 1.5), dpi=300)
    
    positions = [1, 2, 3]
    flierprops = dict(marker='.', color='gray', markersize=5)
    whis = 2.0
    bp = ax.boxplot(df.loc[:,['ir_bic3_delta','r4_bic3_delta','r2_bic3_delta']],
                    positions=positions, widths=0.5,
                     flierprops=flierprops, whis=whis, patch_artist=True)
    color1 = ['gray', colors[0], colors[1]]
    for patch, color in zip(bp['boxes'], color1):
        patch.set_facecolor(color)
    # 修改中位数线的颜色
    for median in bp['medians']:
        median.set_color('k')
        median.set_linewidth(1)
    # 去掉边框线
    for item in bp['boxes']:
        item.set_linewidth(0)
    for item in bp['caps']:
        item.set_linewidth(0)
    ax.set_xticklabels(['L0','L1','L2'])
    # ax.set_ylabel('BIC')
    # ax.set_ylim([35, 70])
 
    
stats.ttest_ind(df['ir_m1'], df['r4_m1'])  
stats.ttest_ind(df['r2_bic2_delta'], df['ir_bic2_delta'])  
stats.ttest_ind(df['r2_bic3_delta'], df['r4_bic3_delta'])  
p*np.sqrt(9)

# 采用permutation test
# np.sum(df['r4_bic2_delta']<df['ir_bic2_delta'].mean())
# np.sum(df['r2_bic3_delta']<df['r4_bic3_delta'].mean())

   
#%% plot BIC delta
with plt.style.context(style_path):
    get_ipython().run_line_magic('matplotlib', 'inline')
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(1,1, figsize=(1.5, 1.5), dpi=300)
    
    positions = [1, 2, 3]
    flierprops = dict(marker='.', color='gray', markersize=5)
    whis = 2.0
    bp = ax.boxplot(df.loc[:,['ir_m1', 'r4_m1','r2_m1']],
                    positions=positions, widths=0.5,
                     flierprops=flierprops, whis=whis, patch_artist=True)
    color1 = ['gray', colors[0], colors[1]]
    for patch, color in zip(bp['boxes'], color1):
        patch.set_facecolor(color)
    # 修改中位数线的颜色
    for median in bp['medians']:
        median.set_color('k')
        median.set_linewidth(1)
    # 去掉边框线
    for item in bp['boxes']:
        item.set_linewidth(0)
    for item in bp['caps']:
        item.set_linewidth(0)
    ax.set_xticklabels(['L0','L1','L2'])
    ax.set_ylabel('BIC')
    # ax.set_ylim([40, 100])

#### plot 3个序列的m2
with plt.style.context(style_path):
    get_ipython().run_line_magic('matplotlib', 'inline')
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(1,1, figsize=(1.5, 1.5), dpi=300)
    
    positions = [1, 2, 3]
    flierprops = dict(marker='.', color='gray', markersize=5)
    whis = 2.0
    tmp = df.loc[:,['ir_m2', 'r4_m2','r2_m2']].values# - df.loc[:,['ir_m1', 'r4_m1','r2_m1']].values
    bp = ax.boxplot(tmp,
                    positions=positions, widths=0.5,
                     flierprops=flierprops, whis=whis, patch_artist=True)
    color1 = ['gray', colors[0], colors[1]]
    for patch, color in zip(bp['boxes'], color1):
        patch.set_facecolor(color)
    # 修改中位数线的颜色
    for median in bp['medians']:
        median.set_color('k')
        median.set_linewidth(1)
    # 去掉边框线
    for item in bp['boxes']:
        item.set_linewidth(0)
    for item in bp['caps']:
        item.set_linewidth(0)
    ax.set_xticklabels(['L0','L1','L2'])
    ax.set_ylabel('BIC')
    # ax.set_ylim([10, 110])
    
#### plot 3个序列的m3
with plt.style.context(style_path):
    get_ipython().run_line_magic('matplotlib', 'inline')
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(1,1, figsize=(1.5, 1.5), dpi=300)
    
    positions = [1, 2, 3]
    flierprops = dict(marker='.', color='gray', markersize=5)
    whis = 2.0
    tmp = df.loc[:,['ir_m32', 'r4_m32','r2_m32']].values# - df.loc[:,['ir_m31', 'r4_m31','r2_m31']].values
    bp = ax.boxplot(tmp,
                    positions=positions, widths=0.5,
                     flierprops=flierprops, whis=whis, patch_artist=True)
    color1 = ['gray', colors[0], colors[1]]
    for patch, color in zip(bp['boxes'], color1):
        patch.set_facecolor(color)
    # 修改中位数线的颜色
    for median in bp['medians']:
        median.set_color('k')
        median.set_linewidth(1)
    # 去掉边框线
    for item in bp['boxes']:
        item.set_linewidth(0)
    for item in bp['caps']:
        item.set_linewidth(0)
    ax.set_xticklabels(['L0','L1','L2'])
    ax.set_ylabel('BIC')
    # ax.set_ylim([35, 70])
 
    
t,p = stats.ttest_ind(df['r4_m2'][:50], df['r2_m2'][:50])  
t,p = stats.ttest_ind(df['r4_m2']-df['r4_m1'], df['ir_m2']-df['ir_m1'])  
p*np.sqrt(9)

# 采用permutation test
np.sum(df['r4_m2']<df['ir_m2'].mean())

#%% plot BIC, 按task
with plt.style.context(style_path):
    get_ipython().run_line_magic('matplotlib', 'inline')
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(1,1, figsize=(1.5, 1.5), dpi=300)
    
    positions = [1, 2]
    flierprops = dict(marker='.', color='gray', markersize=5)
    whis = 2.0
    bp = ax.boxplot(df.loc[:,['ir_m1', 'ir_m2']],
                    positions=positions, widths=0.5,
                     flierprops=flierprops, whis=whis, patch_artist=True)
    color1 = ['gray', colors[0], colors[1]]
    for patch, color in zip(bp['boxes'], color1):
        patch.set_facecolor(color)
    # 修改中位数线的颜色
    for median in bp['medians']:
        median.set_color('k')
        median.set_linewidth(1)
    # 去掉边框线
    for item in bp['boxes']:
        item.set_linewidth(0)
    for item in bp['caps']:
        item.set_linewidth(0)
    ax.set_xticklabels(['L0','L1'])
    ax.set_ylabel('BIC')
    # ax.set_ylim([40, 100])

with plt.style.context(style_path):
    get_ipython().run_line_magic('matplotlib', 'inline')
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(1,1, figsize=(1.5, 1.5), dpi=300)
    
    positions = [1, 2]
    flierprops = dict(marker='.', color='gray', markersize=5)
    whis = 2.0
    bp = ax.boxplot(df.loc[:,['r4_m1', 'r4_m2']],
                    positions=positions, widths=0.5,
                     flierprops=flierprops, whis=whis, patch_artist=True)
    color1 = ['gray', colors[0], colors[1]]
    for patch, color in zip(bp['boxes'], color1):
        patch.set_facecolor(color)
    # 修改中位数线的颜色
    for median in bp['medians']:
        median.set_color('k')
        median.set_linewidth(1)
    # 去掉边框线
    for item in bp['boxes']:
        item.set_linewidth(0)
    for item in bp['caps']:
        item.set_linewidth(0)
    ax.set_xticklabels(['L0','L1'])
    ax.set_ylabel('BIC')
    # ax.set_ylim([40, 100])

with plt.style.context(style_path):
    get_ipython().run_line_magic('matplotlib', 'inline')
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(1,1, figsize=(1.5, 1.5), dpi=300)
    
    positions = [1, 2]
    flierprops = dict(marker='.', color='gray', markersize=5)
    whis = 2.0
    bp = ax.boxplot(df.loc[:,['r2_m1', 'r2_m2']],
                    positions=positions, widths=0.5,
                     flierprops=flierprops, whis=whis, patch_artist=True)
    color1 = ['gray', colors[0], colors[1]]
    for patch, color in zip(bp['boxes'], color1):
        patch.set_facecolor(color)
    # 修改中位数线的颜色
    for median in bp['medians']:
        median.set_color('k')
        median.set_linewidth(1)
    # 去掉边框线
    for item in bp['boxes']:
        item.set_linewidth(0)
    for item in bp['caps']:
        item.set_linewidth(0)
    ax.set_xticklabels(['L0','L1'])
    ax.set_ylabel('BIC')
    # ax.set_ylim([40, 100])
    
with plt.style.context(style_path):
    get_ipython().run_line_magic('matplotlib', 'inline')
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(1,1, figsize=(1.5, 1.5), dpi=300)
    
    positions = [1, 2, 3, 4]
    flierprops = dict(marker='.', color='gray', markersize=5)
    whis = 2.0
    bp = ax.boxplot(df.loc[:,['r2_m1', 'r2_m2','r2_m31', 'r2_m32']],
                    positions=positions, widths=0.5,
                     flierprops=flierprops, whis=whis, patch_artist=True)
    color1 = ['gray', colors[0], colors[1]]
    for patch, color in zip(bp['boxes'], color1):
        patch.set_facecolor(color)
    # 修改中位数线的颜色
    for median in bp['medians']:
        median.set_color('k')
        median.set_linewidth(1)
    # 去掉边框线
    for item in bp['boxes']:
        item.set_linewidth(0)
    for item in bp['caps']:
        item.set_linewidth(0)
    ax.set_xticklabels(['L0','L1','L2-1','L2-2'])
    ax.set_ylabel('BIC')
    # ax.set_ylim([40, 100])    
#%% violin plot, geo

#### plot 3个序列的m1
with plt.style.context(style_path):
    get_ipython().run_line_magic('matplotlib', 'inline')
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(1,1, figsize=(1.5, 1.5), dpi=300)
    
    df_melted = df[['ir_m1','r4_m1','r2_m1']].iloc[:50,:].melt(
            var_name='Variable', value_name='Value')
    
    # 画小提琴图
    color1 = ['gray', colors[0], colors[1]]
    ax = sns.violinplot(x='Variable', y='Value', ax=ax, data=df_melted)
    
    for patch, color in zip(ax.collections[::1], color1):  # 每个 violin 有 2 个 PolyCollection（轮廓和填充）
        patch.set_facecolor(color)
    ax.set_xlim([-1, 3])
    ax.set_ylim([30, 100])
    ax.set_xticklabels(['L0','L1','L2'])
    ax.set_xlabel('Sequence')
    ax.set_ylabel('BIC')

#### plot 3个序列的m2
with plt.style.context(style_path):
    get_ipython().run_line_magic('matplotlib', 'inline')
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(1,1, figsize=(1.5, 1.5), dpi=300)
    
    df_melted = df[['ir_bic2_delta','r4_bic2_delta','r2_bic2_delta']].iloc[:50,:].melt(
            var_name='Variable', value_name='Value')
    # df_melted = df[['ir_m2','r4_m2','r2_m2']].melt(
    #         var_name='Variable', value_name='Value')
    # 画小提琴图
    color1 = ['gray', colors[0], colors[1]]
    ax = sns.violinplot(x='Variable', y='Value', ax=ax, data=df_melted)
    
    for patch, color in zip(ax.collections[::1], color1):  # 每个 violin 有 2 个 PolyCollection（轮廓和填充）
        patch.set_facecolor(color)
    ax.set_xlim([-1, 3])
    ax.set_ylim([-190, 50]) # error, 错误label: > 1
    ax.set_xticklabels(['L0','L1','L2'])
    ax.set_xlabel('Sequence')
    ax.set_ylabel('Δ BIC')
    
#### plot 3个序列的m3
with plt.style.context(style_path):
    get_ipython().run_line_magic('matplotlib', 'inline')
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(1,1, figsize=(1.5, 1.5), dpi=300)
    
    df_melted = df[['ir_bic3_delta','r4_bic3_delta','r2_bic3_delta']].iloc[:50,:].melt(
            var_name='Variable', value_name='Value')
    
    # 画小提琴图
    color1 = ['gray', colors[0], colors[1]]
    ax = sns.violinplot(x='Variable', y='Value', ax=ax, data=df_melted)
    
    for patch, color in zip(ax.collections[::1], color1):  # 每个 violin 有 2 个 PolyCollection（轮廓和填充）
        patch.set_facecolor(color)
    ax.set_xlim([-1, 3])
    ax.set_ylim([-70, 60])
    ax.set_xticklabels(['L0','L1','L2'])
    ax.set_xlabel('Sequence')
    ax.set_ylabel('Δ BIC')
    
stats.ttest_ind(df['r2_m2'], df['r4_m2'])  
stats.ttest_ind(df['ir_bic2_delta'][:50], df['r_bic2_delta'][:50])  
t,p = stats.ttest_ind(df['ir_bic3_delta'][:50], df['r2_bic3_delta'][:50])  
p*np.sqrt(9)

# plt.hist(df['r2_bic2_delta']-df['ir_bic2_delta'])

np.sum((df['r4_bic2_delta']-df['ir_bic2_delta'])>0)
# stats.ttest_1samp(df['r2_bic2_delta'][:50], df['r4_bic2_delta'][:50].mean())

# 采用permutation test
# np.sum(df['r4_bic2_delta']<df['ir_bic2_delta'].mean())
# np.sum(df['r2_bic3_delta']<df['ir_bic3_delta'].mean())
#%% violin plot, lan/mus

#### plot 3个序列的m1
with plt.style.context(style_path):
    get_ipython().run_line_magic('matplotlib', 'inline')
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(1,1, figsize=(1.5, 1.5), dpi=300)
    
    df_melted = df[['ir_m1','r2_m1',]].melt(
            var_name='Variable', value_name='Value')
    
    # 画小提琴图
    color1 = ['gray', colors[0], colors[1]]
    ax = sns.violinplot(x='Variable', y='Value', ax=ax, data=df_melted, width=0.5)
    
    for patch, color in zip(ax.collections[::1], color1):  # 每个 violin 有 2 个 PolyCollection（轮廓和填充）
        patch.set_facecolor(color)
    ax.set_ylim([20, 90])
    ax.set_xticklabels(['L0','L1'])
    ax.set_xlabel('Sequence')
    ax.set_ylabel('BIC')

#### plot 3个序列的m2
with plt.style.context(style_path):
    get_ipython().run_line_magic('matplotlib', 'inline')
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(1,1, figsize=(1.5, 1.5), dpi=300)
    
    df_melted = df[['ir_bic2_delta','r2_bic2_delta']].melt(
            var_name='Variable', value_name='Value')
    
    # 画小提琴图
    color1 = ['gray', colors[0], colors[1]]
    ax = sns.violinplot(x='Variable', y='Value', ax=ax, data=df_melted, width=0.5)
    
    for patch, color in zip(ax.collections[::1], color1):  # 每个 violin 有 2 个 PolyCollection（轮廓和填充）
        patch.set_facecolor(color)
    ax.set_ylim([-180, 20])
    ax.set_xticklabels(['L0','L1'])
    ax.set_xlabel('Sequence')
    ax.set_ylabel('Δ BIC')
    
    
t,p = stats.ttest_ind(df['ir_m1'], df['r2_m1'])      
t,p = stats.ttest_ind(df['ir_bic2_delta'], df['r2_bic2_delta'])  
p*np.sqrt(9)
#%% plot geo-model3
with plt.style.context(style_path):
    get_ipython().run_line_magic('matplotlib', 'inline')
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(1,1, figsize=(1.5, 1.5), dpi=300)
    
    positions = [1, 2,3.5,4.5]
    flierprops = dict(marker='.', color='gray', markersize=5)
    whis = 2.0
    bp = ax.boxplot(df.loc[:,['r4_m31','r4_m32','r2_m31','r2_m32']],
                    positions=positions, widths=0.5,
                     flierprops=flierprops, whis=whis, patch_artist=True)
    color1 = [colors[0],colors[0], colors[1], colors[1]]
    for patch, color in zip(bp['boxes'], color1):
        patch.set_facecolor(color)
    # 修改中位数线的颜色
    for median in bp['medians']:
        median.set_color('k')
        median.set_linewidth(1)
    # 去掉边框线
    for item in bp['boxes']:
        item.set_linewidth(0)
    for item in bp['caps']:
        item.set_linewidth(0)
    ax.set_xticklabels(['']*4)
    ax.set_ylabel('BIC')
    ax.set_ylim([30, 70])

#%% plot lan
with plt.style.context(style_path):
    import seaborn as sns
    get_ipython().run_line_magic('matplotlib', 'inline')
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(1,1, figsize=(1.8, 1.5), dpi=300)
    
    positions = [1, 2,3.5,4.5]
    flierprops = dict(marker='.', color='gray', markersize=5)
    whis = 2.0
    bp = ax.boxplot(df.loc[:,['r2_m1','r2_m2','r2_m31','r2_m32']],
                    positions=positions, widths=0.5,
                     flierprops=flierprops, whis=whis, patch_artist=True)
    color1 = ['gray', colors[0], colors[1], colors[1]]
    for patch, color in zip(bp['boxes'], color1):
        patch.set_facecolor(color)
    # 修改中位数线的颜色
    for median in bp['medians']:
        median.set_color('k')
        median.set_linewidth(1)
    # 去掉边框线
    for item in bp['boxes']:
        item.set_linewidth(0)
    for item in bp['caps']:
        item.set_linewidth(0)
    ax.set_xticklabels(['']*4)
    ax.set_ylabel('BIC')
    # ax.set_ylim([-20, 120])
#%%  plot 3D 轨迹图, 每条灰线是单次repeat
# geo_spca = pkl.load(open(r'F:\Human_chunking_Huashan_intraoperative\Result\spca\geo_fit(time40)_correct(1).pkl','rb'))
# geo_spca = pkl.load(open(r'F:\Human_chunking_Huashan_intraoperative\Result\spca\geo_fit(time40)_correct(1_3)_PermData.pkl','rb'))
# rank3d_value = geo_spca['rank3d_value_gr2']
# fit_result = geo_spca['fit_result_c']

rank3d_value = rank3d_value_gr4
# fit_result = fit_result_a

plot_condition  = rank3d_value[:50,:,:]
# output = fit_result[-1]['output']
# pc_exp = exp_var[0].mean(0)[:3]*100

ts_per_rank = plot_condition.shape[1]//8
with plt.style.context(style_path):
    # get_ipython().run_line_magic('matplotlib', 'inline')
    get_ipython().run_line_magic('matplotlib', 'qt5')
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig = plt.figure(figsize=(2, 2), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(left=0, right=0.8, bottom=0, top=0.8)
    for i in range(plot_condition.shape[0]):
        y_mean = plot_condition[i,:,:].T
        ax.plot(plot_condition[i,:,0], plot_condition[i,:,1], plot_condition[i,:,2], 
                color='gray', linewidth=0.3, alpha=0.3)
        
    #### plot 轨迹均值
    y_mean = plot_condition[:,:,:].mean(0).T
    ax.plot(y_mean[0,:], y_mean[1,:], y_mean[2,:], color='k', linewidth=1, alpha=1)
    order = [1,2,3,4,5,6,7,8]
    k=0
    for i in (np.arange(4)*ts_per_rank+ts_per_rank//2).astype(int):
        ax.scatter(y_mean[0,i], y_mean[1,i], y_mean[2,i], s=100, color=colors[0])
        # ax.text(y_mean[0,i]-1, y_mean[1,i], y_mean[2,i]+1, order[k])
        k+=1
    for i in (np.arange(4,8)*ts_per_rank+ts_per_rank//2).astype(int):
        ax.scatter(y_mean[0,i], y_mean[1,i], y_mean[2,i], s=100, color=colors[1])
        # ax.text(y_mean[0,i], y_mean[1,i], y_mean[2,i]+1, order[k])
        k+=1
    
    # # #### 平面1
    grid_x = np.linspace(-5, 5, 10)
    grid_y = np.linspace(-5, 5, 10)
    xx, yy = np.meshgrid(grid_x, grid_y)
    
    # ### 画model1 1个平面
    # axis1, axis2 = output['m1']
    # centroid = output['m1_centroid']
    # plane_points = centroid + xx[...,None]*axis1 + yy[...,None]*axis2
    # X, Y, Z = plane_points[...,0], plane_points[...,1], plane_points[...,2]
    # ax.plot_surface(X, Y, Z, alpha=0.3, color='gray')
    
    ### 画model2 两个平面
    # axis1, axis2 = output['m2_1']
    # centroid = output['m2_1_centroid']
    # plane_points = centroid + xx[...,None]*axis1 + yy[...,None]*axis2
    # X, Y, Z = plane_points[...,0], plane_points[...,1], plane_points[...,2]
    # ax.plot_surface(X, Y, Z, alpha=0.3, color=colors[0])
    
    # axis1, axis2 = output['m2_2']
    # centroid = output['m2_2_centroid']
    # plane_points = centroid + xx[...,None]*axis1 + yy[...,None]*axis2
    # X, Y, Z = plane_points[...,0], plane_points[...,1], plane_points[...,2]
    # ax.plot_surface(X, Y, Z, alpha=0.3, color=colors[1])
    
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_zticks([])
    # ax.view_init(elev=34, azim=-37)
    # ax.view_init(elev=46, azim=-120)
    ax.view_init(elev=51, azim=-90)
    # ax.view_init(elev=45, azim=-56) # lan r4
    # ax.view_init(elev=23, azim=141) # mus ir
    # ax.view_init(elev=52, azim=145) # mus r2
    axislim = [-7, 7]
    ax.set_xlim(axislim)
    ax.set_ylim(axislim)
    ax.set_zlim(axislim)
    
    ax.tick_params(axis='x', pad=-2)  # X轴
    ax.tick_params(axis='y', pad=-2)  # Y轴
    ax.tick_params(axis='z', pad=-2)  # Z轴

    ax.set_xlabel('') # 'PC1 ('+str(pc_exp[0])[:4]+'%)', labelpad=-5
    ax.set_ylabel('') # 'PC1 ('+str(pc_exp[0])[:4]+'%)', labelpad=-5
    ax.set_zlabel('') # 'PC1 ('+str(pc_exp[0])[:4]+'%)', labelpad=-5
    
    ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.grid(False)
    # ax.set_box_aspect([1,1,1])  # 让xyz轴等比例

    # plt.tight_layout()
#%%  plot 3D 轨迹图, 每条灰线是perm trial
# geo_spca = pkl.load(open(r'F:\Human_chunking_Huashan_intraoperative\Result\spca\10sub_geo_fit(time40)_correct(1)_PermData.pkl','rb'))
# geo_spca = pkl.load(open(r'F:\Human_chunking_Huashan_intraoperative\Result\spca\10sub_lan_fit(time40)_correct(1_3)_PermData.pkl','rb'))
geo_spca = pkl.load(open(r'F:\Human_chunking_Huashan_intraoperative\Result\spca\10sub_mus_fit(time40)_correct(1_3)_PermData.pkl','rb'))

# geo_spca = pkl.load(open(r'F:\Human_chunking_Huashan_intraoperative\Result\spca\lan_fit(time40)_correct(1_3)_PermData.pkl','rb'))
# geo_spca = pkl.load(open(r'F:\Human_chunking_Huashan_intraoperative\Result\spca\mus_fit(time40)_correct(1_3)_PermData.pkl','rb'))
# fr_reduce = geo_spca['fr_reduce_ir']
# fit_result = geo_spca['fit_result_a']

# fr_reduce = geo_spca['fr_reduce_r2']
# fit_result = geo_spca['fit_result_c']

fr_reduce = fr_reduce_r4
fit_result = fit_result_b

rep = 0 # r2:15 #7; lan_r4=32, 37
pc_exp = geo_spca['exp_var'][2][rep]*100

tmp  = fr_reduce[rep][:3,:,:] # neu*trial*time
plot_condition = np.transpose(tmp, (1, 2, 0))

output = fit_result[rep]['output']
# pc_exp = exp_var[0].mean(0)[:3]*100

ts_per_rank = plot_condition.shape[1]//8
with plt.style.context(style_path):
    from mpl_toolkits.mplot3d import Axes3D, art3d
    plt.rcParams['axes.linewidth'] = 0.3   # 对二维轴生效（若你也画 2D 图）
    plt.rcParams['xtick.major.width'] = 0.3      # X轴主刻度线粗细
    plt.rcParams['ytick.major.width'] = 0.3      # Y轴主刻度线粗细
    # plt.rcParams['ztick.major.width'] = 0.2      # Y轴主刻度线粗细

    # get_ipython().run_line_magic('matplotlib', 'inline')
    get_ipython().run_line_magic('matplotlib', 'qt5')
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig = plt.figure(figsize=(2, 2), dpi=300)
    # ax = fig.add_subplot(111, projection='3d')
    ax = fig.add_axes([0.15, 0.1, 0.75, 0.8], projection='3d')

    plt.subplots_adjust(left=0, right=0.8, bottom=0, top=0.8)
    for i in range(plot_condition.shape[0]):
        plot_condition[i,:,:] = gaussian_filter1d(plot_condition[i,:,:], sigma=2, axis=0)
        yy = plot_condition[i,:,:]
        ax.plot(yy[:,0], yy[:,1], yy[:,2], 
                color='gray', linewidth=0.3, alpha=0.3)
        
    #### plot 轨迹均值
    y_mean = plot_condition[:,:,:].mean(0).T
    ax.plot(y_mean[0,:], y_mean[1,:], y_mean[2,:], color='k', linewidth=1, alpha=1)
    order = [1,2,3,4,5,6,7,8]
    k=0
    for i in (np.arange(4)*ts_per_rank+ts_per_rank//2).astype(int):
        ax.scatter(y_mean[0,i], y_mean[1,i], y_mean[2,i], s=50, color=colors[0])
        # ax.text(y_mean[0,i]-1, y_mean[1,i], y_mean[2,i]+1, order[k])
        k+=1
    for i in (np.arange(4,8)*ts_per_rank+ts_per_rank//2).astype(int):
        ax.scatter(y_mean[0,i], y_mean[1,i], y_mean[2,i], s=50, color=colors[1])
        # ax.text(y_mean[0,i], y_mean[1,i], y_mean[2,i]+1, order[k])
        k+=1
    
    #### 画平面
    grid_x = np.linspace(-4, 5, 10)
    grid_y = np.linspace(-4, 5, 10)
    xx, yy = np.meshgrid(grid_x, grid_y)
    
    # #### 画model1 1个平面
    # axis1, axis2 = output['m1']
    # centroid = output['m1_centroid']
    # plane_points = centroid + xx[...,None]*axis1 + yy[...,None]*axis2
    # X, Y, Z = plane_points[...,0], plane_points[...,1], plane_points[...,2]
    # ax.plot_surface(X, Y, Z, alpha=0.3, color='gray')
    
    ### 画model2 两个平面
    axis1, axis2 = output['m2_1']
    centroid = output['m2_1_centroid']
    plane_points = centroid + xx[...,None]*axis1 + yy[...,None]*axis2
    X, Y, Z = plane_points[...,0], plane_points[...,1], plane_points[...,2]
    ax.plot_surface(X, Y, Z, alpha=0.3, color=colors[0])
    
    axis1, axis2 = output['m2_2']
    centroid = output['m2_2_centroid']
    plane_points = centroid + xx[...,None]*axis1 + yy[...,None]*axis2
    X, Y, Z = plane_points[...,0], plane_points[...,1], plane_points[...,2]
    ax.plot_surface(X, Y, Z, alpha=0.3, color=colors[1])
    
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_zticks([])
    # ax.view_init(elev=51, azim=-110)
    # ax.view_init(elev=7, azim=-50) # lanr4, rep=32; lanir, 10
    # ax.view_init(elev=36, azim=-45) # musr4, rep=32; lanir, 10
    ax.view_init(elev=34, azim=123) # musir, 6
    
    axislim = [-8, 8]
    ax.set_xlim([-8, 8])
    ax.set_ylim(axislim)
    ax.set_zlim(axislim)
    
    ax.tick_params(axis='x', pad=-2)  # X轴
    ax.tick_params(axis='y', pad=-2)  # Y轴
    ax.tick_params(axis='z', pad=-2)  # Z轴

    # ax.set_xlabel('PC1') # 'PC1 ('+str(pc_exp[0])[:4]+'%)', labelpad=-5
    # ax.set_ylabel('PC2') # 'PC1 ('+str(pc_exp[0])[:4]+'%)', labelpad=-5
    # ax.set_zlabel('PC3') # 'PC1 ('+str(pc_exp[0])[:4]+'%)', labelpad=-5
    
    # ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    # ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    # ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    xlim, ylim, zlim = ax.get_xlim(), ax.get_ylim(), ax.get_zlim()

    # 画 12 条棱线
    for s, e in [
        [(xlim[0], ylim[0], zlim[0]), (xlim[1], ylim[0], zlim[0])],
        [(xlim[0], ylim[1], zlim[0]), (xlim[1], ylim[1], zlim[0])],
        [(xlim[0], ylim[0], zlim[1]), (xlim[1], ylim[0], zlim[1])], # geo
        # [(xlim[0], ylim[1], zlim[1]), (xlim[1], ylim[1], zlim[1])],
        [(xlim[0], ylim[0], zlim[0]), (xlim[0], ylim[1], zlim[0])],
        [(xlim[1], ylim[0], zlim[0]), (xlim[1], ylim[1], zlim[0])],
        # [(xlim[0], ylim[0], zlim[1]), (xlim[0], ylim[1], zlim[1])], # geo
        [(xlim[1], ylim[0], zlim[1]), (xlim[1], ylim[1], zlim[1])],
        [(xlim[0], ylim[0], zlim[0]), (xlim[0], ylim[0], zlim[1])], # geo
        [(xlim[1], ylim[0], zlim[0]), (xlim[1], ylim[0], zlim[1])],
        # [(xlim[0], ylim[1], zlim[0]), (xlim[0], ylim[1], zlim[1])],
        [(xlim[1], ylim[1], zlim[0]), (xlim[1], ylim[1], zlim[1])]
    ]:
        line = art3d.Line3D(*zip(s, e), color='k', linewidth=0.3)
        ax.add_line(line)
    
    ax.grid(False)
    # ax.set_box_aspect([1,1,1])  # 让xyz轴等比例

    # plt.tight_layout()
#%% plot 3个模型的示意图
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import splprep, splev
from sklearn.decomposition import PCA


# 数据
n = 40
half = n // 2

# 时间参数 t0从0到1
t0 = np.linspace(0, 1, n)

# 设计平滑过渡函数 alpha，从0到1，控制平面切换过程
alpha = np.clip(10 * (t0 - 0.4), 0, 1)

# 前半段近似x-y平面曲线 (z小，绕z轴转动)
x1 = 3*np.cos(1 * np.pi * t0) * (1 - alpha)
y1 = 3*np.sin(1 * np.pi * t0) * (1 - alpha)
z1 = 0.1 * (1 - alpha)

# 后半段近似y-z平面曲线 (x小，绕x轴转动)
x2 = 0.1 * alpha
y2 = 3*np.cos(1 * np.pi * t0) * alpha
z2 = 3*np.sin(1 * np.pi * t0) * alpha

# 叠加得到整体轨迹
x = x1 + x2
y = y1 + y2
z = z1 + z2

x_data = np.stack([x,y,z]).T
x_data = x_data[::-1,:]
# 平滑曲线拟合
tck, u = splprep(x_data.T, s=2.0)
u_fine = np.linspace(0, 1, 300)
x_smooth = np.array(splev(u_fine, tck)).T

# 拟合前后两个平面
def fit_plane(points):
    pca = PCA(n_components=2)
    pca.fit(points)
    center = np.mean(points, axis=0)
    normal = np.cross(pca.components_[0], pca.components_[1])
    return center, normal

center1, normal1 = fit_plane(x_data[:20])
center2, normal2 = fit_plane(x_data[20:])

def draw_plane(ax, center, normal, size=5, color='gray', alpha=0.3):
    xx, yy = np.meshgrid(np.linspace(-size, size, 10), np.linspace(-size, size, 10))
    normal = normal / np.linalg.norm(normal)
    v1 = np.cross(normal, [1,0,0])
    if np.linalg.norm(v1) < 1e-2:
        v1 = np.cross(normal, [0,1,0])
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.cross(normal, v1)
    v2 = v2 / np.linalg.norm(v2)
    plane = center[:, None, None] + v1[:, None, None]*xx + v2[:, None, None]*yy
    # ax.plot_surface(plane[0], plane[1], plane[2], alpha=alpha, color=color, edgecolor='none', linewidth=0)
    return plane
    
# 可视化
with plt.style.context(style_path):
    get_ipython().run_line_magic('matplotlib', 'inline')
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig = plt.figure(figsize=(3, 3), dpi=300,facecolor='none')
    ax = fig.add_subplot(111, projection='3d')
    
    # 画平滑曲线
    ax.plot(x_smooth[:,0], x_smooth[:,1], x_smooth[:,2], color=colors[0], lw=2, label='Trajectory')
    # 原始点标注
    seg1 = [4,9,14,19]
    seg2 = [24,29,34,39]
    ax.scatter(x_data[seg1,0], x_data[seg1,1], x_data[seg1,2], color=colors[0], s=100, alpha=1)
    ax.scatter(x_data[seg2,0], x_data[seg2,1], x_data[seg2,2], color=colors[1], s=100, alpha=1)
    
    # 平面绘制
    p3d = draw_plane(ax, center1, normal1, size=2, color=colors[0], alpha=0.5)
    ax.plot_surface(p3d[0,:,:], p3d[1,:,:], p3d[2,:,:], color=colors[0], alpha=0.3, edgecolor='none', linewidth=0)
    
    p3d = draw_plane(ax, center2, normal2, size=2, color=colors[1], alpha=0.5)
    ax.plot_surface(p3d[0,:,:], p3d[1,:,:], p3d[2,:,:], color=colors[1], alpha=0.3, edgecolor='none', linewidth=0)
   
    # 画坐标轴
    # ax.quiver(0,0,0, 2,0,0, color='k', arrow_length_ratio=0.1)
    # ax.quiver(0,0,0, 0,2,0, color='k', arrow_length_ratio=0.1)
    # ax.quiver(0,0,0, 0,0,2, color='k', arrow_length_ratio=0.1)

    # 轴设置
    ax.set_xlabel('X', labelpad=10)
    ax.set_ylabel('Y', labelpad=10)
    ax.set_zlabel('Z', labelpad=10)
    # ax.set_title('Illustrative 3D Trajectory with Two Orthogonal Planes', fontsize=14)
    # ax.legend(loc='upper right')
    ax.view_init(elev=30, azim=-50)
    
    # 去掉坐标轴和刻度
    ax.set_axis_off()
    
    # 或者你也可以分别隐藏网格和坐标轴：
    # 去掉网格线
    ax.grid(False)
    
    # 去掉坐标轴刻度和标签
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    
    # plt.savefig(r'C:\Wen\OneDrive\Project\Human_sqchunk_intraoperative\Manuscript\Figures\Figure3\surface_no_grid.png',
    #             transparent=True, dpi=300)
    plt.show()
    
#%% 拟合、比较3个模型
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import acos, degrees

# ------------------- 工具函数 -------------------
def angle_between(v1, v2):
    cos_theta = np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0)
    return degrees(acos(cos_theta))

def fit_plane(points):
    centroid = points.mean(axis=0)
    centroid[:] = 0
    _, _, vh = np.linalg.svd(points - centroid)
    _, _, vh = np.linalg.svd(points)
    normal = vh[-1]
    distances = np.abs((points - centroid) @ normal)
    return normal, centroid, distances.sum()

def plot_plane(ax, normal, point_on_plane, color, alpha=0.3):
    d = -point_on_plane.dot(normal)
    xx, yy = np.meshgrid(np.linspace(-1, 8, 10), np.linspace(-1, 4, 10))
    zz = (-normal[0]*xx - normal[1]*yy - d) / normal[2]
    ax.plot_surface(xx, yy, zz, color=color, alpha=alpha)

# ------------------- 模型评分 -------------------
def model_1(points):
    normal, centroid, residual = fit_plane(points)
    return residual, [(normal, centroid)]

def model_2(points):
    normal1, c1, r1 = fit_plane(points[:4])
    normal2, c2, r2 = fit_plane(points[4:])
    inter_angle = angle_between(normal1, normal2)
    # penalty = -inter_angle
    residual = np.abs(inter_angle)
    return residual, [(normal1, c1), (normal2, c2)]

def model_3(points):
    v1 = points[1] - points[0]
    v2 = points[3] - points[2]
    v3 = points[5] - points[4]
    v4 = points[7] - points[6]
    angle_group1 = angle_between(v1, v2)
    angle_group2 = angle_between(v3, v4)
    avg1 = (v1 + v2) / 2
    avg2 = (v3 + v4) / 2
    group_diff = angle_between(avg1, avg2)
    score = angle_group1 + angle_group2 - group_diff
    return score, [(points[0], points[1]), (points[2], points[3]),
                   (points[4], points[5]), (points[6], points[7])]

# ------------------- 主程序 -------------------
def compare_models(points):
    score1, plane1 = model_1(points)
    score2, planes2 = model_2(points)
    score3, dirs3 = model_3(points)

    scores = [score1, score2, score3]
    best = int(np.argmin(scores)) + 1

    print(f"Model 1 score: {score1:.4f}")
    print(f"Model 2 score: {score2:.4f}")
    print(f"Model 3 score: {score3:.4f}")
    print(f"✅ Best model: Model {best}")

    # 绘图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(points[:, 0], points[:, 1], points[:, 2], 'ko-', label='Points')

    if best == 1:
        for normal, c in plane1:
            plot_plane(ax, normal, c, 'skyblue')
        ax.set_title("Best: Model 1 (Single Plane)")

    elif best == 2:
        colors = ['lightgreen', 'lightcoral']
        for (normal, c), color in zip(planes2, colors):
            plot_plane(ax, normal, c, color)
        ax.set_title("Best: Model 2 (Two Planes)")

    elif best == 3:
        colors = ['red', 'red', 'blue', 'blue']
        for (a, b), color in zip(dirs3, colors):
            ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color=color, linewidth=3)
        ax.set_title("Best: Model 3 (Grouped Directions)")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.legend()
    plt.tight_layout()
    plt.show()


dis_perm = np.zeros([repetition, 1])
for repeat in range(repetition):
    points = rank3d_value[repeat,:,:]
    compare_models(points)
    
# plt.close('all')
#%% 计算曲率和绕率
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from scipy.interpolate import splprep, splev
from scipy.signal import savgol_filter

# === 数据点序列 ===
x = rank3d_value[0,:,:]
# tt = sliding_window_view(x.T, window_shape=3, axis=1)
# x = tt[:,::2].mean(axis=-1).T

# 构造样条曲线
# tck, u = splprep(x.T, s=0)
# u_fine = np.linspace(0, 1, 160)  # 增加点数
# x_fine = np.array(splev(u_fine, tck)).T
# x = x_fine

# x = savgol_filter(x, window_length=10, polyorder=3, axis=0)
# === Frenet–Serret 分析函数 ===
def compute_frenet_frame(x):
    dx = np.gradient(x, axis=0)
    ddx = np.gradient(dx, axis=0)
    dddx = np.gradient(ddx, axis=0)

    T = dx / np.linalg.norm(dx, axis=1, keepdims=True)
    dT = np.gradient(T, axis=0)
    N = dT / np.linalg.norm(dT, axis=1, keepdims=True)
    B = np.cross(T, N)

    curvature = np.linalg.norm(np.cross(dx, ddx), axis=1) / (np.linalg.norm(dx, axis=1)**3 + 1e-8)
    torsion = np.einsum('ij,ij->i', np.cross(dx, ddx), dddx) / (np.linalg.norm(np.cross(dx, ddx), axis=1)**2 + 1e-8)

    return curvature, torsion, T, N, B

# === 法向量夹角计算函数 ===
def angle_between(v1, v2):
    cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return np.arccos(cos_angle) * 180 / np.pi

# === 1. Frenet–Serret 分析 ===
curvature, torsion, T, N, B = compute_frenet_frame(x)

# === 2. 局部滑动 PCA 主平面变化 ===
window_size = 5
normals = []
for i in range(len(x) - window_size + 1):
    pca = PCA(n_components=3)
    pca.fit(x[i:i + window_size])
    normals.append(pca.components_[-1])
normals = np.array(normals)
angles = [angle_between(normals[i], normals[i + 1]) for i in range(len(normals) - 1)]

# === 3. 自动检测拐点位置 ===
angles_arr = np.array(angles)
turning_point_index = np.argmax(angles_arr) + window_size  # 滑窗补偿

# === 4. 可视化：三维轨迹 + 拐点 ===
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x[:, 0], x[:, 1], x[:, 2], '-o', label='Trajectory', markersize=3)
ax.scatter(*x[turning_point_index], color='red', s=80,
           label=f'Turning Point (index={turning_point_index})')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Curve with Detected Turning Point')
ax.legend()
plt.tight_layout()
plt.show()

# === 5. 可视化：曲率与挠率 ===
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
ax1.plot(curvature, label='Curvature', color='blue')
ax1.set_ylabel('Curvature')
ax1.legend()
ax2.plot(torsion, label='Torsion', color='green')
ax2.set_ylabel('Torsion')
ax2.set_xlabel('Point Index')
ax2.legend()
plt.suptitle('Curvature and Torsion of the Curve')
plt.tight_layout()
plt.show()

# === 6. 可视化：滑窗 PCA 主平面变化 ===
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(angles, label='Angle between local plane normals')
ax.axhline(90, linestyle='--', color='gray')
ax.set_ylabel('Angle (degrees)')
ax.set_xlabel('Window Index')
ax.set_title('Local Plane Normal Angle Change (Sliding Window PCA)')
ax.legend()
plt.tight_layout()
plt.show()

# === 输出拐点索引及坐标 ===
print(f"Detected turning point index: {turning_point_index}")
print("Coordinates:", x[turning_point_index])
#%% 计算脱离平面位置
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def analyze_3d_curve(curve, front_ratio=0.5, back_ratio=0.5, deviation_threshold=1e-2):
    """
    参数：
        curve: Nx3 array，表示曲线的N个三维点
        front_ratio: 用于拟合前半段平面的比例
        back_ratio: 用于拟合后半段平面的比例
        deviation_threshold: 脱离平面的最小距离判断阈值
    返回：
        一个包含三个量化指标的字典
    """
    curve = np.asarray(curve)
    N = len(curve)
    x0 = curve[0]
    x_end = curve[-1]

    # --- 1. 绕回程度 ---
    d_end = np.linalg.norm(x_end - x0)
    d_max = np.max(np.linalg.norm(curve - x0, axis=1))
    return_ratio = d_end / d_max

    # --- 2. 拟合前段平面 ---
    front_N = int(front_ratio * N)
    front_points = curve[:front_N]
    pca_front = PCA(n_components=3).fit(front_points)
    n1 = pca_front.components_[-1]  # 法向量
    mean1 = pca_front.mean_

    # 点到平面距离
    distances = np.abs((curve - mean1) @ n1)
    max_offset = np.max(distances)

    # 找出开始脱离平面的点索引
    baseline = np.mean(distances[:front_N]) + 2 * np.std(distances[:front_N])
    i_deviation = np.argmax(distances > max(baseline, deviation_threshold))

    # --- 3. 拟合后段平面 ---
    back_N = int(back_ratio * N)
    back_points = curve[-back_N:]
    pca_back = PCA(n_components=3).fit(back_points)
    n2 = pca_back.components_[-1]

    # 平面法向量夹角
    cos_theta = np.abs(np.dot(n1, n2)) / (np.linalg.norm(n1) * np.linalg.norm(n2))
    plane_angle_deg = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

    # --- 返回指标 ---
    return {
        "绕回程度(Return Ratio)": return_ratio,
        "最大脱离距离(Max Deviation from Plane)": max_offset,
        "脱离平面的位置索引(Index of Departure)": i_deviation,
        "平面夹角(Plane Angle in Degrees)": plane_angle_deg,
    }

# --- 示例：生成一个具有绕回和转折的三维曲线 ---
def generate_example_curve(n=200):
    t = np.linspace(0, 2 * np.pi, n)
    x = np.cos(t)
    y = np.sin(t)
    z = np.piecewise(t, [t < np.pi, t >= np.pi],
                     [lambda t: 0.1 * t, lambda t: 2 - 0.1 * t])
    # 让后半段偏离xy平面
    z[t > np.pi] += 1.5 * np.sin(t[t > np.pi])  
    return np.vstack([x, y, z]).T

# --- 运行分析 ---
curve = rank3d_value[0,:,:]
metrics = analyze_3d_curve(curve)
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

# 可视化
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(*curve.T, color='blue')
ax.scatter(*curve[0], color='green', label='Start')
ax.scatter(*curve[-1], color='red', label='End')
ax.scatter(*curve[metrics["脱离平面的位置索引(Index of Departure)"]], color='orange', label='Deviation Point')
ax.legend()
plt.title("3D Curve Analysis")
plt.show()

#%% plot3d, 8 ranks和最佳拟合平面
from scipy.ndimage import gaussian_filter1d
get_ipython().run_line_magic('matplotlib', 'qt5')

y_mean = rank3d_value_gr4[0,:,:].T
# for i in range(3):
#     y_mean[i,:] = gaussian_filter1d(y_mean[i,:], 1)

with plt.style.context(style_path):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(y_mean[0,:], y_mean[1,:], y_mean[2,:], color='k', alpha=0.5)
    for i in np.arange(4)*5+4:
        ax.scatter(y_mean[0,i], y_mean[1,i], y_mean[2,i], s=100, color=colors[0])
        ax.text(y_mean[0,i], y_mean[1,i], y_mean[2,i], str(i+1))
    for i in np.arange(4,8)*5+4:
        ax.scatter(y_mean[0,i], y_mean[1,i], y_mean[2,i], s=100, color=colors[1])
        ax.text(y_mean[0,i], y_mean[1,i], y_mean[2,i], str(i+1))
        
    # 生成网格用于绘制平面
    x_vals = np.linspace(min(pp[:, 0]), max(pp[:, 0]), 10)
    y_vals = np.linspace(min(pp[:, 1]), max(pp[:, 1]), 10)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    # #### 平面1
    a,b,d = plane1
    Z = a * X + b * Y + d
    # 绘制拟合平面
    ax.plot_surface(X, Y, Z, color=colors[0], alpha=0.3)
    
    #### 平面2
    a,b,d = plane2
    Z = a * X + b * Y + d
    # 绘制拟合平面
    ax.plot_surface(X, Y, Z, color=colors[1], alpha=0.3)
    
    #### 平面12
    # a,b,d = plane12
    # Z = a * X + b * Y + d
    # # 绘制拟合平面
    # ax.plot_surface(X, Y, Z, color=colors[2], alpha=0.3)
    
    # 轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Best Fit Plane for chunk1 and chunk2')
    ax.legend()
    # plt.savefig(r'F:\Human_chunking_Huashan_intraoperative\Result\allgeo_gr2_3d.png')
#%% plot distance index 
import seaborn as sns

file = r'F:\Human_chunking_Huashan_intraoperative\Result\spca\dist2plane.xlsx'
dist = pd.read_excel(file)
tmp = [remove_outlier(dist.iloc[:,i]).values for i in range(dist.shape[1])]

df = pd.DataFrame({'idx':np.arange(dist.shape[0])})
df['geo_ir'] = np.nan
df['geo_r4_c'] = np.nan
df['geo_r4_e'] = np.nan
df['geo_r2_c'] = np.nan
df['geo_r2_e'] = np.nan

df.loc[:len(tmp[0])-1, 'geo_ir'] = tmp[0]
df.loc[:len(tmp[1])-1, 'geo_r4_c'] = tmp[1]
df.loc[:len(tmp[2])-1, 'geo_r4_e'] = tmp[2]
df.loc[:len(tmp[3])-1, 'geo_r2_c'] = tmp[3]
df.loc[:len(tmp[4])-1, 'geo_r2_e'] = tmp[4]

stats.ttest_ind(tmp[3], tmp[4])

with plt.style.context(style_path):
    get_ipython().run_line_magic('matplotlib', 'inline')
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(1,1, figsize=(2.5, 2), dpi=300)
    
    # df_melted = df[['geo_ir','geo_r4_c','geo_r4_e']].melt(var_name='Variable', value_name='Value')
    df_melted = df[['geo_ir','geo_r2_c','geo_r2_e']].melt(var_name='Variable', value_name='Value')

    # 画小提琴图
    sns.violinplot(x='Variable', y='Value', ax=ax, data=df_melted, color=colors[1])
    
    # 设置 x 轴标签
    # ax.set_xticks([1, 2, 3], dist.columns)
    # ax.set_ylabel("Value")
#%% plot 3D
# get_ipython().run_line_magic('matplotlib', 'qt5')
# y = x_reduce3d
# with plt.style.context(style_path):
#     colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
    
#     y_mean = y[:, :500, :].mean(1)
#     ax.plot(y_mean[0,:], y_mean[1,:], y_mean[2,:], color=colors[0])
#     for i in range(8):
#         ax.scatter(y_mean[0,i], y_mean[1,i], y_mean[2,i], s=100, color=colors[i])
        
#     # y_mean = y[:, 500:, :].mean(1)
#     # ax.plot(y_mean[0,:], y_mean[1,:], y_mean[2,:], color=colors[1])
#     # for i in range(8):
#     #     ax.scatter(y_mean[0,i], y_mean[1,i], y_mean[2,i], s=100, color=colors[i])

#%% plot 8 ranks 在2D图上的分布
# y = x_reduce3d
# y_mean = y[:, :500, :].mean(1)

# pp = y_mean[:3,:].T

# get_ipython().run_line_magic('matplotlib', 'inline')
# with plt.style.context(style_path):
#     colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
#     fig, ax = plt.subplots(1,1, figsize=(2.5, 2), dpi=300) 
#     # ax.plot(x[0,10:50,:].T)
    
#     ax.plot(y_mean[0,:], y_mean[1,:], color='k')
#     for i in range(4):
#         ax.scatter(y_mean[0,i], y_mean[1,i], s=100, color=colors[0])
#         ax.text(y_mean[0,i], y_mean[1,i], str(i+1))
#     for i in range(4,8):
#         ax.scatter(y_mean[0,i], y_mean[1,i], s=100, color=colors[1])
#         ax.text(y_mean[0,i], y_mean[1,i], str(i+1))
    
    # plt.savefig(r'F:\Human_chunking_Huashan_intraoperative\Result\allgeo_gir.png')  # 支持多种格式：.png, .jpg, .pdf, .svg 等
#%% 在spca降维后3维空间中的feature weight
fea_imp_mean = np.zeros([fea_imp_ir.shape[0], 3])
fea_imp_mean[:,0] = fea_imp_ir.mean(1)
fea_imp_mean[:,1] = fea_imp_r4.mean(1)
fea_imp_mean[:,2] = fea_imp_r2.mean(1)

with plt.style.context(style_path):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(1,1, figsize=(2.5, 2), dpi=300) 
    
    ax.scatter(fea_imp_mean[:,0], fea_imp_mean[:,2])
    
stats.spearmanr(fea_imp_mean[:,0], fea_imp_mean[:,1])
stats.spearmanr(fea_imp_mean[:,0], fea_imp_mean[:,2])
stats.spearmanr(fea_imp_mean[:,1], fea_imp_mean[:,2])

#%% 计算8个rank间的欧氏距离
from scipy.spatial.distance import cdist

geo_spca = pkl.load(open(r'F:\Human_chunking_Huashan_intraoperative\Result\spca\10sub_geo_fit(time40)_correct(1)_PermData.pkl','rb'))
rank3d_value_gir = geo_spca['rank3d_value_gir']
rank3d_value_gr4 = geo_spca['rank3d_value_gr4']
rank3d_value_gr2 = geo_spca['rank3d_value_gr2']

rep_num = rank3d_value_gr4.shape[0]

dis_ir = np.zeros([rep_num, 8,8])
dis_r4 = np.zeros([rep_num, 8,8])
dis_r2 = np.zeros([rep_num, 8,8])

for i in range(rep_num):
    curve_a = rank3d_value_gir[i,:,:]
    curve_b = rank3d_value_gr4[i,:,:]
    curve_c = rank3d_value_gr2[i,:,:]
    
    curve_a = curve_a.reshape(8,5,3)[:,:2,:].mean(1)
    dis_ir[i,:,:] = cdist(curve_a, curve_a)   # 返回 (8,8) 的欧氏距离矩阵
    
    curve_b = curve_b.reshape(8,5,3)[:,:2,:].mean(1)
    dis_r4[i,:,:] = cdist(curve_b, curve_b)   # 返回 (8,8) 的欧氏距离矩阵

    curve_c = curve_c.reshape(8,5,3)[:,:2,:].mean(1)
    dis_r2[i,:,:] = cdist(curve_c, curve_c)   # 返回 (8,8) 的欧氏距离矩阵
#%%
with plt.style.context(style_path):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][0:]
    fig, axs = plt.subplots(1, 1, figsize=(1.6, 1.4), dpi=300) 
    # h1 = axs.pcolormesh(dis_ir.mean(0), vmin=0, vmax=10) # , vmin=.15, vmax=0.35 , cmap = plt.cm.RdBu_r
    h1 = axs.pcolormesh(dis_r4.mean(0), vmin=1.5, vmax=11) # , vmin=.15, vmax=0.35 , cmap = plt.cm.RdBu_r
    # h1 = axs.pcolormesh(dis_r2.mean(0), vmin=0.5, vmax=14) # , vmin=.15, vmax=0.35 , cmap = plt.cm.RdBu_r
    axs.invert_yaxis()
    axs.set_xticks(np.arange(8)+0.5, np.arange(1,9))
    axs.set_yticks(np.arange(8)+0.5, np.arange(1,9))
    axs.set_xlabel('Prediction')
    axs.set_ylabel('Data')
    fig.colorbar(h1)    
    
#%%
dis_ir[:,np.arange(8), np.arange(8)] = np.nan
cm_ir_within = (np.nanmean(dis_ir[:,:4,:4], -1).mean(-1) + np.nanmean(dis_ir[:,4:,4:], -1).mean(-1))/2
cm_ir_between = (dis_ir[:,:4,4:].mean(-1).mean(-1) + dis_ir[:,4:,:4].mean(-1).mean(-1))/2

dis_r4[:,np.arange(8), np.arange(8)] = np.nan
cm_r4_within = (np.nanmean(dis_r4[:,:4,:4], -1).mean(-1) + np.nanmean(dis_r4[:,4:,4:], -1).mean(-1))/2
cm_r4_between = (dis_r4[:,:4,4:].mean(-1).mean(-1) + dis_r4[:,4:,:4].mean(-1).mean(-1))/2

dis_r2[:,np.arange(8), np.arange(8)] = np.nan
cm_r2_within = (np.nanmean(dis_r2[:,:4,:4], -1).mean(-1) + np.nanmean(dis_r2[:,4:,4:], -1).mean(-1))/2
cm_r2_between = (dis_r2[:,:4,4:].mean(-1).mean(-1) + dis_r2[:,4:,:4].mean(-1).mean(-1))/2


df = pd.DataFrame({'cm_ir_within':cm_ir_within,'cm_ir_between':cm_ir_between,
                   'cm_r4_within':cm_r4_within, 'cm_r4_between':cm_r4_between,
                   'cm_r2_within':cm_r2_within, 'cm_r2_between':cm_r2_between,})

df['wb_ir'] = df['cm_ir_between']/df['cm_ir_within']
df['wb_r4'] = df['cm_r4_between']/df['cm_r4_within']
df['wb_r2'] = df['cm_r2_between']/df['cm_r2_within']

with plt.style.context(style_path):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][0:]
    fig, axs = plt.subplots(1, 1, figsize=(1.2, 1.2), dpi=300) 
    
    color1 = ['gray', colors[0], colors[1]]
    # df_melted = df[['wb_ir', 'wb_r4','wb_r2']].melt(var_name='Variable', value_name='Value')
    df_melted = df[['cm_ir_between', 'cm_r4_between','cm_r2_between']].melt(var_name='Variable', value_name='Value')
    sns.violinplot(x='Variable', y='Value', ax=axs, data=df_melted, color = 'gray',width=0.5)
    for patch, color in zip(axs.collections[::1], color1):  # 每个 violin 有 2 个 PolyCollection（轮廓和填充）
        patch.set_facecolor(color)
    axs.set_ylim([3, 10])
    axs.set_xticklabels([' ', ' '])    
    axs.set_xlabel(None)    
    axs.set_ylabel('Cross distance')    
 
df.mean()
# stats.ttest_ind(df['wb_ir'][:50], df['wb_r2'][:50])
stats.ttest_ind(df['cm_ir_between'][:50], df['cm_r2_between'][:50])
# stats.ttest_1samp(df['wb_ir'][:50], df['wb_r2'].mean())
# cm_ir_between.mean()

####
dis_r4 = dis_r4[:,4:,4:]
dis_r2 = dis_r2[:,4:,4:]

bb = 2
cm_r4_within = (np.nanmean(dis_r4[:,:bb,:bb], -1).mean(-1) + np.nanmean(dis_r4[:,bb:,bb:], -1).mean(-1))/2
cm_r4_between = (dis_r4[:,:bb,bb:].mean(-1).mean(-1) + dis_r4[:,bb:,:bb].mean(-1).mean(-1))/2

cm_r2_within = (np.nanmean(dis_r2[:,:bb,:bb], -1).mean(-1) + np.nanmean(dis_r2[:,bb:,bb:], -1).mean(-1))/2
cm_r2_between = (dis_r2[:,:bb,bb:].mean(-1).mean(-1) + dis_r2[:,bb:,:bb].mean(-1).mean(-1))/2

df = pd.DataFrame({'cm_r4_within':cm_r4_within, 'cm_r4_between':cm_r4_between,
                   'cm_r2_within':cm_r2_within, 'cm_r2_between':cm_r2_between,})

df['wb_r4'] = df['cm_r4_between']/(df['cm_r4_within']+df['cm_r4_between'])
df['wb_r2'] = df['cm_r2_between']/(df['cm_r2_within']+df['cm_r2_between'])

with plt.style.context(style_path):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axs = plt.subplots(1, 1, figsize=(1.2, 1.2), dpi=300) 
    
    color1 = [colors[0], colors[1]]
    df_melted = df[['cm_r4_between', 'cm_r2_between']].melt(var_name='Variable', value_name='Value')
    # df_melted = df[['wb_r4','wb_r2']].melt(var_name='Variable', value_name='Value')
    sns.violinplot(x='Variable', y='Value', ax=axs, data=df_melted,width=0.5)
    for patch, color in zip(axs.collections[::1], color1):  # 每个 violin 有 2 个 PolyCollection（轮廓和填充）
        patch.set_facecolor(color)
    
    axs.set_ylim([1, 12])
    axs.set_xlim([-0.8, 1.8])
    axs.set_xticklabels([' ', ' '])    
    axs.set_xlabel(None)    
    axs.set_ylabel('Cross distance')    
    
# stats.ttest_ind(df['wb_r4'], df['wb_r2'])
stats.ttest_ind(df['cm_r4_between'], df['cm_r2_between'])

    



    
