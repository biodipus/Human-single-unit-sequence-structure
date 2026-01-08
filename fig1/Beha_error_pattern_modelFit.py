# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 14:28:22 2025

@author: Wen
"""
get_ipython().run_line_magic('reset', '-sf')
style_path = r'C:\Wen\OneDrive\CodeHub\Python\style_paper.mplstyle'
get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import pandas as pd
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl
import scipy.io as sio
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest
#%% fun
def addtrialidx(y):
    y['trial'] = 0
    if 'name' in y.columns:
        sl_indices = y[y['name'] == 'sl'].index
    else:
        sl_indices = y[y['rank'] == -1].index
    trial_num = 1
    start_idx = 0
    for end_idx in sl_indices:
        y.loc[start_idx:end_idx-1, 'trial'] = trial_num
        trial_num += 1
        start_idx = end_idx + 1
    # 处理最后一个 'sl' 之后的行
    y.loc[start_idx:, 'trial'] = trial_num
    return y

def addtrialidx_mus(y):
    y['trial'] = 0
    if len(y[y['name'] == -999].index)>0:
        sl_indices = y[y['name'] == -999].index
    else:    
        sl_indices = y[y['target'] == -999].index
    trial_num = 1
    start_idx = 0
    for end_idx in sl_indices:
        y.loc[start_idx:end_idx-1, 'trial'] = trial_num
        trial_num += 1
        start_idx = end_idx + 1
    # 处理最后一个 'sl' 之后的行
    y.loc[start_idx:, 'trial'] = trial_num
    return y

def get_acc_lan(y):
    acc = []
    for trial in range(y['trial'].max()):
        target = y.loc[y['trial']==trial+1, 'rank'].values # 语言文件中name,rank是刺激，target是被试的回答
        response = y.loc[y['trial']==trial+1, 'target'].values.astype(float)
        x = np.zeros([8, 8])
        for i in range(8):
            idx = response[i] == target
            x[i,idx] = 1 
        acc.append(x)
    acc = np.stack(acc, axis=2)
    return acc

def get_resp_interval(x):
    dur_chunk = np.zeros([len(x), 2])
    interval = np.zeros([len(x), 7])
    
    for i in range(len(x)):
        resp = x[i]
        dur_chunk[i,0] = resp.loc[3,'end'] - resp.loc[0,'start']
        dur_chunk[i,1]= resp.loc[7,'end'] - resp.loc[4,'start']
        interval[i,:] = resp.loc[:,'start'].values[1:] - resp.loc[:,'end'].values[0:-1]
    return dur_chunk, interval

# 计算正确率矩阵
def geo_acc_matrix(y):
    acc = []
    trialNum = y['trial'].max()
    for trial in range(trialNum):
        target = y.loc[y['trial']==trial+1,'target'].values
        response = y.loc[y['trial']==trial+1,'name'].values.astype(float)
        response[response<0] = np.nan
        x = np.zeros([8, 8])
        for i in range(8):
            idx = response[i] == target
            if sum(idx)>0:
                x[i,idx] = 1 
        acc.append(x)
    acc = np.stack(acc, axis=2)
    # acc_alltask['geo_ir'] = acc # rank*reponse rank*trial
    return acc

def lan_acc_matrix(y):
    acc = []
    trialNum = y['trial'].max()
    for trial in range(trialNum):
        target = y.loc[y['trial']==trial+1,'target'].values
        response = y.loc[y['trial']==trial+1,'report'].values.astype(float)
        response[response<0] = np.nan
        x = np.zeros([8, 8])
        for i in range(8):
            idx = response[i] == target
            if sum(idx)>0:
                x[i,idx] = 1 
        acc.append(x)
    acc = np.stack(acc, axis=2)
    # acc_alltask['geo_ir'] = acc # rank*reponse rank*trial
    return acc

# def mus_acc_matrix(y):
#     acc = []
#     trialNum = y['trial'].max()

def mus_acc_matrix(y):
    acc = []
    trialNum = y['trial'].max()
    for trial in range(trialNum):
        err = y.loc[y['trial']==trial+1,'error'].values
        x = np.zeros([8, 8])
        idx = np.where(err==0)[0]
        x[idx,idx] = 1
        acc.append(x)
    acc = np.stack(acc, axis=2)
    # acc_alltask['geo_ir'] = acc # rank*reponse rank*trial
    return acc    
#%% path & subs
path = r'F:\Human_chunking_Huashan_intraoperative\Data\Recording\PFC'
# file = path + '\\fr_beha_PFC(9sub_geo_254).pkl'
# file = path + '\\fr_beha_PFC(9sub_lan_250).pkl'
# file = path + '\\fr_beha_PFC(9sub_mus_231).pkl'
# file = path + '\\fr_beha_PFC(9sub_153).pkl'
# file = path + '\\fr_beha_PFC(9sub_155)_drift.pkl'
file = path + '\\fr_beha_PFC(10sub_160)_drift_new2.pkl'

# path = r'F:\Human_chunking_Huashan_intraoperative\Data\Recording\motor'
# file = path + '\\fr_beha_motor(2sub_67)_drift.pkl'

data = pkl.load(open(file, 'rb'))
beha = data['beha']
subs = data['subs']
# trialinfo_all = pd.concat(data['trialInfo'])
#%% 增加trial index
for s in range(len(subs)):
    tmp = beha[s]
    for x in tmp:
        if "mus" in x:
            tmp[x] = addtrialidx_mus(tmp[x])
        else:
            tmp[x] = addtrialidx(tmp[x])
    beha[s] = tmp
#%% geo/lan 正确matrix
accmatrix_geo = [[],[],[]]
for i in range(len(subs)):
    accmatrix_geo[0].append(geo_acc_matrix(beha[i]['geo_ir']))
    accmatrix_geo[1].append(geo_acc_matrix(beha[i]['geo_r4']))
    accmatrix_geo[2].append(geo_acc_matrix(beha[i]['geo_r2']))
#### lan正确matrix
accmatrix_lan = [[],[]]
for i in range(len(subs)):
    accmatrix_lan[0].append(lan_acc_matrix(beha[i]['lan_ir']))
    accmatrix_lan[1].append(lan_acc_matrix(beha[i]['lan_r']))
#### mus正确matrix
accmatrix_mus = [[],[],[]]
for i in range(len(subs)):
    accmatrix_mus[0].append(mus_acc_matrix(beha[i]['mus_ir']))
    accmatrix_mus[1].append(mus_acc_matrix(beha[i]['mus_r4']))
    accmatrix_mus[2].append(mus_acc_matrix(beha[i]['mus_r2']))
#%% plot 正确率矩阵
#### geo
with plt.style.context(style_path):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axs = plt.subplots(3, len(subs), figsize=(7.5, 2.8), dpi=300) 
    for sub in range(len(subs)):
        for c in range(3): # lan=2; 
            y = accmatrix_geo[c][sub].mean(2)
            # y = accmatrix_lan[c][sub].mean(2)
            # y = accmatrix_mus[c][sub].mean(2)
            h1 = axs[c,sub].pcolormesh(y, vmin=0, vmax=1) # 
            axs[c,sub].invert_yaxis()
            
            axs[c,sub].set_xticks([])
            axs[c,sub].set_yticks([])
            if sub==0 and c==2:
                axs[c,sub].set_xticks(np.arange(8)+0.5, np.arange(1,9))
                axs[c,sub].set_yticks(np.arange(8)+0.5, np.arange(1,9))
            if c==0:
                axs[c,sub].set_title('P '+ str(sub+1))
                
    axs[2,0].set_xlabel('Reponse')
    axs[2,0].set_ylabel('Stimulus')
    plt.tight_layout()
    # fig.colorbar(h1)

#### geo example subject
with plt.style.context(style_path):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axs = plt.subplots(1, 3, figsize=(4.6, 1.7), dpi=300) 
    sub = 6
    for c in range(3): # lan=2; 
        y = accmatrix_geo[c][sub].mean(2)
        # y = accmatrix_lan[c][sub].mean(2)
        # y = accmatrix_mus[c][sub].mean(2)
        h1 = axs[c].pcolormesh(y, vmin=0.0, vmax=0.75) # 
        axs[c].invert_yaxis()
        
        axs[c].set_xticks([])
        axs[c].set_yticks([])
        

        axs[c].set_xticks(np.arange(8)+0.5, np.arange(1,9)) 
        axs[c].set_yticks(np.arange(8)+0.5, np.arange(1,9))             
    axs[0].set_xlabel('Reponse')
    axs[0].set_ylabel('Stimulus')
    axs[0].set_title('L0 task', size=10)
    axs[1].set_title('L1 task', size=10)
    axs[2].set_title('L2 task', size=10)
    # fig.colorbar(h1)
    plt.tight_layout(w_pad=4)
    
    
#### lan
with plt.style.context(style_path):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axs = plt.subplots(2, len(subs), figsize=(7.5, 1.9), dpi=300) 
    for sub in range(len(subs)):
        for c in range(2): # lan=2; 
            y = accmatrix_lan[c][sub].mean(2)
            h1 = axs[c,sub].pcolormesh(y, vmin=0, vmax=.5) # 
            axs[c,sub].invert_yaxis()
            
            axs[c,sub].set_xticks([])
            axs[c,sub].set_yticks([])
            if sub==0 and c==2:
                axs[c,sub].set_xticks(np.arange(8)+0.5, np.arange(1,9)) 
                axs[c,sub].set_yticks(np.arange(8)+0.5, np.arange(1,9)) 
            
            if c==0:
                axs[c,sub].set_title('P '+ str(sub+1))
                
    axs[1,0].set_xlabel('Reponse')
    axs[1,0].set_ylabel('Stimulus')
    plt.tight_layout()
#%% 计算整体正确率
acc_alltask = {}
err_allsubs_geo = []
acc_allsubs_geo = np.zeros([len(subs), 3])
acc_allsubs_lan = np.zeros([len(subs), 2])
acc_allsubs_mus = np.zeros([len(subs), 2])
for sub in range(len(subs)):
    info = data['trialInfo'][sub].copy()
    trialnum = info.shape[0]
    info['acc'] = 0
    
    idx_gir = 0
    idx_gr4 = 0
    idx_gr2 = 0
    idx_lir = 0
    idx_lr4 = 0
    idx_mir = 0
    idx_mr4 = 0
    idx_mr2 = 0
    for i in range(trialnum):
        if info.loc[i, 'task'] == 'geo_ir':
            diagonal = accmatrix_geo[0][sub][:,:,idx_gir].diagonal()
            if np.all(diagonal == 1):
                info.loc[i,'acc'] = 1 # 正确
            elif np.all(accmatrix_geo[0][sub][:,:,idx_gir] == 0):
                info.loc[i,'acc'] = 2 # 没回答
            else:
                info.loc[i,'acc'] = 3 # 错误回答
            idx_gir += 1
                
        if info.loc[i, 'task'] == 'geo_r4':
            diagonal = accmatrix_geo[1][sub][:,:,idx_gr4].diagonal()
            if np.all(diagonal == 1):
                info.loc[i,'acc'] = 1
            elif np.all(accmatrix_geo[1][sub][:,:,idx_gr4] == 0):
                info.loc[i,'acc'] = 2
            else:
                info.loc[i,'acc'] = 3
            idx_gr4 += 1
        
        if info.loc[i, 'task'] == 'geo_r2':
            diagonal = accmatrix_geo[2][sub][:,:,idx_gr2].diagonal()
            if np.all(diagonal == 1):
                info.loc[i,'acc'] = 1
            elif np.all(accmatrix_geo[2][sub][:,:,idx_gr2] == 0):
                info.loc[i,'acc'] = 2
            else:
                info.loc[i,'acc'] = 3
            idx_gr2 += 1
        
        #### language
        if info.loc[i, 'task'] == 'lan_ir':
            diagonal = accmatrix_lan[0][sub][:,:,idx_lir].diagonal()
            if np.all(diagonal == 1):
                info.loc[i,'acc'] = 1 # 正确
            elif np.all(accmatrix_lan[0][sub][:,:,idx_lir] == 0):
                info.loc[i,'acc'] = 2 # 没回答
            else:
                info.loc[i,'acc'] = 3 # 错误回答
            idx_lir += 1
                
        if info.loc[i, 'task'] == 'lan_r':
            diagonal = accmatrix_lan[1][sub][:,:,idx_lr4].diagonal()
            if np.all(diagonal == 1):
                info.loc[i,'acc'] = 1
            elif np.all(accmatrix_lan[1][sub][:,:,idx_lr4] == 0):
                info.loc[i,'acc'] = 2
            else:
                info.loc[i,'acc'] = 3
            idx_lr4 += 1
        
        #### music
        if info.loc[i, 'task'] == 'mus_ir':
            tmp = data['beha'][sub]['mus_ir'].copy()
            tmp = tmp.loc[tmp['trial']==idx_mir+1, 'error']
            # stim = tmp.loc[tmp['trial']==idx_mir+1, 'pitch_stim']
            # resp = tmp.loc[tmp['trial']==idx_mir+1, 'pitch']
            # mask = ~np.isnan(resp)
            # r, p = stats.pearsonr(stim[mask], resp[mask])
            # r = r*sum(mask)/8
            if np.all(tmp == 0):
                info.loc[i,'acc'] = 1 # 正确
            elif np.all(tmp == np.nan):
                info.loc[i,'acc'] = 2 # 没回答
            else:
                info.loc[i,'acc'] = 3 # 错误回答
            idx_mir += 1
                
        if info.loc[i, 'task'] == 'mus_r4':
            tmp = data['beha'][sub]['mus_r4'].copy()
            tmp = tmp.loc[tmp['trial']==idx_mr4+1, 'error']
            if np.all(tmp == 0):
                info.loc[i,'acc'] = 1 # 正确
            elif np.all(tmp == np.nan):
                info.loc[i,'acc'] = 2 # 没回答
            else:
                info.loc[i,'acc'] = 3 # 错误回答
            idx_mr4 += 1
                
        if info.loc[i, 'task'] == 'mus_r2':
            tmp = data['beha'][sub]['mus_r2'].copy()
            tmp = tmp.loc[tmp['trial']==idx_mr2+1, 'error']
            if np.all(tmp == 0):
                info.loc[i,'acc'] = 1 # 正确
            elif np.all(tmp == np.nan):
                info.loc[i,'acc'] = 2 # 没回答
            else:
                info.loc[i,'acc'] = 3 # 错误回答
            idx_mr2 += 1
            
    err_allsubs_geo.append(info)
    data['trialInfo'][sub] = info
    
    # info = info.loc[info['task'].str.contains('geo'),:].reset_index(drop=True)
    tmp = info.groupby('task')['acc'].apply(lambda x: (x == 1).mean())
    
    acc_allsubs_geo[sub,:] = [tmp['geo_ir'], tmp['geo_r4'], tmp['geo_r2']]
    acc_allsubs_lan[sub,:] = [tmp['lan_ir'], tmp['lan_r']]
    acc_allsubs_mus[sub,:] = [tmp['mus_ir'], tmp['mus_r2']]
    # acc_allsubs[sub,:] = [tmp['mus_ir'], tmp['mus_r4'], tmp['mus_r2']]
  
# df = pd.DataFrame({'sub':subs,'geo_ir':acc_allsubs[:,0],'geo_r4':acc_allsubs[:,1],
#                    'geo_r2':acc_allsubs[:,1]})
# path = r'F:\Human_chunking_Huashan_intraoperative\Result\Beh_acc_rt'
# pkl.dump({'err_allsubs':err_allsubs},open(path+'\\geo_errtype.pkl','wb'))

with plt.style.context(style_path):
    import seaborn as sns
    acc_allsubs = acc_allsubs_geo.copy()
    
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axs = plt.subplots(1,1, figsize=(1.2, 1.2), dpi=300)
    yerr = acc_allsubs.std(0)/np.sqrt(6)

    axs.boxplot(acc_allsubs[:,0], positions=[0], widths=0.3, patch_artist=True, boxprops=dict(facecolor='gray',edgecolor='None'),
                flierprops=dict(marker='.', color=colors[0], markersize=2),medianprops=dict(color='k', linewidth=1))
    axs.boxplot(acc_allsubs[:,1], positions=[1], widths=0.3, patch_artist=True, boxprops=dict(facecolor=colors[0],edgecolor='None'),
                flierprops=dict(marker='.', color=colors[0], markersize=2),medianprops=dict(color='k', linewidth=1))
    axs.boxplot(acc_allsubs[:,2], positions=[2], widths=0.3, patch_artist=True, boxprops=dict(facecolor=colors[1],edgecolor='None'),
                flierprops=dict(marker='.', color=colors[0], markersize=2),medianprops=dict(color='k', linewidth=1))
    
    # axs.bar([0,1,2], acc_allsubs.mean(0))
    # jitter_strength = 0.1   # 控制抖动幅度
    # x_jittered = np.random.uniform(-jitter_strength, jitter_strength, size=acc_allsubs.shape[0])
    # axs.scatter(0+x_jittered, acc_allsubs[:,0], color='k', alpha=0.5)
    # axs.scatter(1+x_jittered, acc_allsubs[:,1], color='k', alpha=0.5)
    # axs.scatter(2+x_jittered, acc_allsubs[:,2], color='k', alpha=0.5)
    axs.set_xlim([-0.8, 2.8])
    axs.set_ylim([-0.05, 1.2])
    axs.set_xticks([0,1,2],['L0', 'L1', 'L2'])
    axs.set_yticks([0,0.5,1],['0', '50', '100'])
    axs.set_xlabel('Sequence')
    axs.set_ylabel('Accuracy (%)')
    
stats.ttest_rel(acc_allsubs[:,0], acc_allsubs[:,1], nan_policy='omit')
stats.ttest_rel(acc_allsubs[:,0], acc_allsubs[:,2], nan_policy='omit')


with plt.style.context(style_path):
    import seaborn as sns
    # acc_allsubs = acc_allsubs_mus.copy()
    acc_allsubs = acc_allsubs_lan.copy()
    
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axs = plt.subplots(1,1, figsize=(1, 1.2), dpi=300)
    yerr = acc_allsubs.std(0)/np.sqrt(6)

    axs.boxplot(acc_allsubs[:,0], positions=[0], widths=0.3, patch_artist=True, boxprops=dict(facecolor='gray',edgecolor='None'),
                flierprops=dict(marker='.', color=colors[0], markersize=2),medianprops=dict(color='k', linewidth=1))
    axs.boxplot(acc_allsubs[:,1], positions=[1], widths=0.3, patch_artist=True, boxprops=dict(facecolor=colors[0],edgecolor='None'),
                flierprops=dict(marker='.', color=colors[0], markersize=2),medianprops=dict(color='k', linewidth=1))
    
    axs.set_xlim([-0.8, 1.8])
    axs.set_ylim([-0.05, 1.2])
    axs.set_xticks([0,1],['L0', 'L1'])
    axs.set_yticks([0,0.5,1],['0', '50', '100'])
    axs.set_xlabel('Sequence')
    axs.set_ylabel('Accuracy (%)')

stats.ttest_rel(acc_allsubs[:,0], acc_allsubs[:,1], nan_policy='omit')
#%% 保存数据，包括正确错误标记（在data['trialInfo']中）
path = r'F:\Human_chunking_Huashan_intraoperative\Data\Recording\PFC'
file = path + '\\fr_beha_PFC(10sub_160)_errormark_drift_new2.pkl'
# file = path + '\\fr_beha_PFC(9sub_155)_errormark_drift_new.pkl'
# file = path + '\\fr_beha_PFC(9sub_153)_errormark.pkl'
# file = path + '\\fr_beha_PFC(9sub_lan_250)_witherror.pkl'
# file = path + '\\fr_beha_PFC(9sub_mus_250)_witherror.pkl'
pkl.dump(data, open(file, 'wb'))

#### motor
# path = r'F:\Human_chunking_Huashan_intraoperative\Data\Recording\motor'
# file = path + '\\fr_beha_motor(2sub_67)_errormark_drift.pkl'
# pkl.dump(data, open(file, 'wb'))
#%% 分别保存正确/错误trial的FR
neuInfo = pd.concat([data['neu_info'][i].loc[:, ['subID','neuid_in_521']] for i in range(len(subs))])

fr_gir = []
fr_gr4 = []
fr_gr2 = []
for sub in range(len(subs)):
    idx_gir = data['trialInfo'][sub].query('task == "geo_ir" and acc==1').index.values
    idx_r4 = data['trialInfo'][sub].query('task == "geo_r4" and acc==1').index.values
    idx_r2 = data['trialInfo'][sub].query('task == "geo_r2" and acc==1').index.values
    fr_gir.append(data['fr'][sub][:,idx_gir,:].mean(1))
    fr_gr4.append(data['fr'][sub][:,idx_r4,:].mean(1))
    fr_gr2.append(data['fr'][sub][:,idx_r2,:].mean(1))
fr_gir = np.concat(fr_gir, axis=0)
fr_gr4 = np.concat(fr_gr4, axis=0)
fr_gr2 = np.concat(fr_gr2, axis=0)

# plt.imshow(fr_gir[:,8:88])
# plt.imshow(fr_gr4[:,8:88])

# fr_lir = []
# fr_lr4 = []
# for sub in range(len(subs)):
#     idx_gir = data['trialInfo'][sub].query('task == "lan_ir" and acc!=2').index.values
#     idx_r4 = data['trialInfo'][sub].query('task == "lan_r" and acc!=2').index.values
#     fr_lir.append(data['fr'][sub][:,idx_gir,:].mean(1))
#     fr_lr4.append(data['fr'][sub][:,idx_r4,:].mean(1))
    
# fr_lir = np.concat(fr_lir, axis=0)
# fr_lr4 = np.concat(fr_lr4, axis=0)

fr_mir = []
fr_mr4 = []
fr_mr2 = []
for sub in range(len(subs)):
    idx_gir = data['trialInfo'][sub].query('task == "mus_ir" and acc!=2').index.values
    idx_r4 = data['trialInfo'][sub].query('task == "mus_r4" and acc!=2').index.values
    idx_r2 = data['trialInfo'][sub].query('task == "mus_r2" and acc!=2').index.values
    fr_mir.append(data['fr'][sub][:,idx_gir,:].mean(1))
    fr_mr4.append(data['fr'][sub][:,idx_r4,:].mean(1))
    fr_mr2.append(data['fr'][sub][:,idx_r2,:].mean(1))
    
fr_mir = np.concat(fr_mir, axis=0)
fr_mr4 = np.concat(fr_mr4, axis=0)
fr_mr2 = np.concat(fr_mr2, axis=0)

#### 保存正确/错误 trial的FR
savepath = r'F:\Human_chunking_Huashan_intraoperative\Data\Recording\PFC\Correct_error_meanFR'
# savefile = savepath + '\\fr_beha_PFC(9sub_geo_155)_error_drift.mat'
# savefile = savepath + '\\fr_beha_PFC(9sub_lan_155)_correct_drift.mat'
# savefile = savepath + '\\fr_beha_PFC(9sub_mus_155)_responded_drift.mat'

# savefile = savepath + '\\fr_beha_PFC(10sub_geo_160)_responded_drift.mat'
# savefile = savepath + '\\fr_beha_PFC(10sub_lan_160)_responded_drift.mat'
savefile = savepath + '\\fr_beha_PFC(10sub_mus_160)_responded_drift.mat'

# savefile = savepath + '\\fr_beha_PFC(10sub_geo_160)_correct_drift.mat'
# savefile = savepath + '\\fr_beha_PFC(10sub_geo_160)_error_drift.mat'

# savepath = r'F:\Human_chunking_Huashan_intraoperative\Data\Recording\motor\Correct_error_meanFR'
# savefile = savepath + '\\fr_beha_motor(2sub_geo_74)_responded_drift.mat'

# sio.savemat(savefile, {'fr_gir':fr_gir, 'fr_gr4':fr_gr4, 'fr_gr2':fr_gr2, 'neuInfo':neuInfo})
# sio.savemat(savefile, {'fr_lir':fr_lir, 'fr_lr4':fr_lr4, 'neuInfo':neuInfo})
sio.savemat(savefile, {'fr_mir':fr_mir, 'fr_mr4':fr_mr4, 'fr_mr2':fr_mr2, 'neuInfo':neuInfo})
#%% 每一个rank的正确率
acc_allrank_geo = [None] * 3
acc_allrank_geo_perm = [None] * 3
for i in range(3):
    tmp = np.concat(accmatrix_geo[i], axis=2)
    acc_allrank_geo[i] = tmp.mean(2).diagonal()*100
    
    acc_allrank_geo_perm[i] = np.zeros([1000,8])
    for pp in range(1000):
        perm_idx = np.random.default_rng().choice(tmp.shape[2], size=70, replace=False)
        tmp2 = tmp[:,:,perm_idx]
        acc_allrank_geo_perm[i][pp,:] = tmp2.mean(2).diagonal()*100
        
with plt.style.context(style_path):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axs = plt.subplots(1,1, figsize=(1.5, 1.5), dpi=300)
    axs.plot(acc_allrank_geo[0], color='gray')
    axs.plot(acc_allrank_geo[1])
    axs.plot(acc_allrank_geo[2])
    axs.scatter(np.arange(8), acc_allrank_geo[0], color='gray')
    axs.scatter(np.arange(8), acc_allrank_geo[1])
    axs.scatter(np.arange(8), acc_allrank_geo[2])
    axs.set_xticks(np.arange(0,8),[str(i) for i in range(1,9)])
    axs.set_xlim([-0.5, 8])
    axs.set_ylim([30, 95])
    axs.vlines(3.5, -10, 110, linestyles='--', color='gray')
    
    # axs.legend(['Irr', 'L1','L2'])
    
### lan
acc_allrank_lan = [None] * 2
acc_allrank_lan_perm = [None] * 2
for i in range(2):
    tmp = np.concat(accmatrix_lan[i], axis=2)
    acc_allrank_lan[i] = tmp.mean(2).diagonal()*100
    
    acc_allrank_lan_perm[i] = np.zeros([1000,8])
    for pp in range(1000):
        perm_idx = np.random.default_rng().choice(tmp.shape[2], size=87, replace=False)
        tmp2 = tmp[:,:,perm_idx]
        acc_allrank_lan_perm[i][pp,:] = tmp2.mean(2).diagonal()*100
    
with plt.style.context(style_path):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axs = plt.subplots(1,1, figsize=(1.5, 1.5), dpi=300)
    axs.plot(acc_allrank_lan[0], color='gray')
    axs.plot(acc_allrank_lan[1])
    axs.scatter(np.arange(8), acc_allrank_lan[0], color='gray')
    axs.scatter(np.arange(8), acc_allrank_lan[1])
    axs.set_xticks(np.arange(0,8),[str(i) for i in range(1,9)])
    axs.set_xlim([-0.5, 8])
    axs.set_ylim([1, 95])
    axs.vlines(3.5, -10, 110, linestyles='--', color='gray')
    
    axs.legend(['L0', 'L1'], loc='lower right')
    axs.set_xlabel('Rank')
    axs.set_ylabel('Accuracy (%)')
    
### mus
acc_allrank_mus = [None] * 3
acc_accu_mus = [None] * 3
acc_allrank_mus_perm = [None]*3
for i in range(3):
    tmp = np.concat(accmatrix_mus[i], axis=2)
    acc_allrank_mus[i] = tmp.mean(2).diagonal()*100
    acc_allrank_mus[i] = acc_allrank_mus[i][1:]
    #### 累计正确率
    tmp = tmp.diagonal(axis1=0, axis2=1)
    tmp = tmp[:,1:]
    accu = np.zeros([tmp.shape[0], 7, 2])
    for j in range(1,8):
        accu[:,j-1, 0] = tmp[:,:j].sum(1)
        mask = accu[:,j-1, 0] == j
        accu[mask,j-1, 1] = 100
    acc_accu_mus[i] = accu
    
    #### perm
    tmp = np.concat(accmatrix_mus[i], axis=2)
    acc_allrank_mus_perm[i] = np.zeros([1000,70,7,2])
    for pp in range(1000):
        perm_idx = np.random.default_rng().choice(tmp.shape[2], size=70, replace=False)
        tmp2 = tmp[:,:,perm_idx]
        #### 累计正确率
        tmp2 = tmp2.diagonal(axis1=0, axis2=1)
        tmp2 = tmp2[:,1:]
        accu = np.zeros([tmp2.shape[0], 7, 2])
        for j in range(1,8):
            accu[:,j-1, 0] = tmp2[:,:j].sum(1)
            mask = accu[:,j-1, 0] == j
            accu[mask,j-1, 1] = 100
        acc_allrank_mus_perm[i][pp,:,:,:] = accu
    
with plt.style.context(style_path):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axs = plt.subplots(1,1, figsize=(1.5, 1.5), dpi=300)
    axs.plot(acc_allrank_mus[0], color='gray')
    # axs.plot(acc_allrank_mus[1])
    axs.plot(acc_allrank_mus[2])
    axs.scatter(np.arange(7), acc_allrank_mus[0], color='gray')
    # axs.scatter(np.arange(7), acc_allrank_mus[1])
    axs.scatter(np.arange(7), acc_allrank_mus[2])
    axs.set_xticks(np.arange(0,7),[str(i)+'-'+str(i+1) for i in range(1,8)],rotation=30)
    axs.set_xlim([-0.5, 7])
    axs.set_ylim([30, 95])
    axs.vlines(2.5, -10, 110, linestyles='--', color='gray')
    
    fig, axs = plt.subplots(1,1, figsize=(1.5, 1.5), dpi=300)
    axs.plot(acc_accu_mus[0][:,:,1].mean(0), color='gray')
    # axs.plot(acc_accu_mus[1][:,:,1].mean(0))
    axs.plot(acc_accu_mus[2][:,:,1].mean(0))
    
    axs.scatter(np.arange(7), acc_accu_mus[0][:,:,1].mean(0), color='gray')
    # axs.scatter(np.arange(7), acc_accu_mus[1][:,:,1].mean(0))
    axs.scatter(np.arange(7), acc_accu_mus[2][:,:,1].mean(0))
    
    axs.set_xticks(np.arange(0,7),[str(1)+'-'+str(i+1) for i in range(1,8)],rotation=30)
    axs.set_xlim([-0.5, 7])
    # axs.set_ylim([30, 95])
    axs.vlines(2.5, -10, 110, linestyles='--', color='gray')
#%% music,计算rank累计正确率
import numpy as np
from scipy.optimize import curve_fit

z_scores = np.zeros([100, 6, 2])
for i in range(100):
    idx = np.random.choice(np.arange(acc_accu_mus[0].shape[0]), size=70, replace=False)
    # 1. 差分法
    diff = np.diff(acc_accu_mus[0][idx,:,1].mean(0))
    z_scores[i,:,0] = (diff - diff.mean()) / (diff.std() + 1e-8)
    diff = np.diff(acc_accu_mus[2][idx,:,1].mean(0))
    z_scores[i,:,1] = (diff - diff.mean()) / (diff.std() + 1e-8)
    # sudden_jump = np.any(np.abs(z_scores) > 2)  # 阈值可调

z_scores = z_scores*-1
with plt.style.context(style_path):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    
    fig, axs = plt.subplots(1,1, figsize=(1.2, 1.4), dpi=300)
    df = pd.DataFrame(z_scores[:,:,0], columns=[f'Col{i+1}' for i in range(6)])
    df_melt = df.melt(var_name='Column', value_name='Value')
    sns.violinplot(x='Column', y='Value', data=df_melt, color="gray", inner=None)
    axs.set_xticks(np.arange(0,6),[str(i+2) for i in range(1,7)],rotation=30)
    axs.set_ylim([-2.9,2.9])
    axs.hlines(y=[-2,2], xmin=-0.5, xmax=5.5, colors='orange', linestyles='dashed', linewidth=0.6)
    axs.set_xlabel('')
    axs.set_ylabel('')
    
    fig, axs = plt.subplots(1,1, figsize=(1.2, 1.4), dpi=300)
    df = pd.DataFrame(z_scores[:,:,1], columns=[f'Col{i+1}' for i in range(6)])
    df_melt = df.melt(var_name='Column', value_name='Value')
    sns.violinplot(x='Column', y='Value', data=df_melt, color=colors[0], inner=None)
    axs.set_xticks(np.arange(0,6),[str(i+2) for i in range(1,7)],rotation=30)
    axs.set_ylim([-2.9,2.9])
    axs.hlines(y=[-2,2], xmin=-0.5, xmax=5.5, colors='orange', linestyles='dashed', linewidth=0.6)
    axs.set_xlabel('')
    axs.set_ylabel('')
    
stats.ttest_1samp(z_scores[:,2,0], 2)
stats.ttest_1samp(z_scores[:,2,1], 2)
#%% 统计8rank正确率趋势符合那种模型（U/分段）
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def u_shape_model(x, params):
    a, b, c = params
    return a * x**2 + b * x + c

def loss(params, x, y):
    y_pred = u_shape_model(x, params)
    return np.sum((y - y_pred)**2)

def fit_u_shape_with_constraint(x, y, a_min=0.01):
    n = len(x)

    # 初始猜测
    init_params = [0.1, 0, np.mean(y)]

    # 约束1：a > a_min
    bounds = [(a_min, None), (None, None), (None, None)]

    # 约束2：中间点低于两端（例如 y[3] 或 y[4] < y[0], y[7]）
    def constraint_middle_lower(params):
        y_fit = u_shape_model(x, params)
        mid = y_fit[n//2 - 1]  # 中间点
        return min(y_fit[0] - mid, y_fit[-1] - mid)  # 要求中间点更小

    cons = ({'type': 'ineq', 'fun': constraint_middle_lower})

    result = minimize(loss, init_params, args=(x, y), bounds=bounds, constraints=cons)

    if not result.success:
        print("拟合失败或无法满足U形约束")
    return result.x  # a, b, c

def model2_piecewise_constant(x, c1, c2):
    return np.array([c1 if i < 4 else c2 for i in x])

def model2_piecewise_linear(x, k1, b1, b2):
    x = np.array(x)
    y = np.empty_like(x, dtype=float)
    for i in range(len(x)):
        if x[i] < 4:
            y[i] = k1 * x[i] + b1
        else:
            y[i] = k1 * x[i] + b2
    return y

def model3_piecewise_linear(x, k1, b1, b2, b3, b4):
    x = np.array(x)
    y = np.empty_like(x, dtype=float)
    for i in range(len(x)):
        if x[i] < 2:
            y[i] = k1 * x[i] + b1
        elif 2<= x[i] < 4:
            y[i] = k1 * x[i] + b2
        elif 4<= x[i] < 6:
            y[i] = k1 * x[i] + b3
        elif 6<= x[i] < 8:
            y[i] = k1 * x[i] + b4
    return y

def model3_piecewise_4segments(x, c1, c2, c3, c4):
    return np.array([
        c1 if i < 2 else
        c2 if i < 4 else
        c3 if i < 6 else
        c4 for i in x
    ])

def compute_bic(n, k, rss):
    return k * np.log(n) + n * np.log(rss / n)

def compute_adjusted_bic(n, k, rss):
    return k * np.log(n*k) + n * np.log(rss / n)

def fit_models(y):
    x = np.arange(len(y))
    n = len(y)
    rssbt = np.abs(y[3].mean() - y[4].mean())
    
    ### 模型1：U形拟合
    # 模型1：U形拟合（加约束）
    popt1 = fit_u_shape_with_constraint(x, y)
    y1_pred = u_shape_model(x, popt1)
    rss1 = np.sum((y - y1_pred) ** 2)
    rss1 = rss1/rssbt
    bic1 = compute_adjusted_bic(n, k=3, rss=rss1)
    
    # popt1, _ = curve_fit(model1_quadratic, x, y)
    # y1_pred = model1_quadratic(x, *popt1)
    # rss1 = np.sum((y - y1_pred)**2)
    # bic1 = compute_bic(n, k=3, rss=rss1)

    ### 模型2：2段常数（4+4）
    popt2, _ = curve_fit(model2_piecewise_linear, x, y)
    y2_pred = model2_piecewise_linear(x, *popt2)
    rss2 = np.sum((y - y2_pred)**2)
    rss2 = rss2/rssbt
    # rss2 = rss2/np.abs(popt2[1]-popt2[2])
    bic2 = compute_adjusted_bic(n, k=3, rss=rss2)

    ### 模型3：4段常数（2+2+2+2）
    popt3, _ = curve_fit(model3_piecewise_linear, x, y)
    y3_pred = model3_piecewise_linear(x, *popt3)
    rss3 = np.sum((y - y3_pred)**2)
    rss3 = rss3/rssbt
    bic3 = compute_adjusted_bic(n, k=5, rss=rss3)

    # 选最优
    bic_all = {"U形模型": bic1, "4+4模型": bic2, "2+2+2+2模型": bic3}
    best_model = min(bic_all, key=bic_all.get)

    return {
        "U形模型": {"BIC": bic1, "RSS": rss1, "参数": popt1},
        "4+4模型": {"BIC": bic2, "RSS": rss2,"参数": popt2},
        "2+2+2+2模型": {"BIC": bic3, "RSS": rss3,"参数": popt3},
        "最优模型": best_model
    }

scaler = MinMaxScaler(feature_range=(1, 100))

#### geo
# fit_result = [None]*3
# fit_result[0] = fit_models(scaler.fit_transform(acc_allrank_geo[0].reshape(-1,1))[:,0])
# fit_result[1] = fit_models(scaler.fit_transform(acc_allrank_geo[1].reshape(-1,1))[:,0])
# fit_result[2] = fit_models(scaler.fit_transform(acc_allrank_geo[2].reshape(-1,1))[:,0])

# df_bic = pd.DataFrame({'irr':[fit_result[0][i]['BIC'] for i in ['U形模型', '4+4模型', '2+2+2+2模型']],
#                        'L1':[fit_result[1][i]['BIC'] for i in ['U形模型', '4+4模型', '2+2+2+2模型']],
#                        'L2':[fit_result[2][i]['BIC'] for i in ['U形模型', '4+4模型', '2+2+2+2模型']]})  

# df_rss = pd.DataFrame({'irr':[fit_result[0][i]['RSS'] for i in ['U形模型', '4+4模型', '2+2+2+2模型']],
#                        'L1':[fit_result[1][i]['RSS'] for i in ['U形模型', '4+4模型', '2+2+2+2模型']],
#                        'L2':[fit_result[2][i]['RSS'] for i in ['U形模型', '4+4模型', '2+2+2+2模型']]})
# print("a 拟合结果：", fit_result[0])
# print("b 拟合结果：", fit_result[1])
# print("c 拟合结果：", fit_result[2])

#### lan 
# fit_result = [None]*2
# fit_result[0] = fit_models(scaler.fit_transform(acc_allrank_lan[0].reshape(-1,1))[:,0])
# fit_result[1] = fit_models(scaler.fit_transform(acc_allrank_lan[1].reshape(-1,1))[:,0])

# df_bic = pd.DataFrame({'irr':[fit_result[0][i]['BIC'] for i in ['U形模型', '4+4模型', '2+2+2+2模型']],
#                        'L1':[fit_result[1][i]['BIC'] for i in ['U形模型', '4+4模型', '2+2+2+2模型']]}) 


#### geo perm
df_bic_perm = np.zeros([3,3,50])
fit_result_perm = []
for i in range(50):
    acc_allrank_geo[0] = acc_allrank_geo_perm[0][i,:]
    acc_allrank_geo[1] = acc_allrank_geo_perm[1][i,:]
    acc_allrank_geo[2] = acc_allrank_geo_perm[2][i,:]
    fit_result = [None]*3
    fit_result[0] = fit_models(scaler.fit_transform(acc_allrank_geo[0].reshape(-1,1))[:,0])
    fit_result[1] = fit_models(scaler.fit_transform(acc_allrank_geo[1].reshape(-1,1))[:,0])
    fit_result[2] = fit_models(scaler.fit_transform(acc_allrank_geo[2].reshape(-1,1))[:,0])
    
    df_bic = pd.DataFrame({'irr':[fit_result[0][i]['BIC'] for i in ['U形模型', '4+4模型', '2+2+2+2模型']],
                           'L1':[fit_result[1][i]['BIC'] for i in ['U形模型', '4+4模型', '2+2+2+2模型']],
                           'L2':[fit_result[2][i]['BIC'] for i in ['U形模型', '4+4模型', '2+2+2+2模型']]}) 
    
    df_bic_perm[:,:,i] = df_bic.values # model * task
    fit_result_perm.append(fit_result)

df_bic_perm[np.isinf(df_bic_perm)] = np.nan
df_bic_perm[df_bic_perm<-100] = np.nan
df_bic_perm[df_bic_perm>100] = np.nan

pkl.dump({'df_bic_perm':df_bic_perm,'fit_result_perm':fit_result_perm},
    open(r'F:\Human_chunking_Huashan_intraoperative\Result\Beh_acc_modelFit\geo_10subs.pkl','wb'))

stats.ttest_ind(df_bic_perm[0,1,:50], df_bic_perm[0,2,:50], nan_policy='omit')
stats.ttest_ind(df_bic_perm[0,1,:50], df_bic_perm[0,2,:50], nan_policy='omit')
stats.ttest_ind(df_bic_perm[1,1,:50], df_bic_perm[1,2,:50], nan_policy='omit')

# sum(df_bic_perm[0,0,:]>np.nanmean(df_bic_perm[0,2,:]))/1000

#### lan  perm
df_bic_perm = np.zeros([3,2,50])
fit_result_perm = []
for i in range(50):
    acc_allrank_lan[0] = acc_allrank_lan_perm[0][i,:]
    acc_allrank_lan[1] = acc_allrank_lan_perm[1][i,:]
    
    fit_result = [None]*2
    fit_result[0] = fit_models(scaler.fit_transform(acc_allrank_lan[0].reshape(-1,1))[:,0])
    fit_result[1] = fit_models(scaler.fit_transform(acc_allrank_lan[1].reshape(-1,1))[:,0])
    
    df_bic = pd.DataFrame({'irr':[fit_result[0][i]['BIC'] for i in ['U形模型', '4+4模型', '2+2+2+2模型']],
                           'L1':[fit_result[1][i]['BIC'] for i in ['U形模型', '4+4模型', '2+2+2+2模型']]}) 
    df_bic_perm[:,:,i] = df_bic.values
    fit_result_perm.append(fit_result)

df_bic_perm = df_bic_perm[:2,:2,:]
df_bic_perm[np.isinf(df_bic_perm)] = np.nan
df_bic_perm[df_bic_perm<-100] = np.nan
df_bic_perm[df_bic_perm>100] = np.nan

pkl.dump({'df_bic_perm':df_bic_perm,'fit_result_perm':fit_result_perm},
    open(r'F:\Human_chunking_Huashan_intraoperative\Result\Beh_acc_modelFit\lan_10subs.pkl','wb'))

stats.ttest_ind(df_bic_perm[0,0,:50], df_bic_perm[0,1,:50], nan_policy='omit')
stats.ttest_ind(df_bic_perm[1,0,:50], df_bic_perm[1,1,:50], nan_policy='omit')
#%% bar plot, permutation  
df_bic_perm = pkl.load(open(r'F:\Human_chunking_Huashan_intraoperative\Result\Beh_acc_modelFit\geo_10subs.pkl','rb'))['df_bic_perm'] 
aa = np.nanmean(df_bic_perm[:,:,:50],2).T
bb = np.nanstd(df_bic_perm[:,:,:50],2).T
with plt.style.context(style_path):
    from cycler import cycler
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    plt.rcParams["axes.prop_cycle"] = cycler(color=['gray'] + colors)
    fig, axs = plt.subplots(1,1, figsize=(1.6, 1.5), dpi=300)
    
    n_rows, n_cols = df_bic.shape
    x = np.arange(n_cols)  # 每组 bar 的 x 位置
    bar_width = 0.2
    
    for i in range(n_rows):
        axs.bar(x + i * bar_width, aa[i,:], width=bar_width, label=df_bic.index[i])
        axs.errorbar(x + i * bar_width, aa[i,:], yerr=bb[i,:], color='black', capsize=4, linestyle='none')
    axs.set_xticks([0.2,1.2,2.2])
    axs.set_xticklabels(['L0','L1','L2'])
    axs.set_ylabel('Adjusted BIC')
    axs.set_ylim([0,39])
    axs.set_xlim([-0.5,3])
    axs.set_xlabel('Model')

#### lan. mus
# df_bic_perm = pkl.load(open(r'F:\Human_chunking_Huashan_intraoperative\Result\Beh_acc_modelFit\lan_10subs.pkl','rb'))['df_bic_perm'] 
df_bic_perm = pkl.load(open(r'F:\Human_chunking_Huashan_intraoperative\Result\Beh_acc_modelFit\mus.pkl','rb'))['df_bic_perm'] 
aa = np.nanmean(df_bic_perm,2)
bb = np.nanstd(df_bic_perm,2)
with plt.style.context(style_path):
    from cycler import cycler
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    plt.rcParams["axes.prop_cycle"] = cycler(color=['gray'] + colors)
    fig, axs = plt.subplots(1,1, figsize=(1.1, 1.5), dpi=300)
    
    n_rows, n_cols = df_bic_perm[:,:,0].shape
    x = np.arange(n_cols)  # 每组 bar 的 x 位置
    bar_width = 0.2
    
    for i in range(n_rows):
        axs.bar(x + i * bar_width, aa[i,:], width=bar_width, label=df_bic.index[i])
        axs.errorbar(x + i * bar_width, aa[i,:], yerr=bb[i,:], color='black', capsize=4, linestyle='none')
        
    axs.set_xticks([0.2,1.2])
    axs.set_xticklabels(['L0','L1'])
    axs.set_ylabel('Adjusted BIC')
    axs.set_ylim([0,40])
    axs.set_xlim([-0.5,2])
    axs.set_xlabel('Model')
#%% violin plot
with plt.style.context(style_path):
    get_ipython().run_line_magic('matplotlib', 'inline')
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(1,1, figsize=(1.5, 1.5), dpi=300)
    color1 = ['gray',colors[0],colors[1],'gray',colors[0],colors[1],'gray',colors[0],colors[1]]
    positions = [1, 2, 3,   5, 6, 7,   9, 10, 11]  # 每列画在这些 x 坐标上
    k=0
    for i in range(3):
        for j in range(3):
            y = df_bic_perm[j,i,:50]
            y = y[~np.isnan(y)] 
            parts  = ax.violinplot(y, positions=[positions[k]],widths=0.7, showmeans=False, showmedians=False,
                                   showextrema=False)
            
            for pc in parts['bodies']:
                pc.set_facecolor(color1[k])
                pc.set_edgecolor('None')
                pc.set_alpha(0.8)
            if 'cbars' in parts:
                parts['cbars'].set_color(color1[k])
                parts['cbars'].set_linewidth(0.5)
                
            k+=1
        
        ax.set_xlim([-1, 13])
        ax.set_ylim([-10, 40])
        ax.set_xticklabels(['L0','L1','L2'])
        ax.set_xlabel('Task')
        ax.set_ylabel('BIC')
    
#%% 统计8rank正确率趋势符合那种模型（U/分段） music
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import least_squares
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def exp_decay(t, A, k, C):
    return A * np.exp(-k * t) + C

def residuals_exp_decay(params, x, y):
    return exp_decay(x, *params) - y

def model2_piecewise_constant(x, c1, c2):
    return np.array([c1 if i < 4 else c2 for i in x])

def model2_piecewise_linear(x, k1, b1, b2):
    x = np.array(x)
    y = np.empty_like(x, dtype=float)
    for i in range(len(x)):
        if x[i] < 3:
            y[i] = k1 * x[i] + b1
        else:
            y[i] = k1 * x[i] + b2
    return y

def residuals_model2_piecewise_linear(params, x, y):
    return model2_piecewise_linear(x, *params) - y

def model3_piecewise_linear(x, k1, b1, b2, b3, b4):
    x = np.array(x)
    y = np.empty_like(x, dtype=float)
    for i in range(len(x)):
        if x[i] < 2:
            y[i] = k1 * x[i] + b1
        elif 2 <= x[i] < 4:
            y[i] = k1 * x[i] + b2
        elif 4 <= x[i] < 6:
            y[i] = k1 * x[i] + b3
        elif 6 <= x[i] < 8:
            y[i] = k1 * x[i] + b4
    return y

def residuals_model3_piecewise_linear(params, x, y):
    return model3_piecewise_linear(x, *params) - y

def compute_bic(n, k, rss):
    return k * np.log(n) + n * np.log(rss / n)

def compute_adjusted_bic(n, k, rss):
    return k * np.log(n*k) + n * np.log(rss / n)

def fit_models(y):
    x = np.arange(len(y))
    n = len(y)
    rssbt = np.abs(y[3].mean() - y[4].mean())

    ### 模型1：U形拟合
    p0_1 = [1.0, 0.1, 0.0]  # 初始猜测
    res1 = least_squares(residuals_exp_decay, p0_1, args=(x, y), max_nfev=1000)
    popt1 = res1.x
    y1_pred = exp_decay(x, *popt1)
    rss1 = np.sum((y - y1_pred) ** 2) / rssbt
    bic1 = compute_adjusted_bic(n, k=3, rss=rss1)

    ### 模型2：2段线性
    p0_2 = [0.1, 0.0, 0.0]  # 初始猜测
    res2 = least_squares(residuals_model2_piecewise_linear, p0_2, args=(x, y), max_nfev=1000)
    popt2 = res2.x
    y2_pred = model2_piecewise_linear(x, *popt2)
    rss2 = np.sum((y - y2_pred) ** 2) / rssbt
    bic2 = compute_adjusted_bic(n, k=3, rss=rss2)

    ### 模型3：4段线性
    p0_3 = [0.1, 0.0, 0.0, 0.0, 0.0]  # 初始猜测
    res3 = least_squares(residuals_model3_piecewise_linear, p0_3, args=(x, y), max_nfev=1000)
    popt3 = res3.x
    y3_pred = model3_piecewise_linear(x, *popt3)
    rss3 = np.sum((y - y3_pred) ** 2) / rssbt
    bic3 = compute_adjusted_bic(n, k=5, rss=rss3)

    # 选最优
    bic_all = {"U形模型": bic1, "4+4模型": bic2, "2+2+2+2模型": bic3}
    best_model = min(bic_all, key=bic_all.get)

    return {
        "U形模型": {"BIC": bic1, "RSS": rss1, "参数": popt1},
        "4+4模型": {"BIC": bic2, "RSS": rss2,"参数": popt2},
        "2+2+2+2模型": {"BIC": bic3, "RSS": rss3,"参数": popt3},
        "最优模型": best_model
    }

scaler = MinMaxScaler(feature_range=(1, 100))

#### mus
# fit_result = [None]*2
# fit_result[0] = fit_models(scaler.fit_transform(acc_accu_mus[0][:,:,1].mean(0).reshape(-1,1))[:,0])
# fit_result[1] = fit_models(scaler.fit_transform(acc_accu_mus[2][:,:,1].mean(0).reshape(-1,1))[:,0])

# df_bic = pd.DataFrame({'irr':[fit_result[0][i]['BIC'] for i in ['U形模型', '4+4模型', '2+2+2+2模型']],
#                        'L1':[fit_result[1][i]['BIC'] for i in ['U形模型', '4+4模型', '2+2+2+2模型']]}) 

#### mus  perm
df_bic_perm = np.zeros([3,2,50])
fit_result_perm = []
for i in range(50):
    print(i)
    acc_accu_mus[0] = acc_allrank_mus_perm[0][i,:,:,:]
    acc_accu_mus[1] = acc_allrank_mus_perm[2][i,:,:,:]
    
    fit_result = [None]*2
    fit_result[0] = fit_models(scaler.fit_transform(acc_accu_mus[0][:,:,1].mean(0).reshape(-1,1))[:,0])
    fit_result[1] = fit_models(scaler.fit_transform(acc_accu_mus[1][:,:,1].mean(0).reshape(-1,1))[:,0])
    
    df_bic = pd.DataFrame({'irr':[fit_result[0][i]['BIC'] for i in ['U形模型', '4+4模型', '2+2+2+2模型']],
                           'L1':[fit_result[1][i]['BIC'] for i in ['U形模型', '4+4模型', '2+2+2+2模型']]}) 
    df_bic_perm[:,:,i] = df_bic.values
    fit_result_perm.append(fit_result)

df_bic_perm = df_bic_perm[:2,:2,:]
df_bic_perm[np.isinf(df_bic_perm)] = np.nan
df_bic_perm[df_bic_perm<-100] = np.nan
df_bic_perm[df_bic_perm>100] = np.nan

pkl.dump({'df_bic_perm':df_bic_perm,'fit_result_perm':fit_result_perm},
    open(r'F:\Human_chunking_Huashan_intraoperative\Result\Beh_acc_modelFit\mus_10subs.pkl','wb'))

stats.ttest_ind(df_bic_perm[0,0,:50], df_bic_perm[0,1,:50], nan_policy='omit')
stats.ttest_ind(df_bic_perm[1,0,:50], df_bic_perm[1,1,:50], nan_policy='omit')
#%% 统计8rank正确率趋势符合那种模型（U/分段）, plot
#### geo
fit_result = pkl.load(open(r'F:\Human_chunking_Huashan_intraoperative\Result\Beh_acc_modelFit\geo_10subs.pkl',
                           'rb'))['fit_result_perm'][12]
with plt.style.context(style_path):
    from cycler import cycler
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    plt.rcParams["axes.prop_cycle"] = cycler(color=['gray'] + colors)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axs = plt.subplots(1,3, figsize=(4, 1.5), dpi=300)
    
    linetype = [['-','--','--'],['--','-','--'],['--','--','-']]
    for task in range(3):
        x = np.arange(8)
        y = scaler.fit_transform(acc_allrank_geo[task].reshape(-1,1))
        y = scaler.inverse_transform(y)

        p = fit_result[task]['U形模型']['参数']
        y1_pred = u_shape_model(x, p)
        axs[task].plot(scaler.inverse_transform(y1_pred.reshape(-1,1)), linestyle=linetype[task][0])
        
        p = fit_result[task]['4+4模型']['参数']
        y2_pred = model2_piecewise_linear(x, *p)
        axs[task].plot(scaler.inverse_transform(y2_pred.reshape(-1,1)), linestyle=linetype[task][1])
        
        p = fit_result[task]['2+2+2+2模型']['参数']
        y3_pred = model3_piecewise_linear(x, *p)
        axs[task].plot(scaler.inverse_transform(y3_pred.reshape(-1,1)), linestyle=linetype[task][2])
        
        axs[task].scatter(x, y, color=colors[task], edgecolor=[], alpha=1)
        axs[task].vlines(3.5, y.min()-7, y.max()+5, linestyles='--', color='gray')
        axs[task].set_xlim(-0.5, 7.8)
        axs[task].set_xticks(np.arange(8),[str(i+1) for i in range(8)])
        # axs[task].set_facecolor(colors[task])
        axs[task].patch.set_alpha(0.1)
    axs[0].set_ylabel('Accuracy (%)')
    
#### lan
fit_result = pkl.load(open(r'F:\Human_chunking_Huashan_intraoperative\Result\Beh_acc_modelFit\lan_10subs.pkl',
                           'rb'))['fit_result_perm'][12]
with plt.style.context(style_path):
    from cycler import cycler
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    plt.rcParams["axes.prop_cycle"] = cycler(color=['gray'] + colors)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axs = plt.subplots(1,2, figsize=(2.2, 1.5), dpi=300)
    
    for task in range(2):
        x = np.arange(8)
        y = scaler.fit_transform(acc_allrank_lan[task].reshape(-1,1))
        y = scaler.inverse_transform(y)

        p = fit_result[task]['U形模型']['参数']
        y1_pred = u_shape_model(x, p)
        axs[task].plot(scaler.inverse_transform(y1_pred.reshape(-1,1)))
        
        p = fit_result[task]['4+4模型']['参数']
        y2_pred = model2_piecewise_linear(x, *p)
        axs[task].plot(scaler.inverse_transform(y2_pred.reshape(-1,1)))
        
        axs[task].scatter(x, y, color=colors[task], edgecolor=[], alpha=1)
        axs[task].vlines(3.5, y.min()-7, y.max()+7, linestyles='--', color='gray')
        axs[task].set_xlim(-0.5, 7.8)
        axs[task].set_xticks(np.arange(8),[str(i+1) for i in range(8)])
        
    axs[0].set_ylabel('Accuracy (%)')
    plt.tight_layout()

#### music
fit_result = pkl.load(open(r'F:\Human_chunking_Huashan_intraoperative\Result\Beh_acc_modelFit\mus_10subs.pkl',
                           'rb'))['fit_result_perm'][12]
with plt.style.context(style_path):
    from cycler import cycler
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    plt.rcParams["axes.prop_cycle"] = cycler(color=['gray'] + colors)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axs = plt.subplots(1,2, figsize=(2.2, 1.5), dpi=300)
    
    tt = [0,2]
    for task in range(2):
        x = np.arange(7)
        y = scaler.fit_transform(acc_accu_mus[tt[task]][:,:,1].mean(0).reshape(-1,1))
        y = scaler.inverse_transform(y)

        p = fit_result[task]['U形模型']['参数']
        y1_pred = exp_decay(x, *p)
        axs[task].plot(scaler.inverse_transform(y1_pred.reshape(-1,1)))
        
        p = fit_result[task]['4+4模型']['参数']
        y2_pred = model2_piecewise_linear(x, *p)
        axs[task].plot(scaler.inverse_transform(y2_pred.reshape(-1,1)))
        
        axs[task].scatter(x, y, color=colors[task], edgecolor=[], alpha=1)
        axs[task].vlines(2.5, 0, 85, linestyles='--', color='gray')
        axs[task].set_xlim(-0.5, 6.8)
        axs[task].set_ylim(0, 85)
        axs[task].set_xticks(np.arange(7),[str(i+2) for i in range(7)],rotation=30)
        
    axs[0].set_ylabel('Accuracy (%)')
    
    plt.tight_layout()    
#%% 计算连续4个正确的概率
# condition = ['geo_ir','geo_r4','geo_r2','lan_ir','lan_r','mus_ir','mus_r4','mus_r2']
# acc_count_geo = []
# for s in range(len(subs)):
#     for c in condition:
#         df = beha[s][c]
#         df['acc_count'] = -1
#         trialNum = len(np.unique(df['trial'])) - 1
#         errPat = np.zeros([trialNum, 7])
#         for trial in range(trialNum):
#             # df['name'] = df['name'].fillna(-99)  # 将'a'列的NaN替换为-99
#             trialid = trial+1
#             idx = df.query('trial == @trialid').index[0]
#             for i in range(5):
#                 x = df.loc[idx+i:idx+i+3, 'target'].values
#                 y = df.loc[idx+i:idx+i+3, 'name'].values
#                 beha[s][c].loc[idx+i,'acc_count'] = sum(x==y)
#%% chunk内/间错误比例 （geo）

#### 错误item总数
err_item_num = [None]*3
for i in range(3):
    tmp = np.concat(accmatrix_geo[i], axis=2)
    err_item_num[i] = tmp.shape[2]*8 - sum(tmp.sum(2).diagonal())

#### 错误trial总数8*8矩阵
accmatrix_geo_merge = [None]*3
accmatrix_geo_merge[0] = np.concatenate(accmatrix_geo[0],axis=2).sum(2)
accmatrix_geo_merge[1] = np.concatenate(accmatrix_geo[1],axis=2).sum(2)    
accmatrix_geo_merge[2] = np.concatenate(accmatrix_geo[2],axis=2).sum(2)    

#### 1-5/5-1 chunk起始点交换错误
chunk_start_err = np.zeros([3,1])
chunk_start_err[0,0] = accmatrix_geo_merge[0][0,4] + accmatrix_geo_merge[0][4,0]
chunk_start_err[1,0] = accmatrix_geo_merge[1][0,4] + accmatrix_geo_merge[1][4,0]
chunk_start_err[2,0] = accmatrix_geo_merge[2][0,4] + accmatrix_geo_merge[2][4,0]

#### chunk内/chunk间错误
errtrial_crosschunk = np.zeros([3,2,len(subs)])
for i in range(3):
    for s in range(len(subs)):
        tmp = accmatrix_geo[i][s]
        tmp = tmp.sum(2)
        
        x1 = tmp[:4,:4]
        mask = ~np.eye(x1.shape[0], dtype=bool)
        x1 = x1[mask].sum()
        x2 = tmp[4:,4:]
        mask = ~np.eye(x2.shape[0], dtype=bool)
        x2 = x2[mask].sum()
        within = x1+x2
        
        x1 = tmp[4:,:4]
        mask = ~np.eye(x1.shape[0], dtype=bool)
        x1 = x1[mask].sum()
        x2 = tmp[:4,4:]
        mask = ~np.eye(x2.shape[0], dtype=bool)
        x2 = x2[mask].sum()
        cross = x1+x2
        
        errtrial_crosschunk[i,:,s] = within, cross
errtrial_crosschunk = np.sum(errtrial_crosschunk, axis=2)

# 错误trial总数
err_resp_count = errtrial_crosschunk.sum(axis=1)
#### chunk间错误中减去chunk起始点错误
errtrial_crosschunk[:,[1]] = errtrial_crosschunk[:,[1]] - chunk_start_err


#### 比例 z检验
a1 = errtrial_crosschunk[0,0]  # A组 有效
a2 = errtrial_crosschunk[0,1]   # A组 无效
b1 = errtrial_crosschunk[1,0]   # B组 有效
b2 = errtrial_crosschunk[1,1]   # B组 无效

# 有效数列表
count = [a2, b2]

# 总样本数列表
nobs = [a1 + a2, b1 + b2]

# 执行单尾检验（H1: A组 > B组）
stat, pval = proportions_ztest(count, nobs, alternative='two-sided')

#%% plot chunk内/间 饼图
# 示例数据
dic = {
    'Irr': {'Within': errtrial_crosschunk[0,0]/err_resp_count[0], 
            'Cross': errtrial_crosschunk[0,1]/err_resp_count[0], 
            'Start': chunk_start_err[0,0]/err_resp_count[0]},
    'L1': {'Within': errtrial_crosschunk[1,0]/err_resp_count[1], 
           'Cross': errtrial_crosschunk[1,1]/err_resp_count[1], 
           'Start': chunk_start_err[1,0]/err_resp_count[1]},
    'L2': {'Within': errtrial_crosschunk[2,0]/err_resp_count[2], 
           'Cross': errtrial_crosschunk[2,1]/err_resp_count[2], 
           'Start': chunk_start_err[2,0]/err_resp_count[2]}
}

# 颜色设置
# colors = {
#     'Within': '#ffddaa',
#     'Cross': '#a6bddb',
#     'Start': '#2b8cbe'
# }

colors = {
    'Within': '#6395BC',
    'Cross': '#bdbdbd',
    'Start': '#ffddaa'
}

# 每个扇形的“爆炸”程度
explode = (0.07, 0.07, 0.07)

with plt.style.context(style_path):
    # colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    # fig, axs = plt.subplots(3,1, figsize=(1.6, 3.8), dpi=300)
    fig, axs = plt.subplots(1, 3, figsize=(4.7, 1.8))
    for i, (name, values) in enumerate(dic.items()):
        sizes = [values['Within'], values['Cross'], values['Start']]
        labels = ['Within', 'Cross', 'Start']
        color_list = [colors['Within'], colors['Cross'], colors['Start']]
    
        axs[i].pie(
            sizes,
            labels=labels,
            autopct='%1.1f%%',
            startangle=0,
            explode=explode,
            colors=color_list,
            textprops={'fontsize': 8}
        )
        # axs[i].set_title(f'{name} Proportions', fontsize=8)
        axs[i].axis('equal')  # 保持圆形
    
    # plt.suptitle('2D Pie Charts with Exploded A / b1 / b2 Sections (X, Y, Z)', fontsize=8)
    plt.tight_layout()
    plt.show()
#%% subchunk内/间错误比例 （geo）
from statsmodels.stats.proportion import proportions_ztest

#### 错误item总数
err_item_num = [None]*3
for i in range(3):
    tmp = np.concat(accmatrix_geo[i], axis=2)
    err_item_num[i] = tmp.shape[2]*8 - sum(tmp.sum(2).diagonal())

#### 错误trial总数8*8矩阵
accmatrix_geo_merge = [None]*3
accmatrix_geo_merge[0] = np.concatenate(accmatrix_geo[0],axis=2).sum(2)
accmatrix_geo_merge[1] = np.concatenate(accmatrix_geo[1],axis=2).sum(2)    
accmatrix_geo_merge[2] = np.concatenate(accmatrix_geo[2],axis=2).sum(2)    

#### 1-5/5-1 chunk起始点交换错误
chunk_start_err = np.zeros([3,1])
chunk_start_err[0,0] = accmatrix_geo_merge[0][0,4] + accmatrix_geo_merge[0][4,0]
chunk_start_err[1,0] = accmatrix_geo_merge[1][0,4] + accmatrix_geo_merge[1][4,0]
chunk_start_err[2,0] = accmatrix_geo_merge[2][0,4] + accmatrix_geo_merge[2][4,0]

#### chunk内/chunk间错误
errtrial_crosschunk = np.zeros([3,2,len(subs)])
for i in range(3):
    for s in range(len(subs)):
        tmp = accmatrix_geo[i][s]
        tmp = tmp.sum(2)
        
        x1 = tmp[4:6,4:6]
        mask = ~np.eye(x1.shape[0], dtype=bool)
        x1 = x1[mask].sum()
        x2 = tmp[6:,6:]
        mask = ~np.eye(x2.shape[0], dtype=bool)
        x2 = x2[mask].sum()
        within = x1+x2
        
        x1 = tmp[4:6,6:]
        mask = ~np.eye(x1.shape[0], dtype=bool)
        x1 = x1[mask].sum()
        x2 = tmp[6:,4:6]
        mask = ~np.eye(x2.shape[0], dtype=bool)
        x2 = x2[mask].sum()
        cross = x1+x2
        
        errtrial_crosschunk[i,:,s] = within, cross
errtrial_crosschunk = np.sum(errtrial_crosschunk, axis=2)

# 错误trial总数
err_resp_count = errtrial_crosschunk.sum(axis=1)
#### chunk间错误中减去chunk起始点错误
errtrial_crosschunk[:,[1]] = errtrial_crosschunk[:,[1]] - chunk_start_err


#### 比例 z检验
a1 = errtrial_crosschunk[1,0]  # A组 有效
a2 = errtrial_crosschunk[1,1]   # A组 无效
b1 = errtrial_crosschunk[2,0]   # B组 有效
b2 = errtrial_crosschunk[2,1]   # B组 无效

# 有效数列表
count = [a1, b1]

# 总样本数列表
nobs = [a1 + a2, b1 + b2]

# 执行单尾检验（H1: A组 > B组）
stat, pval = proportions_ztest(count, nobs, alternative='two-sided')

#### plot
dic = {
    'Irr': {'Within': errtrial_crosschunk[0,0]/err_resp_count[0], 
            'Cross': errtrial_crosschunk[0,1]/err_resp_count[0]}, 
    'L1': {'Within': errtrial_crosschunk[1,0]/err_resp_count[1], 
           'Cross': errtrial_crosschunk[1,1]/err_resp_count[1]}, 
    'L2': {'Within': errtrial_crosschunk[2,0]/err_resp_count[2], 
           'Cross': errtrial_crosschunk[2,1]/err_resp_count[2]} 
}

# 颜色设置
colors = {
    'Within': '#ffddaa',
    'Cross': '#a6bddb',
}

# 每个扇形的“爆炸”程度
explode = (0.07, 0.07)

with plt.style.context(style_path):
    # colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axs = plt.subplots(1,3, figsize=(4.5, 2), dpi=300)
    # fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    for i, (name, values) in enumerate(dic.items()):
        sizes = [values['Within'], values['Cross']]
        labels = ['Within', 'Cross']
        color_list = [colors['Within'], colors['Cross']]
    
        axs[i].pie(
            sizes,
            labels=labels,
            autopct='%1.1f%%',
            startangle=90,
            explode=explode,
            colors=color_list,
            textprops={'fontsize': 8}
        )
        # axs[i].set_title(f'{name} Proportions', fontsize=8)
        axs[i].axis('equal')  # 保持圆形
    
    # plt.suptitle('2D Pie Charts with Exploded A / b1 / b2 Sections (X, Y, Z)', fontsize=8)
    plt.tight_layout()
    plt.show()    
#%% geo 错误类型,利用几何相似度，不考虑位置一致性
def to_complex(points):
    """将编号点转为复数坐标（单位圆上的8个点）"""
    return np.array([np.exp(2j * np.pi * (p - 1) / 8) for p in points])

def fourier_descriptor(points, n_des=10):
    """计算傅里叶描述子"""
    z = to_complex(points)
    z_closed = np.append(z, z[0])  # 闭合曲线
    fd = np.fft.fft(z_closed)

    # 去除平移（0阶），归一化尺度（1阶），保留前n个频率
    fd = fd[1:n_des+1] / np.abs(fd[1])
    return fd

def best_aligned_distance(fd1, fd2):
    """在不同相位旋转下，寻找最小描述子距离"""
    min_dist = float('inf')
    for k in range(len(fd1)):
        phase = np.angle(fd1[k]) - np.angle(fd2[k])
        fd2_rot = fd2 * np.exp(1j * phase)
        dist = np.linalg.norm(fd1 - fd2_rot)
        if dist < min_dist:
            min_dist = dist
    return min_dist

def shape_similarity_fd_mirror(points1, points2, n_des=10):
    """考虑镜像后的最大相似度"""
    fd1 = fourier_descriptor(points1, n_des)
    fd2 = fourier_descriptor(points2, n_des)
    fd2_mirror = fourier_descriptor(points2[::-1], n_des)

    dist1 = best_aligned_distance(fd1, fd2)
    dist2 = best_aligned_distance(fd1, fd2_mirror)

    best_dist = min(dist1, dist2)
    return 1 / (1 + best_dist)  # 相似度评分（越大越相似）


def errpattern_geo(cond):
    errPat_allsub = []
    for s in range(len(subs)):
        df = beha[s][cond]
        trialNum = len(np.unique(df['trial'])) - 1
        errPat = np.zeros([trialNum, 9]) + np.nan
        for trial in range(trialNum):
            df['name'] = df['name'].fillna(-99)  # 将'a'列的NaN替换为-99
            idx = df['trial'] == trial + 1
            x = df.loc[idx, 'target'].values
            y = df.loc[idx, 'name'].values.astype(int)
            if np.array_equal(x, y):
                errPat[trial, 0] = 1 # 正确
            elif len(x[x>0])==8 and len(y[y>0])==8:
                A = list(x[:])  # 图形A的四个点编号（1~8）
                B = list(y[:])  # 图形B
                similarity = shape_similarity_fd_mirror(A, B)
                errPat[trial, 3] = similarity
                
                if not np.array_equal(x[:4], y[:4]):
                    # for i in range(5):
                    A = list(x[:4])  # 图形A的四个点编号（1~8）
                    B = list(y[:4])  # 图形B
                    similarity = shape_similarity_fd_mirror(A, B)
                    errPat[trial, 1] = similarity
                if not np.array_equal(x[4:], y[4:]):    
                    A = list(x[4:])  # 图形A的四个点编号（1~8）
                    B = list(y[4:])  # 图形B
                    similarity = shape_similarity_fd_mirror(A, B)
                    errPat[trial, 2] = similarity
        errPat_allsub.append(errPat)
    return errPat_allsub

errPat_allsub = [None]*3
errPat_allsub[0] = errpattern_geo(cond='geo_ir')
errPat_allsub[1] = errpattern_geo(cond='geo_r4')
errPat_allsub[2] = errpattern_geo(cond='geo_r2')

err_count = [None]*3
similarity_allsub = [None]*3
for i in range(3):
    errPat_allsub[i] = np.concatenate(errPat_allsub[i])
    err_count[i] = errPat_allsub[i].sum(0)

np.nanmean(errPat_allsub[0], axis=0)
np.nanmean(errPat_allsub[1], axis=0)
np.nanmean(errPat_allsub[2], axis=0)

stats.ttest_ind(errPat_allsub[0][:,2], errPat_allsub[1][:,2], nan_policy='omit',alternative='less')
stats.ttest_ind(errPat_allsub[0][:,2], errPat_allsub[2][:,2], nan_policy='omit',alternative='less')
#%% geo 错误类型,利用几何相似度，考虑位置一致性，20250711采用
def to_complex(points):
    """将编号点转为复数坐标（单位圆上的8个点）"""
    return np.array([np.exp(2j * np.pi * (p - 1) / 8) for p in points])

def fourier_descriptor(points, n_des=10):
    """计算傅里叶描述子（闭合曲线）"""
    z = to_complex(points)
    z_closed = np.append(z, z[0])
    fd = np.fft.fft(z_closed)
    fd = fd[1:n_des+1] / np.abs(fd[1])  # 去除平移，归一化尺度
    return fd

def best_aligned_distance(fd1, fd2):
    """在不同相位旋转下，寻找最小描述子距离"""
    min_dist = float('inf')
    best_phase = 0
    for k in range(len(fd1)):
        phase = np.angle(fd1[k]) - np.angle(fd2[k])
        fd2_rot = fd2 * np.exp(1j * phase)
        dist = np.linalg.norm(fd1 - fd2_rot)
        if dist < min_dist:
            min_dist = dist
            best_phase = phase
    return min_dist, best_phase

def shape_similarity_with_startpoint(points1, points2, n_des=10, alpha=0.5):
    """考虑起始点位置的 Fourier 相似度"""
    fd1 = fourier_descriptor(points1, n_des)
    fd2 = fourier_descriptor(points2, n_des)
    fd2_mirror = fourier_descriptor(points2[::-1], n_des)

    d1, phase1 = best_aligned_distance(fd1, fd2)
    d2, phase2 = best_aligned_distance(fd1, fd2_mirror)
    best_dist = min(d1, d2)
    use_mirror = d2 < d1
    best_points2 = points2[::-1] if use_mirror else points2

    # -------------------
    # 起点角度惩罚
    theta1 = (points1[0] - 1) * np.pi / 4
    theta2 = (points2[0] - 1) * np.pi / 4
    angle_diff = np.abs(np.angle(np.exp(1j * (theta1 - theta2))))
    position_penalty = alpha * np.sin(angle_diff / 2)
    # -------------------

    total_score = 1 / (1 + best_dist + position_penalty)
    return total_score

def shape_similarity_with_rotation_penalty(points1, points2, n_des=10, alpha=0.25):
    """考虑旋转角度惩罚的形状相似度"""
    fd1 = fourier_descriptor(points1, n_des)
    fd2 = fourier_descriptor(points2, n_des)
    fd2_mirror = fourier_descriptor(points2[::-1], n_des)

    d1, phase1 = best_aligned_distance(fd1, fd2)
    d2, phase2 = best_aligned_distance(fd1, fd2_mirror)

    if d1 <= d2:
        best_dist = d1
        rotation_penalty = alpha * np.abs(np.sin(phase1 / 2))
    else:
        best_dist = d2
        rotation_penalty = alpha * np.abs(np.sin(phase1 / 2))

    score = 1 / (1 + best_dist + rotation_penalty)
    return score

def errpattern_geo(cond):
    errPat_allsub = []
    for s in range(len(subs)):
        df = beha[s][cond]
        trialNum = len(np.unique(df['trial'])) - 1
        errPat = np.zeros([trialNum, 10]) + np.nan
        errPat[:,9] = s+1 # sub 编号
        for trial in range(trialNum):
            df['name'] = df['name'].fillna(-99)  # 将'a'列的NaN替换为-99
            idx = df['trial'] == trial + 1
            x = df.loc[idx, 'target'].values
            y = df.loc[idx, 'name'].values.astype(int)
            if np.array_equal(x, y):
                errPat[trial, 0] = 1 # 正确
            elif len(x[x>0])==8 and len(y[y>0])==8:
                A = list(x[:])  # 图形A的四个点编号（1~8）
                B = list(y[:])  # 图形B
                similarity = shape_similarity_with_rotation_penalty(A, B)
                errPat[trial, 3] = similarity
                
                if not np.array_equal(x[:4], y[:4]):
                    # for i in range(5):
                    A = list(x[:4])  # 图形A的四个点编号（1~8）
                    B = list(y[:4])  # 图形B
                    similarity = shape_similarity_with_rotation_penalty(A, B)
                    errPat[trial, 1] = similarity
                if not np.array_equal(x[4:], y[4:]):    
                    A = list(x[4:])  # 图形A的四个点编号（1~8）
                    B = list(y[4:])  # 图形B
                    similarity = shape_similarity_with_rotation_penalty(A, B)
                    errPat[trial, 2] = similarity
        errPat_allsub.append(errPat)
    return errPat_allsub

errPat_allsub = [None]*3
errPat_allsub[0] = errpattern_geo(cond='geo_ir')
errPat_allsub[1] = errpattern_geo(cond='geo_r4')
errPat_allsub[2] = errpattern_geo(cond='geo_r2')

# simindx_mean = np.zeros([len(subs), 3])
# for i in range(len(subs)):
#     tmp = errPat_allsub[1][i]
#     tmp[tmp[:,0]==1, 3] = 1
#     tmp = np.nanmean(np.nanmean(tmp[:,3], axis=0))
#     simindx_mean[i,2] = tmp
    
# err_count = [None]*3
# similarity_allsub = [None]*3
# for i in range(3):
#     errPat_allsub[i] = np.concatenate(errPat_allsub[i])
#     err_count[i] = errPat_allsub[i].sum(0)

# np.nanmean(errPat_allsub[0], axis=0)
# np.nanmean(errPat_allsub[1], axis=0)
# np.nanmean(errPat_allsub[2], axis=0)

# testid = 3 # 检验4+4整体相似度指标
# stats.ttest_ind(errPat_allsub[0][:,testid], errPat_allsub[1][:,testid], nan_policy='omit',alternative='less')
# stats.ttest_ind(errPat_allsub[0][:,testid], errPat_allsub[2][:,testid], nan_policy='omit',alternative='less')

# testid = 1
# stats.ttest_ind(errPat_allsub[0][:,testid], errPat_allsub[1][:,testid], nan_policy='omit',alternative='less')

#%% plot 几何错误类型的example
import itertools

A = [1, 2, 5, 6]       # 图形A
B = [6, 5, 2, 1]       # 图形B（是A的镜像）
B_perm = list(itertools.permutations(B))
similarity = [None]*len(B_perm)

for p in range(len(B_perm)):
    similarity[p] = shape_similarity_with_rotation_penalty(A, B_perm[p])
sort_idx = np.argsort(similarity)
    
with plt.style.context(style_path):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axs = plt.subplots(1,1, figsize=(3,1.5), dpi=300) 
    df = pd.DataFrame({'sim': similarity})
    
    # 获取唯一值作为x轴类别
    unique_vals = sorted(df['sim'].unique())
    # 建立映射：每个唯一值 → x轴整数位置
    val_to_x = {val: i for i, val in enumerate(unique_vals)}
    # 生成 x 坐标
    x = df['sim'].map(val_to_x)
    # 为了防止重叠，加一点抖动（可选）
    x_jittered = x + np.random.normal(scale=0.01, size=len(x))
    df['sim'] = df['sim'] + np.random.normal(scale=0.01, size=len(x))
    axs.scatter(x_jittered, df['sim'], color='k', edgecolor=[], alpha=0.7)
    axs.set_ylim([0, 1.2])
    axs.set_xticks([])
    axs.set_yticks([0, 0.4, 0.8, 1.2])
    axs.spines['bottom'].set_visible(False)
    axs.spines['left'].set_color('gray')
    # axs.spines['left'].set_visible(False)
    
    # 仅保留左边的 y 轴
    axs.spines['left'].set_linewidth(1)
    axs.tick_params(axis='y', direction='out', length=3, width=1.5, colors='gray', labelsize=10)
    
    # y轴美化
    axs.yaxis.label.set_color('gray')

    
with plt.style.context(style_path):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axs = plt.subplots(2,4, figsize=(4,2), dpi=300) 
    err_temp = []
    k=0
    # for i in [j for j in np.arange(12)]+[13,14,16,17]:#,[12,15,18,19,20,21,22]: #range(len(sort_idx))
    for i in [12,15,18,19,20,21,22]: #range(len(sort_idx))
        a = np.array([3,4,7,8] + A)
        b = np.array([3,4,7,8] + list(B_perm[sort_idx[i]]))
        x = np.zeros([8, 8])
        for j in range(8):
            idx = a[j] == b
            if sum(idx)>0:
                x[j,idx] = 1
        err_temp.append(x)
                
        row = k // 4
        col = k % 4
        ax = axs[row, col]
        h1 = ax.pcolormesh(x, vmin=0, vmax=0.3) # 
        ax.invert_yaxis()
        ax.axis('off')
        # ax.set_title(str(np.round(similarity[sort_idx[i]], 2)))
        k+=1
    plt.tight_layout()
    
#%% geo L1和L2中和错误模板对比，计算整体交换错误比例
def err_match_temp(cond,accmat):
    errPat_allsub = []
    for s in range(len(subs)):
        df = beha[s][cond]
        trialNum = len(np.unique(df['trial'])) - 1
        errPat = np.zeros([trialNum, 8])
        for trial in range(trialNum):
            df['name'] = df['name'].fillna(-99)  # 将'a'列的NaN替换为-99
            idx = df['trial'] == trial + 1
            x = df.loc[idx, 'target'].values
            y = df.loc[idx, 'name'].values.astype(int)
            diff_positions = [i+1 for i in range(8) if x[i] != y[i]]  # 记录不一致的位置（1-based索引）
            if np.array_equal(x, y):
                errPat[trial, 0] = 1 # 正确
            else:
                xx = accmat[s][:,:,trial]
                lag1 = check_matrix(xx[:4,:4], err_temp_2)
                lag2 = check_matrix(xx[4:,4:], err_temp_2)
                if lag1==1 or lag2==1:
                    errPat[trial, 1]=1
                else:
                    errPat[trial, 2]=1
                
        errPat_allsub.append(errPat)
    return errPat_allsub

def check_matrix(x, y):
    for mat in y:
        if np.array_equal(x, mat[4:,4:]):
        # if np.array_equal(x, mat[]):
            return 1
    return 0

# err_temp_2 = err_temp[10:22]
# err_temp_2 = [err_temp[i] for i in [12,15,18,19,20,21,22]]
err_temp_2 = err_temp
errPat_allsub = [None]*3
errPat_allsub[0] = err_match_temp(cond='geo_ir', accmat=accmatrix_geo[0])
errPat_allsub[1] = err_match_temp(cond='geo_r4', accmat=accmatrix_geo[1])
errPat_allsub[2] = err_match_temp(cond='geo_r2', accmat=accmatrix_geo[2])

errPat_geo4_err6 = np.concat(errPat_allsub[1], axis=0)
errPat_geo4_err6count = errPat_geo4_err6.sum(0)

errPat_geo2_err6 = np.concat(errPat_allsub[2], axis=0)
errPat_geo2_err6count = errPat_geo2_err6.sum(0)

#### 比例 z检验
a1 = errPat_geo4_err6count[1]  # A组 有效
a2 = errPat_geo4_err6count[2]   # A组 无效
b1 = errPat_geo2_err6count[1]   # B组 有效
b2 = errPat_geo2_err6count[2]   # B组 无效

# 有效数列表
count = [a1, b1]

# 总样本数列表
nobs = [a1 + a2, b1 + b2]

# 执行单尾检验（H1: A组 > B组）
stat, pval = proportions_ztest(count, nobs, alternative='smaller') # two-sided


#### plot
dic = {
    'L1': {'Str.Err': a1/(a1+a2), 
           'Rnd.Err': a2/(a1+a2)}, 
    'L2': {'Str.Err': b1/(b1+b2), 
           'Rnd.Err': b2/(b1+b2)} 
}

# 颜色设置
colors = {
    'Str.Err': '#B76A66',
    'Rnd.Err': '#bdbdbd',
}

# 每个扇形的“爆炸”程度
explode = (0.07, 0.07)

with plt.style.context(style_path):
    # colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axs = plt.subplots(1,2, figsize=(2.3, 1.8), dpi=300)
    # fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    for i, (name, values) in enumerate(dic.items()):
        sizes = [values['Str.Err'], values['Rnd.Err']]
        labels = [' ', ' ']
        color_list = [colors['Str.Err'], colors['Rnd.Err']]
    
        axs[i].pie(
            sizes,
            labels=labels,
            autopct='%1.1f%%',
            startangle = 90,
            explode=explode,
            colors=color_list,
            textprops={'fontsize': 8}
        )
        # axs[i].set_title(f'{name} Proportions', fontsize=8)
        axs[i].axis('equal')  # 保持圆形
    
    # plt.suptitle('2D Pie Charts with Exploded A / b1 / b2 Sections (X, Y, Z)', fontsize=8)
    plt.tight_layout()
    plt.show() 






