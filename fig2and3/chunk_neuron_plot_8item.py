# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 15:44:04 2024

@author: Wen
"""

get_ipython().run_line_magic('reset', '-sf')
style_path = r'C:\Wen\OneDrive\CodeHub\Python\style_paper.mplstyle'

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
# import mat73
import pickle as pkl
import scipy.io as sio
import sys

from scipy import stats
# from matplotlib_venn import venn3, venn2 #记得安装matplotlib_venn(pip install matplotlib_venn 或者conda install matplotlib_venn)
from matplotlib.patches import Rectangle
import seaborn as sns

sys.path.append(r'C:\Wen\OneDrive\CodeHub\Python\Neural_analysis_tools')
import zscore4fr
#%% function
def stats_of_alltime(x, base_win, n):
    # base = x[n,tidx,:][:,base_win].mean(1)
    base = x[n,:,:mk_ts[3]].mean(1) # 用3种任务中所有trial在整个刺激呈现阶段的活动作为baseline
    # base = x[n,:,base_win].mean(1) # 用3种任务中所有trial在整个刺激呈现阶段的活动作为baseline
    xx = x[n,tidx,:]
    pval = np.zeros([2, xx.shape[1]])
    for t in range(xx.shape[1]):
        xxx = xx[:,t].mean(0)
        # pval[0,t], pval[1,t] = stats.ttest_rel(xxx, base, alternative='greater')
        # pval[0,t], pval[1,t] = stats.ttest_ind(xxx, base, alternative='greater')
        pval[0,t], pval[1,t] = stats.ttest_1samp(base, xxx, alternative='less')
    return pval

def stats_of_alltime2(x, base_win, n):
    # base = x[n,tidx,:][:,base_win].mean(1)
    # base = x[n,tidx,:mk_ts[3]].mean(1) # 用3种任务中所有trial在整个刺激呈现阶段的活动作为baseline
    # base = x[n,tidx,:][:,:].mean(0) # 用3种任务中所有trial在整个刺激呈现阶段的活动作为baseline
    xx = x[n,tidx,:]
    pval = np.zeros([2, xx.shape[1]])
    for t in range(xx.shape[1]):
        base_win = np.setdiff1d(np.arange(0, 95), t)
        base = x[n,tidx,:][:,base_win].mean(0)
        xxx = xx[:,t].mean(0) 
        # pval[0,t], pval[1,t] = stats.ttest_rel(xxx, base, alternative='greater')
        # pval[0,t], pval[1,t] = stats.ttest_ind(xxx, base, alternative='greater')
        pval[0,t], pval[1,t] = stats.ttest_1samp(base, xxx, alternative='less')
    return pval

def stats_of_alltime_Fordrift(x, base_win, n, tidx):
    xx = x[n,tidx,:]
    pval = np.zeros([2, xx.shape[1]])+np.nan
    for t in range(95):
        base_win = np.setdiff1d(np.arange(0, 130), t)
        base = x[n,tidx,:][:,base_win].mean(0)
        xxx = xx[:,t].mean(0) 
        pval[0,t], pval[1,t] = stats.ttest_1samp(base, xxx, alternative='less')
    return pval

def sigtime(x):
    sig = np.zeros([1, x.shape[1]])
    # idx = (x[0,:]>0) & (x[1,:]<0.05)
    idx = x[1,:]<0.05
    sig[0,idx] = 1
    # pval = x
    # sig = pval.copy()
    # sig[0,:] = 0
    # sig[0,pval[0,:]>0] = 1
    # sig[1,:] = 0
    # sig[1,pval[1,:]<0.05] = 1
    # sig = sig[[0],:]*sig[[1],:]
    return sig

def find_succe_sig_time(x):
    # x: channel * time 
    window = 3
    pvalmask = x
    pvalmask = pvalmask.T # 转换成 Time*其他
    pev_sig = pd.DataFrame(pvalmask).rolling(window=window).sum()
    c, r = np.where(pev_sig==window)
    # c = c - window + 1
    pev_sig = np.zeros(pev_sig.shape).T
    for i in range(len(r)):
        pev_sig[r[i], c[i]-window+1:c[i]+1] = 1
    return pev_sig
#%% load FR data
path = r'F:\Human_chunking_Huashan_intraoperative\Data\Recording\PFC'
# data = pkl.load(open(path+'\\fr.pkl', 'rb'))
# data = pkl.load(open(path+'\\fr_PFC(8sub)_479.pkl', 'rb'))
# data = pkl.load(open(path+'\\fr_beha_PFC(9sub_153).pkl', 'rb'))
# data = pkl.load(open(path+'\\fr_beha_PFC(9sub_geo_254).pkl', 'rb'))
# data = pkl.load(open(path+'\\fr_beha_PFC(9sub_155)_drift.pkl', 'rb'))
# data = pkl.load(open(path+'\\fr_beha_PFC(9sub_155)_errormark_drift.pkl', 'rb'))
# data = pkl.load(open(path+'\\fr_beha_PFC(9sub_155)_errormark_drift_new.pkl', 'rb'))
data = pkl.load(open(path+'\\fr_beha_PFC(10sub_160)_errormark_drift_new2.pkl', 'rb'))

path = r'F:\Human_chunking_Huashan_intraoperative\Data\Recording\motor'
data = pkl.load(open(path+'\\fr_beha_motor(2sub_67)_errormark_drift.pkl', 'rb'))

fr = data['fr']
neu_info = data['neu_info']
trialInfo = data['trialInfo']
window = data['window']
step = data['step']
subs= data['subs']
mk_ts = data['mk_ts']

aline_ts = int(window/step/2)
neuNum = sum([len(i) for i in neu_info])
#%% test sig time, compare to baseline
for bdidx in range(8):
    print(bdidx)
    mk_ts = np.array([10, 50, 90, 130])# - aline_ts
    # condition = ['L_Rand','L_Regular','G_Rand','G_Regular4','G_Regular2','M_Rand','M_Regular4','M_Regular2']
    condition = ['lan_ir', 'lan_r','geo_ir', 'geo_r4', 'geo_r2', 'mus_ir', 'mus_r4', 'mus_r2']
    sig_allneu = np.zeros([neuNum, 8, mk_ts[3]])
    k=-1
    
    # [5,15],[15,25],[25,35],[35,45]
    ts_test = [mk_ts[0]+bdidx*10+8, mk_ts[0]+bdidx*10+13] # 4+4 boundary
    # ts_test = [mk_ts[1]+35, mk_ts[1]+40] # 4+4 boundary
    # ts_base = [0, mk_ts[2]]
    test_win = np.arange(ts_test[0], ts_test[1])
    base_win = np.setdiff1d(np.arange(0, mk_ts[2]), test_win)
    
    # # ts_test = [mk_ts[0]-2, mk_ts[0]+2] # start
    # ts_test = [mk_ts[2]-5, mk_ts[2]] # end
    # ts_base = [0, mk_ts[2]]
    # test_win = np.arange(ts_test[0], ts_test[1])
    # base_win = np.setdiff1d(np.arange(0, mk_ts[2]), test_win)
    
    # ts_test = [mk_ts[0]+15, mk_ts[0]+20] # 2+2 boundary
    # ts_test2 = [mk_ts[1]+15, mk_ts[1]+20] # 2+2 boundary
    # test_win1 = np.arange(ts_test[0], ts_test[1])
    # test_win2 = np.arange(ts_test2[0], ts_test2[1])
    # ts_base = [0, mk_ts[2]]
    # test_win = np.concatenate([test_win1, test_win2])
    # base_win = np.setdiff1d(np.arange(0, mk_ts[2]), test_win)
    
    for s in range(len(subs)):
        for i in condition:
            neu_info[s]['b_'+i] = 0
        fr1 = fr[s][:,:,:mk_ts[3]]
        if fr1.shape[0] == 0:
            continue
        #### 对FR标准化
        fr1 = zscore4fr.zscore3d(fr1)
        fr[s][:,:,:mk_ts[3]] = fr1
        ####
        for n in range(fr1.shape[0]):
            k += 1
            for c in range(len(condition)):
                tidx = trialInfo[s]['task'].str.contains(condition[c])
                # pval = stats_of_alltime(fr1, base_win, n)
                # pval = stats_of_alltime2(fr1, base_win, n)
                pval = stats_of_alltime_Fordrift(fr1, base_win, n,tidx)
                sig = sigtime(pval)
                sig = find_succe_sig_time(sig)
                sig_allneu[k,c,:] = sig
            
                tmp = sig[0, test_win].sum() # around b
                # tmp2 = sig[0, test_win2].sum() # around b
                if tmp > 5: # len(test_win)-2:# and tmp2 >0:
                # if tmp > 0 and tmp2 >0:
                    neu_info[s].loc[n, 'b_'+condition[c]] = 1
    
    df = pd.concat(neu_info, axis=0, ignore_index=True)
    df['neuID_allsub'] = np.arange(len(df))
    # df.sum()
    
    saveInfo = {'subs':subs, 'window':window, 'step':step}
    path = r'F:\Human_chunking_Huashan_intraoperative\Result\start_boundary_end_neuron'
    # savefile = path+'\\PFC_9sub_155_bd5('+str(bdidx+1)+').pkl'
    # saveInfo = dict(saveInfo, **{'sig_allneu':sig_allneu, 'neu_info':neu_info,'df':df})
    # pkl.dump(saveInfo, open(savefile,'wb'))
    

#%% 多核并行 test sig time
from joblib import Parallel, delayed
# condition = ['lan_ir', 'lan_r','geo_ir', 'geo_r4', 'geo_r2', 'mus_ir', 'mus_r4', 'mus_r2']

def one_run(bdidx):
    # print(bdidx)
    dataend = 130
    mk_ts = np.array([10, 50, 90, 130])# - aline_ts
    # condition = ['L_Rand','L_Regular','G_Rand','G_Regular4','G_Regular2','M_Rand','M_Regular4','M_Regular2']
    condition = ['lan_ir', 'lan_r','geo_ir', 'geo_r4', 'geo_r2', 'mus_ir', 'mus_r4', 'mus_r2']
    sig_allneu = np.zeros([neuNum, 8, dataend])
    k=-1
    
    # [5,15],[15,25],[25,35],[35,45]
    ts_test = [mk_ts[0]+bdidx*10+4, mk_ts[0]+bdidx*10+15] # 4+4 boundary
    # ts_test = [mk_ts[1]+35, mk_ts[1]+40] # 4+4 boundary
    # ts_base = [0, mk_ts[2]]
    test_win = np.arange(ts_test[0], ts_test[1])
    base_win = np.setdiff1d(np.arange(0, dataend), test_win)
    
    # # ts_test = [mk_ts[0]-2, mk_ts[0]+2] # start
    # ts_test = [mk_ts[2]-5, mk_ts[2]] # end
    # ts_base = [0, mk_ts[2]]
    # test_win = np.arange(ts_test[0], ts_test[1])
    # base_win = np.setdiff1d(np.arange(0, mk_ts[2]), test_win)
    
    # ts_test = [mk_ts[0]+15, mk_ts[0]+20] # 2+2 boundary
    # ts_test2 = [mk_ts[1]+15, mk_ts[1]+20] # 2+2 boundary
    # test_win1 = np.arange(ts_test[0], ts_test[1])
    # test_win2 = np.arange(ts_test2[0], ts_test2[1])
    # ts_base = [0, mk_ts[2]]
    # test_win = np.concatenate([test_win1, test_win2])
    # base_win = np.setdiff1d(np.arange(0, mk_ts[2]), test_win)
    
    for s in range(len(subs)):
        for i in condition:
            neu_info[s]['b_'+i] = 0
        fr1 = fr[s][:,:,:dataend].copy()
        
        if fr1.shape[0] == 0:
            continue
        # #### 对FR标准化
        fr1 = zscore4fr.zscore3d_minmax(fr1)
        # fr[s][:,:,:mk_ts[3]] = fr1
        
        ####
        for n in range(fr1.shape[0]):
            k += 1
            for c in [2,3,4]: #range(len(condition))
                tidx = (trialInfo[s]['task'].str.contains(condition[c])) & (trialInfo[s]['acc']!=2)
                # pval = stats_of_alltime(fr1, base_win, n)
                # pval = stats_of_alltime2(fr1, base_win, n)
                pval = stats_of_alltime_Fordrift(fr1, base_win, n, tidx)
                sig = sigtime(pval)
                sig = find_succe_sig_time(sig)
                sig_allneu[k,c,:] = sig
            
                tmp = sig[0, test_win].sum() # around b
                # tmp2 = sig[0, test_win2].sum() # around b
                if tmp > 5: # len(test_win)-2:# and tmp2 >0:
                # if tmp > 0 and tmp2 >0:
                    neu_info[s].loc[n, 'b_'+condition[c]] = 1
    
    df = pd.concat(neu_info, axis=0, ignore_index=True)
    df['neuID_allsub'] = np.arange(len(df))
    # df.sum()
    
    saveInfo = {'subs':subs, 'window':window, 'step':step}
    path = r'F:\Human_chunking_Huashan_intraoperative\Result\start_boundary_end_neuron'
    
    # savefile = path+'\\PFC_9sub_155_bd6('+str(bdidx+1)+').pkl'
    # savefile = path+'\\PFC_10sub_167_bd_new('+str(bdidx+1)+').pkl'
    # saveInfo = dict(saveInfo, **{'sig_allneu':sig_allneu, 'neu_info':neu_info,'df':df})
    # pkl.dump(saveInfo, open(savefile,'wb'))
    
    savefile = path+'\\Motor_2sub_67_bd3_new('+str(bdidx+1)+').pkl'
    saveInfo = dict(saveInfo, **{'sig_allneu':sig_allneu, 'neu_info':neu_info,'df':df})
    pkl.dump(saveInfo, open(savefile,'wb'))
    
results_all = Parallel(n_jobs=-1)(delayed(one_run)(bdidx) for bdidx in range(8))
 #%%
count = [df.sum()['b_M_Rand'], df.sum()['b_M_Regular4']]  # 两组阳性例数
nobs = [len(df), len(df)]    # 两组总样本量
from statsmodels.stats.proportion import proportions_ztest
z, p = proportions_ztest(count, nobs)
print(z,p)
#%% save boundary neuron 
# saveInfo = {'subs':subs, 'window':window, 'step':step}
# path = r'F:\Human_chunking_Huashan_intraoperative\Result\start_boundary_end_neuron'
# savefile = path+'\\PFC_9sub_155_bd(5).pkl'
# # savefile = path+'\\sub(frontal)_subb_compareTobaseline.pkl'
# saveInfo = dict(saveInfo, **{'sig_allneu':sig_allneu, 'neu_info':neu_info,'df':df})
# pkl.dump(saveInfo, open(savefile,'wb'))
#%% load results, 8个时间段
def load_bd(name):
    path = r'F:\Human_chunking_Huashan_intraoperative\Result\start_boundary_end_neuron'
    file = path + '\\' + name + '.pkl'
    data = pkl.load(open(file,'rb'))
    df = data['df']
    sig_allneu = data['sig_allneu']
    neu_info = data['neu_info']
    return df, sig_allneu

# names = ['PFC_bd(1)_8sub(145)', 'PFC_bd(2)_8sub(145)', 'PFC_bd(3)_8sub(145)', 'PFC_bd(4)_8sub(145)', 
#          'PFC_bd(5)_8sub(145)','PFC_bd(6)_8sub(145)','PFC_bd(7)_8sub(145)','PFC_bd(8)_8sub(145)']

# names = ['PFC_9sub_155_bd4(1)', 'PFC_9sub_155_bd4(2)', 'PFC_9sub_155_bd4(3)', 'PFC_9sub_155_bd4(4)', 
#          'PFC_9sub_155_bd4(5)','PFC_9sub_155_bd4(6)','PFC_9sub_155_bd4(7)','PFC_9sub_155_bd4(8)']

# names = ['PFC_9sub_155_bd6(1)', 'PFC_9sub_155_bd6(2)', 'PFC_9sub_155_bd6(3)', 'PFC_9sub_155_bd6(4)', 
#          'PFC_9sub_155_bd6(5)','PFC_9sub_155_bd6(6)','PFC_9sub_155_bd6(7)','PFC_9sub_155_bd6(8)'] # 采用

# names = ['PFC_9sub_155_bd_new(1)', 'PFC_9sub_155_bd_new(2)', 'PFC_9sub_155_bd_new(3)', 'PFC_9sub_155_bd_new(4)', 
#          'PFC_9sub_155_bd_new(5)','PFC_9sub_155_bd_new(6)','PFC_9sub_155_bd_new(7)','PFC_9sub_155_bd_new(8)']

# names = ['PFC_10sub_167_bd_new(1)', 'PFC_10sub_167_bd_new(2)', 'PFC_10sub_167_bd_new(3)', 'PFC_10sub_167_bd_new(4)', 
#          'PFC_10sub_167_bd_new(5)','PFC_10sub_167_bd_new(6)','PFC_10sub_167_bd_new(7)','PFC_10sub_167_bd_new(8)']

# names = ['Motor_2sub_67_bd3(1)', 'Motor_2sub_67_bd3(2)', 'Motor_2sub_67_bd3(3)', 'Motor_2sub_67_bd3(4)', 
#          'Motor_2sub_67_bd3(5)','Motor_2sub_67_bd3(6)','Motor_2sub_67_bd3(7)','Motor_2sub_67_bd3(8)']

names = ['Motor_2sub_67_bd3_new(1)', 'Motor_2sub_67_bd3_new(2)', 'Motor_2sub_67_bd3_new(3)', 'Motor_2sub_67_bd3_new(4)', 
         'Motor_2sub_67_bd3_new(5)','Motor_2sub_67_bd3_new(6)','Motor_2sub_67_bd3_new(7)','Motor_2sub_67_bd3_new(8)']

# condition = ['L_Rand','L_Regular','G_Rand','G_Regular4','G_Regular2','M_Rand','M_Regular4','M_Regular2']
condition = ['lan_ir', 'lan_r','geo_ir', 'geo_r4', 'geo_r2', 'mus_ir', 'mus_r4', 'mus_r2']
sig_mat_alltime = []
for i in range(8): # 8个时间段
    df, sig_allneu = load_bd(names[i]) # 读取8个时间段中的一个
    tmp = np.zeros([8, sig_allneu.shape[2]])
    for c in range(8): # 8种任务
        idx = df['b_'+condition[c]]==1
        tmp[c,:] = sig_allneu[idx,c,:].sum(0)/len(df)
    sig_mat_alltime.append(tmp)
#%% plot 8个时间段
from cycler import cycler
from scipy.ndimage import gaussian_filter1d

with plt.style.context(style_path):
    get_ipython().run_line_magic('matplotlib', 'inline')
    fig, axs = plt.subplots(1, 3, figsize=(5.8, 1.5), dpi=300, constrained_layout=True)
    # fig, axs = plt.subplots(3, 1, figsize=(1.7, 4.8), dpi=300, constrained_layout=True)
    # colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    cmap = plt.get_cmap('rainbow')
    # colors = [cmap(i/7) for i in range(8)][::-1]  # 除以7确保均匀分布在0-1之间

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    cc1 = ['gray']*7+['k']
    cc2 = ['gray']*3 + colors[0:1] + ['gray']*3+['k']
    cc3 = ['gray',colors[1],'gray',colors[0],'gray',colors[1],'gray','k']
        
    
    sig_rand_max = np.zeros([1,8])
    sig_L1_max = np.zeros([1,8])
    sig_L2_max = np.zeros([1,8])
    for i in range(8):
        sig_rand = sig_mat_alltime[i][[2],:].mean(0)[10:95]
        sig_L1 = sig_mat_alltime[i][[3],:].mean(0)[10:95]
        sig_L2 = sig_mat_alltime[i][[4],:].mean(0)[10:95]
        
        sig_rand_max[0,i] = sig_rand[i*10+5:i*10+15].mean()
        sig_L1_max[0,i] = sig_L1[i*10+5:i*10+15].mean()
        sig_L2_max[0,i] = sig_L2[i*10+5:i*10+15].mean()
        
        sig_rand = gaussian_filter1d(sig_rand, sigma=1.5)
        sig_L1 = gaussian_filter1d(sig_L1, sigma=1.5)
        sig_L2 = gaussian_filter1d(sig_L2, sigma=1.5)
        axs[0].plot(sig_rand.T, color=cc1[i], alpha=1)
        axs[1].plot(sig_L1.T,color=cc2[i],alpha=1)
        axs[2].plot(sig_L2.T,color=cc3[i],alpha=1)
        
    title = ['L0 task', 'L1 task', 'L2 task']
    for n in range(3):
        for tt in range(8):
            axs[n].add_patch(Rectangle((tt*10, -1.2), 5, 6, edgecolor=None, facecolor="gray", alpha=0.25))
        axs[n].set_title(title[n])
        axs[n].vlines(np.array([20, 40, 60]), 0, 0.45, colors='w', linestyle='--',linewidth=1)  
        axs[n].set_xticks(np.arange(0,80,10), [str(ii) for ii in range(1,9)])
        ymax = 0.45
        axs[n].set_ylim([0, ymax+0.1])
        if n==0:
            axs[n].set_xlabel('Rank')
            axs[n].set_ylabel('Proportion')
        else:
            axs[n].set_xlabel(' ')
            axs[n].set_ylabel(' ')
        # axs[n].vlines(np.arange(0,90,10), 0, 0.45, linestyle='--', color='gray', alpha=0.5)
        # axs[n].vlines(np.arange(0,90,10), 0, 0.45, linestyle='--', color='gray', alpha=0.5)
        axs[n].vlines(np.array([20, 40, 60]), 0, ymax+0.1, linestyle='--',linewidth=1.5, color='gray')  
        # axs[n].vlines(np.array([10,20,30,50,60,70]), 0, ymax, linestyle='--',linewidth=1, color='gray', alpha=0.5) 
        axs[n].set_xlim([-3,88])
        
        # for j in range(8):
            # axs[n].add_patch(Rectangle(xy=(j*10+10, -0.01), width=5, height=0.5, 
            #                        edgecolor=None, facecolor="gray", alpha=0.2)) 
        
#%% plot 8个时间段, 百分比差异  
from statsmodels.stats.proportion import proportions_ztest
 
with plt.style.context(style_path):
    fig, axs = plt.subplots(1, 1, figsize=(1.8, 1.5), dpi=300, constrained_layout=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    # cmap = plt.get_cmap('rainbow')
    # colors = [cmap(i/7) for i in range(8)][::-1]  # 除以7确保均匀分布在0-1之间
    
    # l1_rand = sig_L1_max - sig_rand_max
    # l2_rand = sig_L2_max - sig_rand_max
    l2_l1 = sig_L1_max - sig_rand_max
    
    cc2 = ['gray']*3 + colors[0:1] + ['gray']*3+['k']
    cc3 = ['gray',colors[1],'gray',colors[0],'gray',colors[1],'gray','gray']
    
    # axs.plot(np.arange(1,9), l2_l1[0,:], color='gray') 
    axs.bar(np.arange(1,9), l2_l1[0,:], color=cc3) 
    
    axs.set_xticks(np.arange(1,9))
    axs.set_ylim([-0.09, 0.23])
    axs.set_xlabel('Rank')
    axs.set_ylabel('L1 - L0')
    for j in range(8):
        axs.add_patch(Rectangle(xy=(j*10+8, -0.01), width=5, height=0.38, 
                                   edgecolor=None, facecolor="gray", alpha=0.02)) 

tmp = l2_l1*neuNum
tmp = tmp[0,:]
n_A, k_A = neuNum, tmp[3]  # A组：100人做题，70人正确
n_B, k_B = neuNum, (tmp[0]+tmp[2]+tmp[4])/3  # B组：120人做题，95人正确
# --- 方法1：两比例z检验 ---
count = np.array([k_A, k_B])          # 成功数
nobs = np.array([n_A, n_B])           # 总样本数
proportions_ztest(count, nobs)

####
with plt.style.context(style_path):
    fig, axs = plt.subplots(1, 1, figsize=(1.8, 1.5), dpi=300, constrained_layout=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    l2_l1 = sig_L2_max - sig_rand_max
    
    cc2 = ['gray']*3 + colors[0:1] + ['gray']*3+['k']
    cc3 = ['gray',colors[1],'gray',colors[0],'gray',colors[1],'gray','gray']
    
    # axs.plot(np.arange(1,9), l2_l1[0,:], color='gray') 
    axs.bar(np.arange(1,9), l2_l1[0,:], color=cc3) 
    
    axs.set_xticks(np.arange(1,9))
    axs.set_ylim([-0.09, 0.23])
    axs.set_xlabel('Rank')
    axs.set_ylabel('L2 - L0')
    for j in range(8):
        axs.add_patch(Rectangle(xy=(j*10+8, -0.01), width=5, height=0.38, 
                                   edgecolor=None, facecolor="gray", alpha=0.02))     

tmp = l2_l1*neuNum
tmp = tmp[0,:]
n_A, k_A = neuNum, tmp[3]  # tmp[1], tmp[3], tmp[5]   
n_B, k_B = neuNum, (tmp[0]+tmp[2]+tmp[4])/3  # B组：120人做题，95人正确
# --- 方法1：两比例z检验 ---
count = np.array([k_A, k_B])          # 成功数
nobs = np.array([n_A, n_B])           # 总样本数
proportions_ztest(count, nobs)
#%%



#%% plot percent, all condtions, subplot 1*3
get_ipython().run_line_magic('matplotlib', 'inline')

# df, sig_allneu = load_bd(names[0]) # 读取8个时间段中的一个
sig_allneu = sig_allneu[:,:,:]
neuNum = len(df)
mk_ts = np.array([10, 50, 90]) - aline_ts
condition = ['L_Rand','L_Regular','G_Rand','G_Regular4','G_Regular2','M_Rand','M_Regular4','M_Regular2']

with plt.style.context(style_path):
    fig, axs = plt.subplots(1, 3, figsize=(6, 2), dpi=300, constrained_layout=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][0:]
    colors = ['#808080'] + colors
    k=0
    fixi = 3
    for i in range(2):
        # idx = df['b_'+condition[i]]==1
        idx = df['b_'+condition[fixi]]==1
        if i==1:
            i+=1
        y = sig_allneu[idx,k,:].sum(0)/neuNum
        # y = (y-y.mean())
        axs[0].plot(y, colors[i])
        # axs[0,i].plot(sig_allneu[idx,k,:].mean(0), colors[i])
        axs[0].vlines(mk_ts, 0, 0.4, linestyle='--',linewidth=1, color='gray')  
        k+=1
    axs[0].set_title('Language')
    axs[0].set_xticks(np.array([0,20,40,60,80])+8,['0','1','2','3','4'])
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Percent (%)')
    for j in range(8):
        axs[0].add_patch(Rectangle(xy=(j*10+8, -0.02), width=5, height=1, 
                                   edgecolor=None, facecolor="gray", alpha=0.15))    
        
    for i in range(3):
        # idx = df['b_'+condition[i+2]]==1
        idx = df['b_'+condition[fixi]]==1
        
        y = sig_allneu[idx,k,:].sum(0)/neuNum
        # y = (y-y.mean())
        axs[1].plot(y, colors[i])
        # axs[1,i].plot(sig_allneu[idx,k,:].mean(0), colors[i])
        axs[1].vlines(mk_ts, 0, 0.4, linestyle='--',linewidth=1, color='gray')  
        k+=1
    axs[1].set_title('Geometry')
    axs[1].set_xticks(np.array([0,20,40,60,80])+8,['0','1','2','3','4'])
    axs[1].set_xlabel('Time (s)')
    for j in range(8):
        axs[1].add_patch(Rectangle(xy=(j*10+8, -0.02), width=5, height=1, 
                                   edgecolor=None, facecolor="gray", alpha=0.15))  
    
    for i in range(3):
        # idx = df['b_'+condition[i+5]]==1
        idx = df['b_'+condition[fixi]]==1
        
        y = sig_allneu[idx,k,:].sum(0)/neuNum
        # y = (y-y.mean())
        axs[2].plot(y, colors[i])
        # axs[2,i].plot(sig_allneu[idx,k,:].mean(0), colors[i])
        axs[2].vlines(mk_ts, 0, 0.4, linestyle='--',linewidth=1, color='gray')  
        k+=1
    axs[2].set_title('Music')
    axs[2].set_xticks(np.array([0,20,40,60,80])+8,['0','1','2','3','4'])
    axs[2].set_xlabel('Time (s)')
    for j in range(8):
        axs[2].add_patch(Rectangle(xy=(j*10+8, -0.02), width=5, height=1, 
                                   edgecolor=None, facecolor="gray", alpha=0.15))  
#%% plot 每个neuron平均fr的热图/ plot 
from scipy.stats import gaussian_kde

axs_idx = np.array([[0,0],[0,1],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]])
with plt.style.context(style_path):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axs = plt.subplots(3,3, figsize=(5, 3), dpi=300) 
    for i in range(8):
        x = sig_allneu[:,i,:]
        max_positions = np.where(x)[1]
        
        # 对每一行进行z-score标准化
        # x_zscore = np.apply_along_axis(zscore, axis=1, arr=x)
        # x_zscore = x.copy()
        
        # 找出每行最大值首次出现的位置
        # max_positions = np.argmax(x_zscore, axis=1)
        
        # 根据最大值出现位置排序（位置小的在前）
        # sorted_indices = np.argsort(max_positions)
        # x_sorted = x_zscore[sorted_indices]
        
        #### 画histogram图
        n, bins, patches = axs[axs_idx[i,0], axs_idx[i,1]].hist(max_positions, bins=np.arange(0, 81, 5), 
                                   density=True, alpha=0.7, color='skyblue', 
                                   edgecolor='black', label='Histogram')
        
        # 计算并绘制密度拟合线
        density = gaussian_kde(max_positions)
        x_vals = np.linspace(0, 90, 100)
        axs[axs_idx[i,0], axs_idx[i,1]].plot(x_vals, density(x_vals), color='red', linewidth=2, label='Density Fit')
        # axs[axs_idx[i,0], axs_idx[i,1]].set_ylim([0.005, 0.035])

# # 创建热图
# plt.figure(figsize=(10, 20))
# sns.heatmap(x_sorted, cmap='viridis', cbar_kws={'label': 'Z-score'}, linewidths=0)

# # 添加标题和标签
# plt.title('Sorted Heatmap by First Occurrence of Maximum Value')
# plt.xlabel('Columns')
# plt.ylabel('Rows (sorted)')
#%% venn G&M&L
tmp = df.copy()
s1 = set(tmp[tmp['b_G_Regular4']==1]['neuID_allsub'])
s2 = set(tmp[tmp['b_M_Regular4']==1]['neuID_allsub'])
s3 = set(tmp[tmp['b_L_Regular']==1]['neuID_allsub'])

# s1 = set(tmp[tmp['b_M_Rand']==1]['neuID_allsub'])
# s2 = set(tmp[tmp['b_M_Regular4']==1]['neuID_allsub'])
# s3 = set(tmp[tmp['b_M_Regular2']==1]['neuID_allsub'])

# s4 = set(tmp[tmp['is_mem3']==1].index)
# s1.add(-1)
# s2.add(-1)
# s3.add(-1)
s = [s1, s2, s3]
# total = len(s1.union(s2).union(s3))
total = len(tmp)
# total = len(s1.union(s2).union(s3))
print(total)
# total = len(tmp)
with plt.style.context(style_path):
    fig, axs = plt.subplots(1, 1, figsize=(2, 2), dpi=300, constrained_layout=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    # colors = ['#808080'] + colors
    g=venn3(subsets = s, #传入三组数据
            set_labels = ('G', 'M', 'L'), #设置组名 'Entry', 'Memory1', 'Memory2'
            set_colors=colors[:3], #设置圈的颜色，中间颜色不能修改
            # subset_label_formatter=lambda x: str(x) + "\n(" + f"{(x/total):1.0%}" + ")",
            # subset_label_formatter=lambda x:  f"{x/total:1.0%}",
            subset_label_formatter=lambda x: str(x),
            alpha=0.8,#透明度
            normalize_to=1.0,#venn图占据figure的比例，1.0为占满,
            ax=axs
           )
    plt.tight_layout()
#%% venn G&M
tmp = df.copy()
s1 = set(tmp[tmp['b_G_Rand']==1]['neuID_allsub'])
s2 = set(tmp[tmp['b_G_Regular4']==1]['neuID_allsub'])
s3 = set(tmp[tmp['b_G_Regular2']==1]['neuID_allsub'])

# s1 = set(tmp[tmp['b_M_Rand']==1]['neuID_allsub'])
# s2 = set(tmp[tmp['b_M_Regular4']==1]['neuID_allsub'])
# s3 = set(tmp[tmp['b_M_Regular2']==1]['neuID_allsub'])

# s4 = set(tmp[tmp['is_mem3']==1].index)
# s1.add(-1)
# s2.add(-1)
# s3.add(-1)
s = [s1, s2, s3]
# total = len(s1.union(s2).union(s3))
total = len(tmp)
# total = len(s1.union(s2).union(s3))
print(total)
# total = len(tmp)
with plt.style.context(style_path):
    fig, axs = plt.subplots(1, 1, figsize=(2, 2), dpi=300, constrained_layout=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][:2][::-1]
    colors = ['#808080'] + colors
    g=venn3(subsets = s, #传入三组数据
            set_labels = ('Rand', 'R4', 'R2'), #设置组名 'Entry', 'Memory1', 'Memory2'
            set_colors=colors[:3], #设置圈的颜色，中间颜色不能修改
            # subset_label_formatter=lambda x: str(x) + "\n(" + f"{(x/total):1.0%}" + ")",
            # subset_label_formatter=lambda x:  f"{x/total:1.0%}",
            subset_label_formatter=lambda x: str(x),
            alpha=0.8,#透明度
            normalize_to=1.0,#venn图占据figure的比例，1.0为占满,
            ax=axs
           )
    plt.tight_layout()
#%% venn L
tmp = df.copy()
s1 = set(tmp[tmp['b_L_Rand']==1]['neuID_allsub'])
s3 = set(tmp[tmp['b_L_Regular']==1]['neuID_allsub'])

s = [s1, s3]
# total = len(s1.union(s2).union(s3))
total = len(tmp)
# total = len(s1.union(s2).union(s3))
print(total)
# total = len(tmp)
with plt.style.context(style_path):
    fig, axs = plt.subplots(1, 1, figsize=(2, 2), dpi=300, constrained_layout=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][:2][::-1]
    colors = ['#808080'] + colors
    g=venn2(subsets = s, #传入三组数据
            set_labels = ('Rand', 'R2'), #设置组名 'Entry', 'Memory1', 'Memory2'
            set_colors=[colors[0],colors[2]], #设置圈的颜色，中间颜色不能修改
            # subset_label_formatter=lambda x: str(x) + "\n(" + f"{(x/total):1.0%}" + ")",
            # subset_label_formatter=lambda x:  f"{x/total:1.0%}",
            subset_label_formatter=lambda x: str(x),
            alpha=0.8,#透明度
            normalize_to=1.0,#venn图占据figure的比例，1.0为占满,
            ax=axs
           )
    plt.tight_layout()


