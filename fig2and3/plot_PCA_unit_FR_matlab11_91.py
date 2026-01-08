# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 16:00:50 2025

@author: Wen
"""

get_ipython().run_line_magic('reset', '-sf')
style_path = r'C:\Wen\OneDrive\CodeHub\Python\style_paper.mplstyle'

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import pickle as pkl
import scipy.io as sio
import sys
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

from scipy import stats
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from mne.stats import permutation_cluster_test,permutation_cluster_1samp_test,spatio_temporal_cluster_1samp_test
    
#%% funcs
import numpy as np
from scipy.stats import t

def find_clusters_1d(mask):
    clusters = []
    in_clust = False
    start = None
    for i, m in enumerate(mask):
        if m and not in_clust:
            start = i; in_clust = True
        elif (not m) and in_clust:
            clusters.append(np.arange(start, i)); in_clust = False
    if in_clust:
        clusters.append(np.arange(start, len(mask)))
    return clusters

def cluster_correction_from_perm_stats(obs_stat, perm_stats,
                                       p_thresh=0.05, two_sided=True,
                                       use_global_thresh=False,
                                       min_cluster_size=1,
                                       cluster_stat_func=None):
    """
    obs_stat: (n_times,)
    perm_stats: (n_perm, n_times)   -- 每次 shuffle 的 time-series 统计量
    """
    obs_stat = np.asarray(obs_stat)
    perm_stats = np.asarray(perm_stats)
    n_perm, n_times = perm_stats.shape
    if cluster_stat_func is None:
        cluster_stat_func = lambda stat, idx: np.sum(np.abs(stat[idx]))  # cluster mass

    # 1) 构造阈值（经验法）。两选项：per-timepoint 或 全局
    if two_sided:
        if use_global_thresh:
            flat = np.abs(perm_stats).ravel()
            thresh_val = np.percentile(flat, 100*(1 - p_thresh/2))
            upper_thresh = np.full(n_times, thresh_val)
        else:
            upper_thresh = np.percentile(np.abs(perm_stats), 100*(1 - p_thresh/2), axis=0)
        mask_obs = np.abs(obs_stat) > upper_thresh
    else:
        if use_global_thresh:
            flat = perm_stats.ravel()
            thresh_val = np.percentile(flat, 100*(1 - p_thresh))
            upper_thresh = np.full(n_times, thresh_val)
        else:
            upper_thresh = np.percentile(perm_stats, 100*(1 - p_thresh), axis=0)
        mask_obs = obs_stat > upper_thresh

    # 2) 观测 clusters
    clusters_obs = [c for c in find_clusters_1d(mask_obs) if len(c) >= min_cluster_size]
    cluster_stats_obs = [cluster_stat_func(obs_stat, c) for c in clusters_obs]

    # 3) 对每次置换，计算该置换的最大 cluster-stat（同一阈值与min_cluster_size）
    perm_max_stats = np.zeros(n_perm)
    for i in range(n_perm):
        pvec = perm_stats[i]
        if two_sided:
            mask_p = np.abs(pvec) > upper_thresh
        else:
            mask_p = pvec > upper_thresh
        clusters_p = [c for c in find_clusters_1d(mask_p) if len(c) >= min_cluster_size]
        if len(clusters_p) == 0:
            perm_max_stats[i] = 0.0
        else:
            perm_max_stats[i] = np.max([cluster_stat_func(pvec, c) for c in clusters_p])

    # 4) cluster-level p 值
    cluster_pvals = [ (np.sum(perm_max_stats >= cs) + 1) / (n_perm + 1) for cs in cluster_stats_obs ]

    return {
        'clusters_obs': clusters_obs,
        'cluster_stats_obs': cluster_stats_obs,
        'cluster_pvals': cluster_pvals,
        'perm_max_stats': perm_max_stats,
        'upper_thresh': upper_thresh,
        'mask_obs': mask_obs
    }

#%% plot 所有神经元
path = r'F:\Human_chunking_Huashan_intraoperative\Result\bd_neuron_PCA\Cross_task(fig5)'
# path = r'F:\Human_chunking_Huashan_intraoperative\Result\bd_neuron_PCA\fig2'
file = path + '\\PCA_155drift_AllTrials_TopPCgen_AL5_AllPerm'
# file = path + '\\PCA_155drift_AllTrials_Stacked_TopPCgen_AL5_AllPerm.mat'
data = sio.loadmat(file)

qt = 0
pc = 0
pctask = 0
plottask1 = 1

ch_sel = data['chan_idx_set'][0,qt][pctask,pc][:,0] - 1
fr = data['data_set'][0, plottask1][:,10:90]

scaler = StandardScaler()
y = scaler.fit_transform(fr.T).T

#### plot
# neuNum = y.shape[0]
with plt.style.context(style_path):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][0:]

    fig, ax_top = plt.subplots(1,1, figsize=(1, 3), dpi=300) # , constrained_layout=True
    # fig = plt.figure(figsize=(1.5, 5), dpi=300)  # 整体图像大小（可按需改）
    # plt.subplots_adjust(wspace=0.0, hspace=0.4)
    # gs = GridSpec(1, 1, height_ratios=[1.5, 1])  # 上面是下面的3倍高度
    # ax_top = fig.add_subplot(gs[0, 0])
    # ax_bottom = fig.add_subplot(gs[1, 0])
    
    axs = ax_top
    cmap = plt.cm.viridis
    yy = y[~np.isnan(y[:,0]),:]
    neuNum = yy.shape[0]
    
    # 找出每一行最大值第一次出现的索引
    max_indices = np.argmax(yy, axis=1)
    
    # 按照最大值出现的索引排序
    sorted_order = np.argsort(max_indices)
    yy = yy[sorted_order]


    vmin, vmax = 0.5, 2.5
    h = axs.pcolormesh(yy, cmap = cmap, vmin=vmin, vmax=vmax)
    axs.vlines([20,40,60], 0, yy.shape[0], colors='w', linestyle='--')
    axs.set_xticks([0,20,40,60,80], ['0','1', '2', '3','4'])
    # axs.set_ylabel('Neuron')
    axs.set_ylabel(' ')
    # cbar = fig.colorbar(h,ticks=[0.5,1,1.5,2])
    # cbar = fig.colorbar(h,ticks=[0,1,2,3])
    
    # 将 colorbar 绘制到指定的轴 (cax) 中
    # cax = fig.add_axes([0.92, 0.5, 0.02, 0.38])  # 放在主图右侧，宽度很窄
    # cbar = fig.colorbar(h, cax=cax)
    # cbar.set_label('FR (z-score)')


    axs.set_xticks([0,20,40,60,80], ['0','1', '2', '3','4'])
    # axs.set_ylim([-1.2, 2])
    # axs.set_ylabel('FR(z-score)')
    axs.set_ylabel(' ')
#%% plot 按PC选择的神经元的FR
path = r'F:\Human_chunking_Huashan_intraoperative\Result\bd_neuron_PCA\Cross_task(fig5)'
file = path + '\\PCA_155drift_AllTrials_TopPCgen_AL5_AllPerm'
# path = r'F:\Human_chunking_Huashan_intraoperative\Result\bd_neuron_PCA\fig2'
# file = path + '\\PCA_155drift_AllTrials_Stacked_TopPCgen_AL5_AllPerm.mat'

data = sio.loadmat(file)

qt = 1
pc = 0
pctask = 0
plottask1 = 0

ch_sel = data['chan_idx_set'][0,qt][pctask,pc][:,0] - 1
fr = data['data_set'][0, plottask1][ch_sel,10:90]

scaler = StandardScaler()
y = scaler.fit_transform(fr.T).T

#### plot
# neuNum = y.shape[0]
with plt.style.context(style_path):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][0:]

    fig = plt.figure(figsize=(1.5, 1.5), dpi=300)  # 整体图像大小（可按需改）
    plt.subplots_adjust(wspace=0.0, hspace=0.4)
    gs = GridSpec(2, 1, height_ratios=[1.5, 1])  # 上面是下面的3倍高度
    ax_top = fig.add_subplot(gs[0, 0])
    ax_bottom = fig.add_subplot(gs[1, 0])
    
    axs = ax_top
    cmap = plt.cm.viridis
    yy = y[~np.isnan(y[:,0]),:]
    neuNum = yy.shape[0]

    vmin, vmax = 0.5, 2.5
    h = axs.pcolormesh(yy, cmap = cmap, vmin=vmin, vmax=vmax)
    axs.vlines([20,40,60], 0, yy.shape[0], colors='w', linestyle='--')
    axs.set_xticks([0,20,40,60,80], ['0','1', '2', '3','4'])
    # axs.set_ylabel('Neuron')
    axs.set_ylabel(' ')
    # cbar = fig.colorbar(h,ticks=[0.5,1,1.5,2])
    # cbar = fig.colorbar(h,ticks=[0,1,2,3])
    
    # 将 colorbar 绘制到指定的轴 (cax) 中
    cax = fig.add_axes([0.92, 0.5, 0.02, 0.38])  # 放在主图右侧，宽度很窄
    cbar = fig.colorbar(h, cax=cax)
    cbar.set_label('FR (z-score)')
    
    #### 
    axs = ax_bottom
    plotcolors = 'k'
    # plotcolors = colors[1]
    axs = ax_bottom
    ymean = np.nanmean(yy, 0)
    y_sd = np.nanstd(yy, 0)/np.sqrt(neuNum)
    axs.plot(ymean, color=plotcolors)
    axs.fill_between(np.arange(len(ymean)), ymean-y_sd, ymean+y_sd, color=plotcolors, alpha=0.5)

    axs.set_xticks([0,20,40,60,80], ['0','1', '2', '3','4'])
    # axs.set_ylim([-1.2, 2])
    # axs.set_ylabel('FR(z-score)')
    axs.set_ylabel(' ')
#%% plot 3task PC1-3
from statsmodels.stats.multitest import multipletests

## 'GeoRand'	'Geo4'	'Geo2'	'LanRand'	'LanRule'	'MusRand'	'Mus4'	'Mus2'
# path = r'F:\Human_chunking_Huashan_intraoperative\Result\bd_neuron_PCA\fig3'
path = r'F:\Human_chunking_Huashan_intraoperative\Result\bd_neuron_PCA\fig2'
file = path + '\\PCA_10subj_3tasks_160drift_AllTrials_TopPCgen_AL5_AllPerm_trim.mat'
# file = path + '\\PCA_10subj_3tasks_160drift_AllTrials_Stacked_TopPCgen_AL5_AllPerm_trim.mat'

# file = path + '\\PCA_155drift_AllTrials_TopPCgen_AL5_AllPerm_RespTrim_PCMax6.mat'
# file = path + '\\PCA_155drift_AllTrials_Stacked_TopPCgen_AL5_AllPerm_RespTrim.mat'
# file = path + '\\PCA_155drift_CorrectTrials_TopPCgen_AL5_AllPerm_PCmax5.mat'
# file = path + '\\PCA_155drift_CorrectTrials_Stacked_TopPCgen_AL5_AllPerm.mat'

data = sio.loadmat(file)
qt = 1

for pcidx in [0,1,2]:
    pc = pcidx
    
    pctask = 7
    plottask1 = pctask
    # plottask2 = 2
    plot_correct = 0 # 0=correct, 1=error
    # perm_set = data['perm_set'][0,qt][pctask,pc,plottask1]
    
    ch_sel = data['chan_idx_set'][0,qt][pctask,pc][:,0] - 1
    ### correct/error
    fr = data['data_set'][plot_correct, plottask1][ch_sel,10:95]
    # fr_rnd = data['data_set'][plot_correct, plottask2][ch_sel,10:95]
    
    scaler = StandardScaler()
    y = scaler.fit_transform(fr.T).T
    # y_rnd = scaler.fit_transform(fr_rnd.T).T

    #### plot
    pval = data['pval_hold_set'][plot_correct,qt][pctask,pc,plottask1][0,:] ### correct
    # reject, pval, alphacSidak, alphacBonf = multipletests(pval, alpha=0.05, method='fdr_bh')
    # cw = 3
    cw = 6
    pvalmsk = pval<0.0
    # pvalmsk = pval<0.025
    pvalmsk2 = pval<0.005
    
    # neuNum = y.shape[0]
    with plt.style.context(style_path):
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][0:]
        colors = ['k'] + colors
        
        plotcolors = colors
        plotcolors = colors[0]
        bgcolor = colors[plottask1]
        
        # fig, axsall = plt.subplots(2,1, figsize=(1.6, 1.7), dpi=300) # , constrained_layout=True
        fig = plt.figure(figsize=(1.5, 1.5), dpi=300)  # 整体图像大小（可按需改）
        plt.subplots_adjust(wspace=0.0, hspace=0.4)
        gs = GridSpec(2, 1, height_ratios=[1.5, 1])  # 上面是下面的3倍高度
        ax_top = fig.add_subplot(gs[0, 0])
        ax_bottom = fig.add_subplot(gs[1, 0])
        
        axs = ax_top
        cmap = plt.cm.viridis
        yy = y[~np.isnan(y[:,0]),:]
        # yy_rnd = y_rnd[~np.isnan(y_rnd[:,0]),:]
        neuNum = yy.shape[0]
        
        vmin, vmax = 0.5, 2.5
        h = axs.pcolormesh(yy, cmap = cmap, vmin=vmin, vmax=vmax)
        # axs.vlines(np.array([10,30,50,70]),0, yy.shape[0], linestyle='--',linewidth=1, color='gray')
        # axs.vlines(np.array([0,10,20,30,40,50,60,70]),0, yy.shape[0], linestyle='--',linewidth=1, color='gray')
        axs.vlines(np.array([20, 40, 60]), 0, yy.shape[0], colors='w', linestyle='--',linewidth=1)  
        axs.set_xticks(np.arange(0,80,10), [str(ii) for ii in range(1,9)])
        for tt in range(8):
            axs.add_patch(Rectangle((tt*10, 0), 5, neuNum, edgecolor=None, facecolor="gray", alpha=0.5))
 
        # axs.set_ylabel('Neuron')
        axs.set_ylabel(' ')
        # cbar = fig.colorbar(h,ticks=[0.5,1,1.5,2])
        # cbar = fig.colorbar(h,ticks=[0,1,2,3])
        axs.set_ylim([0,neuNum])
        
        # 将 colorbar 绘制到指定的轴 (cax) 中
        cax = fig.add_axes([0.94, 0.5, 0.02, 0.38])  # 放在主图右侧，宽度很窄
        cbar = fig.colorbar(h, cax=cax)
        cbar.set_label('FR (z-score)')
    
        axs = ax_bottom
        ymean = np.nanmean(yy, 0)
        y_sd = np.nanstd(yy, 0)/np.sqrt(neuNum)
        axs.plot(ymean, color=plotcolors)
        axs.fill_between(np.arange(len(ymean)), ymean-y_sd, ymean+y_sd, color=plotcolors, alpha=0.5)
        # 坐标轴背景色
        # axs.set_facecolor(bgcolor)
        axs.patch.set_alpha(0.1)
    
        # 找出连续 >=3 个 1 的区间
        indices = []
        start = None
        for i, val in enumerate(pvalmsk):
            if val == 1:
                if start is None:
                    start = i
            else:
                if start is not None:
                    if i - start > cw:
                        indices.append((start, i))
                    start = None
        # 如果最后到结尾还是连续的1
        if start is not None and len(pvalmsk) - start >= 0:
            indices.append((start, len(pvalmsk)-1))
        
        indices2 = []
        start = None
        for i, val in enumerate(pvalmsk2):
            if val == 1:
                if start is None:
                    start = i
            else:
                if start is not None:
                    if i - start > cw:
                        indices2.append((start, i))
                    start = None
        # 如果最后到结尾还是连续的1
        if start is not None and len(pvalmsk2) - start >= 0:
            indices2.append((start, len(pvalmsk2)-1))
            
        # 在 y=1 画横线
        for (s, e) in indices:
            axs.hlines(y=ymean.max()+0.5, xmin=s, xmax=e, colors='gray', linewidth=1)
        for (s, e) in indices2: 
            axs.hlines(y=ymean.max()+0.5, xmin=s, xmax=e, colors='k', linewidth=1)
        
        # axs.set_xticks([0,20,40,60,80], ['0','1', '2', '3','4'])
        axs.set_xticks(np.arange(0,80,10), [str(ii) for ii in range(1,9)])
        # axs.set_xticks([0,20,40,60,75], ['1(0s)', '3', '5(2s)','7','8 off(3.5s)'])
        for tt in range(8):
            axs.add_patch(Rectangle((tt*10, -1.2), 5, 6, edgecolor=None, facecolor="gray", alpha=0.25))
 
        
        axs.set_ylim([-1.2, 2.7])
        if pcidx==0:
            axs.set_ylim([-1.2, 3.2])
        # axs.set_ylabel('FR-score)')
        axs.set_ylabel(' ')
        
        # figname = r'C:\Wen\OneDrive\Project\Human_sqchunk_intraoperative\Manuscript\Figures\Figure3'
        # figname = figname + '\\pctask'+str(pctask)+'pcplot'+str(plottask1)+'error_'+str(plot_correct)+'.pdf'
        # plt.savefig(figname, transparent=True, dpi=300)


# exp_var1 = data['PC_output'][4,0][:40,0]
# exp_var2 = data['PC_output'][4,1][:40,0]
# exp_var3 = data['PC_output'][4,2][:40,0]

# plt.plot(exp_var1[:20],marker='o')
# plt.plot(exp_var2[:20],marker='o')
# plt.plot(exp_var3[:20],marker='o')
#%% plot cross rule, fig3
from statsmodels.stats.multitest import multipletests

## 'GeoRand'	'Geo4'	'Geo2'	'LanRand'	'LanRule'	'MusRand'	'Mus4'	'Mus2'
path = r'F:\Human_chunking_Huashan_intraoperative\Result\bd_neuron_PCA\fig3'
# path = r'F:\Human_chunking_Huashan_intraoperative\Result\bd_neuron_PCA\fig2'
# file = path + '\\PCA_155drift_AllTrials_TopPCgen_AL5_AllPerm_RespTrim_PCMax6.mat'
# file = path + '\\PCA_155drift_AllTrials_Stacked_TopPCgen_AL5_AllPerm_RespTrim.mat'
# file = path + '\\PCA_155drift_CorrectTrials_TopPCgen_AL5_AllPerm_PCmax5.mat'
# file = path + '\\PCA_155drift_CorrectTrials_Stacked_TopPCgen_AL5_AllPerm.mat'
file = path + '\\PCA_10subj_160drift_CorrectTrials_TopPCgen_AL5_AllPerm_StatExtend_withError_trim.mat'

data = sio.loadmat(file)
qt = 1

for ppp in [0,1,2]:
    pc = 1
    
    pctask = 1
    plottask1 = ppp
    # plottask2 = 2
    plot_correct = 0 # 0=correct, 1=error
    # perm_set = data['perm_set'][0,qt][pctask,pc,plottask1]
    
    ch_sel = data['chan_idx_set'][0,qt][pctask,pc][:,0] - 1
    ### correct/error
    fr = data['data_set'][plot_correct, plottask1][ch_sel,10:95]
    # fr_rnd = data['data_set'][plot_correct, plottask2][ch_sel,10:95]
    
    scaler = StandardScaler()
    y = scaler.fit_transform(fr.T).T
    # y_rnd = scaler.fit_transform(fr_rnd.T).T

    #### plot
    pval = data['pval_hold_set'][plot_correct,qt][pctask,pc,plottask1][0,1:90] ### correct
    # reject, pval, alphacSidak, alphacBonf = multipletests(pval, alpha=0.05, method='fdr_bh')
    cw = 6
    # cw = 2
    # pvalmsk = pval<0.0
    pvalmsk = pval<0.025
    pvalmsk2 = pval<0.005
    
    # neuNum = y.shape[0]
    with plt.style.context(style_path):
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][0:]
        colors = ['gray'] + colors
        
        plotcolors = 'k'
        plotcolors = colors[1]
        bgcolor = colors[plottask1]
        
        # fig, axsall = plt.subplots(2,1, figsize=(1.6, 1.7), dpi=300) # , constrained_layout=True
        fig = plt.figure(figsize=(1.5, 1.5), dpi=300)  # 整体图像大小（可按需改）
        plt.subplots_adjust(wspace=0.0, hspace=0.4)
        gs = GridSpec(2, 1, height_ratios=[1.5, 1])  # 上面是下面的3倍高度
        ax_top = fig.add_subplot(gs[0, 0])
        ax_bottom = fig.add_subplot(gs[1, 0])
        
        axs = ax_top
        cmap = plt.cm.viridis
        yy = y[~np.isnan(y[:,0]),:]
        # yy_rnd = y_rnd[~np.isnan(y_rnd[:,0]),:]
        neuNum = yy.shape[0]
        
        vmin, vmax = 0.5, 2.5
        h = axs.pcolormesh(yy, cmap = cmap, vmin=vmin, vmax=vmax)
        # axs.vlines(np.array([10,30,50,70]),0, yy.shape[0], linestyle='--',linewidth=1, color='gray')
        # axs.vlines(np.array([0,10,20,30,40,50,60,70]),0, yy.shape[0], linestyle='--',linewidth=1, color='gray')
        axs.vlines(np.array([20, 40, 60]), 0, yy.shape[0], colors='w', linestyle='--',linewidth=1)  
        axs.set_xticks(np.arange(0,80,10), [str(ii) for ii in range(1,9)])
        for tt in range(8):
            axs.add_patch(Rectangle((tt*10, 0), 5, yy.shape[0], edgecolor=None, facecolor="gray", alpha=0.5))
 
        # axs.set_ylabel('Neuron')
        axs.set_ylabel(' ')
        # cbar = fig.colorbar(h,ticks=[0.5,1,1.5,2])
        # cbar = fig.colorbar(h,ticks=[0,1,2,3])
        axs.set_ylim([0,yy.shape[0]])
        
        # 将 colorbar 绘制到指定的轴 (cax) 中
        cax = fig.add_axes([0.94, 0.5, 0.02, 0.38])  # 放在主图右侧，宽度很窄
        cbar = fig.colorbar(h, cax=cax)
        cbar.set_label('FR (z-score)')
    
        axs = ax_bottom
        ymean = np.nanmean(yy, 0)
        y_sd = np.nanstd(yy, 0)/np.sqrt(neuNum)
        axs.plot(ymean, color=plotcolors)
        axs.fill_between(np.arange(len(ymean)), ymean-y_sd, ymean+y_sd, color=plotcolors, alpha=0.5)
        # 坐标轴背景色
        # axs.set_facecolor(bgcolor)
        axs.patch.set_alpha(0.1)
    
        # 找出连续 >=3 个 1 的区间
        indices = []
        start = None
        for i, val in enumerate(pvalmsk):
            if val == 1:
                if start is None:
                    start = i
            else:
                if start is not None:
                    if i - start > cw:
                        indices.append((start, i))
                    start = None
        # 如果最后到结尾还是连续的1
        if start is not None and len(pvalmsk) - start >= 0:
            indices.append((start, len(pvalmsk)-1))
        
        indices2 = []
        start = None
        for i, val in enumerate(pvalmsk2):
            if val == 1:
                if start is None:
                    start = i
            else:
                if start is not None:
                    if i - start > cw:
                        indices2.append((start, i))
                    start = None
        # 如果最后到结尾还是连续的1
        if start is not None and len(pvalmsk2) - start >= 0:
            indices2.append((start, len(pvalmsk2)-1))
            
        # 在 y=1 画横线
        for (s, e) in indices:
            axs.hlines(y=ymean.max()+0.5, xmin=s, xmax=e, colors='gray', linewidth=1)
        for (s, e) in indices2: 
            axs.hlines(y=ymean.max()+0.5, xmin=s, xmax=e, colors='k', linewidth=1)
        
        # axs.set_xticks([0,20,40,60,80], ['0','1', '2', '3','4'])
        axs.set_xticks(np.arange(0,80,10), [str(ii) for ii in range(1,9)])
        # axs.set_xticks([0,20,40,60,75], ['1(0s)', '3', '5(2s)','7','8 off(3.5s)'])
        for tt in range(8):
            axs.add_patch(Rectangle((tt*10, -1.2), 5, 6, edgecolor=None, facecolor="gray", alpha=0.25))
 
        
        axs.set_ylim([-1.2, 2])
        if ppp==0:
            axs.set_ylim([-1.2, 3.2])
        # axs.set_ylabel('FR-score)')
        axs.set_ylabel(' ')
        
        figname = r'C:\Wen\OneDrive\Project\Human_sqchunk_intraoperative\Manuscript\Figures\Figure3'
        figname = figname + '\\10sub_pctask'+str(pctask)+'pcplot'+str(plottask1)+'error_'+str(plot_correct)+'.pdf'
        plt.savefig(figname, transparent=True, dpi=300)
#%% plot cross rule, motor
from statsmodels.stats.multitest import multipletests

## 'GeoRand'	'Geo4'	'Geo2'	'LanRand'	'LanRule'	'MusRand'	'Mus4'	'Mus2'
path = r'F:\Human_chunking_Huashan_intraoperative\Result\bd_neuron_PCA\Motor'
# file = path + '\\PCA_M1_Ctrl_AllTrials_TopPCgen_AL5_AllPerm_RespTrim.mat'
file = path + '\\PCA_M1_Ctrl_AllTrials_Stacked_TopPCgen_AL5_AllPerm_RespTrim.mat'

data = sio.loadmat(file)

qt = 1
for pcidx in [0,1,2]:
# for pcidx in [2]:
    pc = pcidx
    
    pctask = 0
    plottask1 = pctask
    # plottask2 = 2
    plot_correct = 0 # 0=correct, 1=error
    # perm_set = data['perm_set'][0,qt][pctask,pc,plottask1]
    
    ch_sel = data['chan_idx_set'][0,qt][pctask,pc][:,0] - 1
    ### correct/error
    fr = data['data_set'][plot_correct, plottask1][ch_sel,10:95]
    # fr_rnd = data['data_set'][plot_correct, plottask2][ch_sel,10:95]
    
    scaler = StandardScaler()
    y = scaler.fit_transform(fr.T).T
    # y_rnd = scaler.fit_transform(fr_rnd.T).T

    #### plot
    pval = data['pval_hold_set'][plot_correct,qt][pctask,pc,plottask1][0,:] ### correct
    # reject, pval, alphacSidak, alphacBonf = multipletests(pval, alpha=0.05, method='fdr_bh')
    cw = 6
    # cw = 3
    pvalmsk = pval<0.0#25
    pvalmsk2 = pval<0.005
    
    # neuNum = y.shape[0]
    with plt.style.context(style_path):
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][0:]
        colors = ['gray'] + colors
        
        plotcolors = 'k'
        # plotcolors = colors[1]
        bgcolor = colors[plottask1]
        
        # fig, axsall = plt.subplots(2,1, figsize=(1.6, 1.7), dpi=300) # , constrained_layout=True
        fig = plt.figure(figsize=(1.5, 1.5), dpi=300)  # 整体图像大小（可按需改）
        plt.subplots_adjust(wspace=0.0, hspace=0.4)
        gs = GridSpec(2, 1, height_ratios=[1.5, 1])  # 上面是下面的3倍高度
        ax_top = fig.add_subplot(gs[0, 0])
        ax_bottom = fig.add_subplot(gs[1, 0])
        
        axs = ax_top
        cmap = plt.cm.viridis
        yy = y[~np.isnan(y[:,0]),:]
        # yy_rnd = y_rnd[~np.isnan(y_rnd[:,0]),:]
        neuNum = yy.shape[0]
        
        vmin, vmax = 0.5, 2.5
        h = axs.pcolormesh(yy, cmap = cmap, vmin=vmin, vmax=vmax)
        # axs.vlines(np.array([10,30,50,70]),0, yy.shape[0], linestyle='--',linewidth=1, color='gray')
        # axs.vlines(np.array([0,10,20,30,40,50,60,70]),0, yy.shape[0], linestyle='--',linewidth=1, color='gray')
        axs.vlines(np.array([20, 40, 60]), 0, yy.shape[0], colors='w', linestyle='--',linewidth=1)  
        axs.set_xticks(np.arange(0,80,10), [str(ii) for ii in range(1,9)])
        for tt in range(8):
            axs.add_patch(Rectangle((tt*10, 0), 5, yy.shape[0], edgecolor=None, facecolor="gray", alpha=0.5))
 
        # axs.set_ylabel('Neuron')
        axs.set_ylabel(' ')
        # cbar = fig.colorbar(h,ticks=[0.5,1,1.5,2])
        # cbar = fig.colorbar(h,ticks=[0,1,2,3])
        axs.set_ylim([0,yy.shape[0]])
        
        # 将 colorbar 绘制到指定的轴 (cax) 中
        cax = fig.add_axes([0.94, 0.5, 0.02, 0.38])  # 放在主图右侧，宽度很窄
        cbar = fig.colorbar(h, cax=cax)
        cbar.set_label('FR (z-score)')
    
        axs = ax_bottom
        ymean = np.nanmean(yy, 0)
        y_sd = np.nanstd(yy, 0)/np.sqrt(neuNum)
        axs.plot(ymean, color=plotcolors)
        axs.fill_between(np.arange(len(ymean)), ymean-y_sd, ymean+y_sd, color=plotcolors, alpha=0.5)
        # 坐标轴背景色
        # axs.set_facecolor(bgcolor)
        axs.patch.set_alpha(0.1)
    
        # 找出连续 >=3 个 1 的区间
        indices = []
        start = None
        for i, val in enumerate(pvalmsk):
            if val == 1:
                if start is None:
                    start = i
            else:
                if start is not None:
                    if i - start > cw:
                        indices.append((start, i))
                    start = None
        # 如果最后到结尾还是连续的1
        if start is not None and len(pvalmsk) - start >= 0:
            indices.append((start, len(pvalmsk)-1))
        
        indices2 = []
        start = None
        for i, val in enumerate(pvalmsk2):
            if val == 1:
                if start is None:
                    start = i
            else:
                if start is not None:
                    if i - start > cw:
                        indices2.append((start, i))
                    start = None
        # 如果最后到结尾还是连续的1
        if start is not None and len(pvalmsk2) - start >= 0:
            indices2.append((start, len(pvalmsk2)-1))
            
        # 在 y=1 画横线
        for (s, e) in indices:
            axs.hlines(y=ymean.max()+0.5, xmin=s, xmax=e, colors='gray', linewidth=1)
        for (s, e) in indices2: 
            axs.hlines(y=ymean.max()+0.5, xmin=s, xmax=e, colors='k', linewidth=1)
        
        # axs.set_xticks([0,20,40,60,80], ['0','1', '2', '3','4'])
        axs.set_xticks(np.arange(0,80,10), [str(ii) for ii in range(1,9)])
        # axs.set_xticks([0,20,40,60,75], ['1(0s)', '3', '5(2s)','7','8 off(3.5s)'])
        for tt in range(8):
            axs.add_patch(Rectangle((tt*10, -1.2), 5, 6, edgecolor=None, facecolor="gray", alpha=0.25))
 
        
        axs.set_ylim([-1.2, 2.7])
        if pcidx==0:
            axs.set_ylim([-1.2, 3.2])
        # axs.set_ylabel('FR-score)')
        axs.set_ylabel(' ')
        
#%% 比较显著时间的早晚， end, L0 vs L1
path = r'F:\Human_chunking_Huashan_intraoperative\Result\bd_neuron_PCA\fig2'
# file = path + '\\PCA_155drift_CorrectTrials_TopPCgen_AL5_AllPerm_PCmax5.mat'
file = path + '\\PCA_155drift_AllTrials_TopPCgen_AL5_AllPerm_RespTrim_PCMax6.mat'
data = sio.loadmat(file)

qt = 1
pc1 = 0
pc2 = 0
pc3 = 4
plottask1 = 0
plottask2 = 1
plottask3 = 2

### L0
ch_sel = data['chan_idx_set'][0,qt][plottask1,pc1][:,0] - 1
fr = data['data_set'][0, plottask1][ch_sel,10:90][:,60:] 
### L1/L2
ch_sel = data['chan_idx_set'][0,qt][plottask2,pc2][:,0] - 1
fr2 = data['data_set'][0, plottask2][ch_sel,10:90][:,60:]

ch_sel = data['chan_idx_set'][0,qt][plottask3,pc3][:,0] - 1
fr3 = data['data_set'][0, plottask3][ch_sel,10:90][:,60:]


maxidx = np.zeros([1000,3])
for r in range(1000):
    perm_idx = np.random.default_rng().choice(fr.shape[0], size=10, replace=False)
    maxidx[r,0] = fr[perm_idx,:].mean(0).argmax()*0.05+3
    maxidx[r,1] = fr2[perm_idx,:].mean(0).argmax()*0.05+3
    maxidx[r,2] = fr3[perm_idx,:].mean(0).argmax()*0.05+3

with plt.style.context(style_path):
    from scipy.stats import gaussian_kde
    task = ['L0 task', 'L1 task', 'L2 task']
    fig, axs = plt.subplots(1, 3, figsize=(3.3, 1.2), dpi=300, constrained_layout=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color1 = ['gray', colors[0], colors[1]]
    for i in range(3):
        dat = maxidx[:,i]
        kde = gaussian_kde(dat, bw_method=0.5)
        x = np.linspace(3, 4.15, 200)
        if i==1:
            x = np.linspace(3, 4.15, 200)
        y = kde(x)
        axs[i].plot(x, y, color=color1[i], lw=1)
        axs[i].fill_between(x, y, color=color1[i], alpha=0.3)
        axs[i].vlines([4], 0, y.max(), linestyle='-',linewidth=1, color='k')  
        axs[i].vlines([3.75], 0, y.max(), linestyle='--',linewidth=1, color='gray')  
        axs[i].set_title(task[i])
        axs[i].set_xticks([3, 4],['3s','4s'])

    axs[0].set_xlabel('Time (s)')    
    axs[0].set_ylabel('Density (peak time)')    
    
print(sum(maxidx[:,0]>3.75)/1000)
print(sum(maxidx[:,1]>3.75)/1000)
print(sum(maxidx[:,2]>3.75)/1000)

with plt.style.context(style_path):
    import seaborn as sns
    
    fig, axs = plt.subplots(1, 1, figsize=(1.7, 1.5), dpi=300, constrained_layout=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    
    df = pd.DataFrame({'x1':maxidx[:,0],'x2':maxidx[:,1],'x3':maxidx[:,2]})
    df_melted = df.melt(
            var_name='Variable', value_name='Value')
    
    # 画小提琴图
    color1 = ['gray', colors[0], colors[1]]
    axs = sns.violinplot(x='Variable', y='Value', ax=axs, data=df_melted)
    
    color1 = ['gray', colors[0], colors[1]]
    for patch, color in zip(axs.collections[::1], color1):  # 每个 violin 有 2 个 PolyCollection（轮廓和填充）
        patch.set_facecolor(color)
    # sns.kdeplot(x=maxidx[:,0], fill=True, color='gray')  # fill=True 表示填充曲线下面积
    # sns.kdeplot(x=maxidx[:,1], fill=True, color=colors[0])  # fill=True 表示填充曲线下面积
    axs.set_xticks([0,1,2],['L0','L1','L2'])
    axs.set_xlabel('Task')
    axs.set_ylabel('Peak time (s)')
    axs.set_ylim([2.8, 4.3])
    
print(stats.ttest_ind(maxidx[:,0], maxidx[:,2]))
print(maxidx.mean(0))
#%% 比较最早显著时刻，  L0， L1
path = r'F:\Human_chunking_Huashan_intraoperative\Result\bd_neuron_PCA\fig2'
# file = path + '\\PCA_155drift_AllTrials_TopPCgen_AL5_AllPerm_RespTrim_StatExtend.mat'
file = path + '\\PCA_10subj_3tasks_160drift_AllTrials_TopPCgen_AL5_AllPerm.mat'
data = sio.loadmat(file)
perm = data['perm_hold_set'][0,1]

qt = 1
# pc1 = 0
# pc2 = 0
# pc3 = 0
# plottask1 = 0
# plottask2 = 1
# plottask3 = 2

pc1 = 1
pc2 = 1
pc3 = 2
plottask1 = 0
plottask2 = 1
plottask3 = 2

### L0
ch_sel = data['chan_idx_set'][0,qt][plottask1,pc1][:,0] - 1
fr = data['data_set'][0, plottask1][ch_sel,10:90][:,:] 
scaler = StandardScaler()
fr = scaler.fit_transform(fr.T).T
### L1/L2
ch_sel = data['chan_idx_set'][0,qt][plottask2,pc2][:,0] - 1
fr2 = data['data_set'][0, plottask2][ch_sel,10:90][:,:]
scaler = StandardScaler()
fr2 = scaler.fit_transform(fr2.T).T

ch_sel = data['chan_idx_set'][0,qt][plottask3,pc3][:,0] - 1
fr3 = data['data_set'][0, plottask3][ch_sel,10:90][:,:]
scaler = StandardScaler()
fr3 = scaler.fit_transform(fr3.T).T

perm1 = perm[0,pc1,0]
perm2 = perm[1,pc2,1]
perm3 = perm[2,pc3,2]

maxidx = np.zeros([1000,3])+np.nan
for r in range(1000):
    perm_idx = np.random.default_rng().choice(fr.shape[0], size=10, replace=False)
    tmp = fr[perm_idx,:].mean(0)
    tmp2 = fr2[perm_idx,:].mean(0)
    tmp3 = fr3[perm_idx,:].mean(0)
    
    pval3t = np.zeros([80,3,1000]) 
    for i in range(80):
        pval3t[i,0,r] = np.sum(perm1[:,i]>tmp[i])/1000
        pval3t[i,1,r] = np.sum(perm2[:,i]>tmp2[i])/1000
        pval3t[i,2,r] = np.sum(perm3[:,i]>tmp3[i])/1000
    
    for j in range(3):
        pval = pval3t[:,j,r]
        pvalmsk = pval<0.005
        
        indices = []
        cw = 4
        start = None
        for i, val in enumerate(pvalmsk):
            if val == 1:
                if start is None:
                    start = i
            else:
                if start is not None:
                    if i - start > cw:
                        indices.append(start)
                    start = None
        # 如果最后到结尾还是连续的1
        if start is not None and len(pvalmsk) - start >= 0:
            indices.append(start)
        
        if len(indices)>0:
            maxidx[r,j] = np.min(indices)
maxidx = maxidx*0.05

with plt.style.context(style_path):
    from scipy.stats import gaussian_kde
    task = ['L0', 'L1', 'L2']
    fig, axs = plt.subplots(1, 3, figsize=(2, 1.1), dpi=300, constrained_layout=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color1 = ['gray', 'gray', 'gray']
    for i in range(3):
        dat = maxidx[:,i]
        dat = dat[dat>-1]
        kde = gaussian_kde(dat, bw_method=0.5)
        x = np.linspace(2.5, 4.15, 200)
        if i==1:
            x = np.linspace(2.5, 4.15, 200)
        y = kde(x)
        axs[i].plot(x, y, color=color1[i], lw=1)
        axs[i].fill_between(x, y, color=color1[i], alpha=0.3)
        axs[i].vlines([3.75], 0, y.max()+1, linestyle='-',linewidth=1, color='k')  
        axs[i].vlines([3.5], 0, y.max()+1, linestyle='--',linewidth=1, color='gray')  
        axs[i].set_title(task[i])
        axs[i].set_xticks([2.5,3,3.5],['6','7','8'])
        axs[i].set_ylim([-0.1, y.max()+1.5])

    axs[0].set_xlabel('Rank')    
    axs[0].set_ylabel('Density')   

print(sum(maxidx[:,0]>3.5)/1000)
print(sum(maxidx[:,1]>3.5)/1000)
print(sum(maxidx[:,2]>3.5)/1000)

#### L1 PC
if pc2==1:
    with plt.style.context(style_path):
        from scipy.stats import gaussian_kde
        task = ['L0', 'L1', 'L2']
        fig, axs = plt.subplots(1, 2, figsize=(1.2, 1.1), dpi=300, constrained_layout=True)
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        color1 = ['gray', colors[0], colors[0]]
        for i in range(2):
            dat = maxidx[:,i+1]
            dat = dat[dat>-1]
            kde = gaussian_kde(dat, bw_method=0.5)
            x = np.linspace(1, 3, 200)
            if i==1:
                x = np.linspace(1, 3, 200)
            y = kde(x)
            axs[i].plot(x, y, color=color1[i+1], lw=1)
            axs[i].fill_between(x, y, color=color1[i+1], alpha=0.3)
            axs[i].vlines([2], 0, y.max()+1, linestyle='--',linewidth=1, color='gray')  
            axs[i].set_title(task[i+1])
            axs[i].set_xticks([1.5,2,2.5],['4','5','6'])
            axs[i].set_ylim([-0.1, y.max()+1.5])
    
        axs[0].set_xlabel(' ')    
        # axs[0].set_ylabel('Density (peak time)')   

print(sum(maxidx[:,1]>2)/1000)
print(sum(maxidx[:,2]>2)/1000)
#%% 比较最早显著时刻， L2
path = r'F:\Human_chunking_Huashan_intraoperative\Result\bd_neuron_PCA\fig2'
# file = path + '\\PCA_155drift_AllTrials_Stacked_TopPCgen_AL5_AllPerm_RespTrim_StatExtend.mat'
file = path + '\\PCA_10subj_3tasks_160drift_AllTrials_Stacked_TopPCgen_AL5_AllPerm.mat'
data = sio.loadmat(file)
perm = data['perm_hold_set'][0,1]

qt = 1
pc1 = 2
plottask1 = 2

### L0
ch_sel = data['chan_idx_set'][0,qt][plottask1,pc1][:,0] - 1
fr = data['data_set'][0, plottask1][ch_sel,10:90][:,:] 
scaler = StandardScaler()
fr = scaler.fit_transform(fr.T).T


perm1 = perm[0,pc1,0]

maxidx = np.zeros([1000,3])+np.nan
for r in range(1000):
    perm_idx = np.random.default_rng().choice(fr.shape[0], size=10, replace=False)
    tmp = fr[perm_idx,:].mean(0)
    
    pval3t = np.zeros([80,3,1000]) 
    for i in range(80):
        pval3t[i,0,r] = np.sum(perm1[:,i]>tmp[i])/1000
    
    for j in range(1):
        pval = pval3t[:,j,r]
        pvalmsk = pval<0.025
        
        indices = []
        cw = 2
        start = None
        for i, val in enumerate(pvalmsk):
            if val == 1:
                if start is None:
                    start = i
            else:
                if start is not None:
                    if i - start > cw:
                        indices.append(start)
                    start = None
        # 如果最后到结尾还是连续的1
        if start is not None and len(pvalmsk) - start >= 0:
            indices.append(start)
        
        indices = np.array(indices).astype(float)
        indices[indices<30] = indices[indices<30]-20
        indices[(30<indices) & (indices<50)] = indices[(30<indices) & (indices<50)]-40
        indices[(50<indices) & (indices<70)] = indices[(50<indices) & (indices<70)]-60
        indices[(70<indices) & (indices<90)] = np.nan
        if len(indices)>0:
            maxidx[r,j] = np.nanmean(indices)
            
maxidx = maxidx*0.05

with plt.style.context(style_path):
    from scipy.stats import gaussian_kde
    task = ['L0 task', 'L1 task', 'L2 task']
    fig, axs = plt.subplots(1, 1, figsize=(0.6, 1.1), dpi=300, constrained_layout=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i in range(1):
        dat = maxidx[:,i]
        dat = dat[dat>-1]
        kde = gaussian_kde(dat, bw_method=0.5)
        x = np.linspace(-1, 1, 200)
        if i==1:
            x = np.linspace(-1, 1, 200)
        y = kde(x)
        axs.plot(x, y, color=colors[1], lw=1)
        axs.fill_between(x, y, color=colors[1], alpha=0.3)
        # axs[i].vlines([0], 0, y.max(), linestyle='-',linewidth=1, color='k')  
        axs.vlines([0], 0, y.max()+1, linestyle='--',linewidth=1, color='gray')  
        axs.set_title('L2')
        axs.set_xticks([0],['3/5/7'])
        axs.set_ylim([-0.1, y.max()+1.5])

    axs.set_xlabel(' ')    
    # axs[0].set_ylabel('Density (peak time)')   

print(sum(maxidx[:,0]>0)/1000)
#%% 比较显著时间的早晚， 泛化到L1和L2中的end
path = r'F:\Human_chunking_Huashan_intraoperative\Result\bd_neuron_PCA\fig2'
file = path + '\\PCA_155drift_CorrectTrials_TopPCgen_AL5_AllPerm_PCmax5.mat'
data = sio.loadmat(file)

qt = 1
pc1 = 0
pctask1 = 0
plottask1 = 0
plottask2 = 1
plottask3 = 2

### L0
ch_sel = data['chan_idx_set'][0,qt][pctask1,pc1][:,0] - 1
fr = data['data_set'][0, plottask1][ch_sel,10:90][:,60:] 
### L1/L2
# ch_sel = data['chan_idx_set'][0,qt][pctask2,pc2][:,0] - 1
fr2 = data['data_set'][0, plottask2][ch_sel,10:90][:,60:]

# ch_sel = data['chan_idx_set'][0,qt][pctask3,pc3][:,0] - 1
fr3 = data['data_set'][0, plottask3][ch_sel,10:90][:,60:]


maxidx = np.zeros([50,3])
for r in range(50):
    perm_idx = np.random.default_rng().choice(fr.shape[0], size=5, replace=False)
    maxidx[r,0] = fr[perm_idx,:].mean(0).argmax()*0.05+3
    maxidx[r,1] = fr2[perm_idx,:].mean(0).argmax()*0.05+3
    maxidx[r,2] = fr3[perm_idx,:].mean(0).argmax()*0.05+3

with plt.style.context(style_path):
    import seaborn as sns
    
    fig, axs = plt.subplots(1, 1, figsize=(1.4, 1.5), dpi=300, constrained_layout=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    
    df = pd.DataFrame({'x1':maxidx[:,0],'x2':maxidx[:,1],'x3':maxidx[:,2]})
    df_melted = df.melt(
            var_name='Variable', value_name='Value')
    
    # 画小提琴图
    color1 = ['gray', colors[0], colors[1]]
    axs = sns.violinplot(x='Variable', y='Value', ax=axs, data=df_melted)
    
    color1 = ['gray', colors[0], colors[1]]
    for patch, color in zip(axs.collections[::1], color1):  # 每个 violin 有 2 个 PolyCollection（轮廓和填充）
        patch.set_facecolor(color)
    # sns.kdeplot(x=maxidx[:,0], fill=True, color='gray')  # fill=True 表示填充曲线下面积
    # sns.kdeplot(x=maxidx[:,1], fill=True, color=colors[0])  # fill=True 表示填充曲线下面积
    axs.set_xticks([0,1,2],['L0','L1','L2'])
    axs.set_xlabel('Task')
    axs.set_ylabel('Sig. time (s)')
    
    axs.set_ylim([2.8, 4.3])
    
print(stats.ttest_ind(maxidx[:,2], maxidx[:,1]))
print(maxidx.mean(0))
#%% 比较显著时间的早晚， 泛化到L2中的L1
path = r'F:\Human_chunking_Huashan_intraoperative\Result\bd_neuron_PCA\fig2'
file = path + '\\PCA_155drift_CorrectTrials_TopPCgen_AL5_AllPerm_PCmax5.mat'
data = sio.loadmat(file)

qt = 1
pc1 = 1
pctask1 = 0
plottask1 = 0
plottask2 = 1
plottask3 = 2

### L0
ch_sel = data['chan_idx_set'][0,qt][pctask1,pc1][:,0] - 1
fr = data['data_set'][0, plottask1][ch_sel,10:90][:,30:50] 
### L1/L2
# ch_sel = data['chan_idx_set'][0,qt][pctask2,pc2][:,0] - 1
fr2 = data['data_set'][0, plottask2][ch_sel,10:90][:,30:50]

# ch_sel = data['chan_idx_set'][0,qt][pctask3,pc3][:,0] - 1
fr3 = data['data_set'][0, plottask3][ch_sel,10:90][:,30:50]


maxidx = np.zeros([100,3])
for r in range(100):
    perm_idx = np.random.default_rng().choice(fr.shape[0], size=5, replace=False)
    maxidx[r,0] = fr[perm_idx,:].mean(0).argmax()*0.05+3
    maxidx[r,1] = fr2[perm_idx,:].mean(0).argmax()*0.05+3
    maxidx[r,2] = fr3[perm_idx,:].mean(0).argmax()*0.05+3

with plt.style.context(style_path):
    import seaborn as sns
    
    fig, axs = plt.subplots(1, 1, figsize=(1.4, 1.5), dpi=300, constrained_layout=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    
    df = pd.DataFrame({'x2':maxidx[:,1],'x3':maxidx[:,2]})
    df_melted = df.melt(
            var_name='Variable', value_name='Value')
    
    # 画小提琴图
    color1 = [colors[0], colors[1]]
    axs = sns.violinplot(x='Variable', y='Value', ax=axs, data=df_melted)
    
    color1 = [colors[0], colors[1]]
    for patch, color in zip(axs.collections[::1], color1):  # 每个 violin 有 2 个 PolyCollection（轮廓和填充）
        patch.set_facecolor(color)
    # sns.kdeplot(x=maxidx[:,0], fill=True, color='gray')  # fill=True 表示填充曲线下面积
    # sns.kdeplot(x=maxidx[:,1], fill=True, color=colors[0])  # fill=True 表示填充曲线下面积
    axs.set_xticks([0,1],['L1','L2'])
    axs.set_xlabel('Task')
    axs.set_ylabel('Peak time (s)')
    
    axs.set_ylim([2.4, 4.5])
print(stats.ttest_ind(maxidx[:,0], maxidx[:,2]))
print(maxidx.mean(0))
#%% plot exp var
#### plot exp var, bar
path = r'F:\Human_chunking_Huashan_intraoperative\Result\bd_neuron_PCA\fig2'
file = path + '\\PCA_155drift_CorrectTrials_TopPCgen_AL5_AllPerm_PCmax5.mat'
data = sio.loadmat(file)
exp_var1 = data['PC_output'][4,0][:40,0]
exp_var2 = data['PC_output'][4,1][:40,0]
exp_var3 = data['PC_output'][4,2][:40,0]

path = r'F:\Human_chunking_Huashan_intraoperative\Result\bd_neuron_PCA\fig2'
file = path + '\\PCA_155drift_CorrectTrials_Stacked_TopPCgen_AL5_AllPerm.mat'
data = sio.loadmat(file)
exp_var3_2 = data['PC_output'][4,2][:40,0]

with plt.style.context(style_path):
    # fig, axs = plt.subplots(1, 1, figsize=(1.8, 1.2), dpi=300, constrained_layout=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    
    fig = plt.figure(figsize=(1.8, 1.2), dpi=300)  # 整体图像大小（可按需改）
    plt.subplots_adjust(wspace=0.1, hspace=0)
    gs = GridSpec(1, 3, width_ratios=[1,2,3])  # 上面是下面的3倍高度
    ax1 = fig.add_subplot(gs[0, 0])
    ax2= fig.add_subplot(gs[0, 1])
    ax3= fig.add_subplot(gs[0, 2])
    
    # y = np.array([exp_var1[0],exp_var2[0],exp_var2[1],exp_var3[4],exp_var3[1],exp_var3[4]])
    # axs.bar([0,1.5,2.6,4.1,5.2,6.3], y, color=colors)   
    # axs.set_xlim([-0.8,7.5])
    # axs.set_xticks([0,1.5,2.6,4.1,5.2,6.3],['PC1','PC1','PC2','PC5','PC2','PC3'])
    
    ax1.bar([0.5], exp_var1[0], width=0.5, color='k')   
    ax1.set_xlim([0,1])
    ax1.set_ylim([0,18])
    ax1.set_xticks([0.5],['PC1'])
    
    ax2.bar([0.5,1.5], exp_var2[0:2], width=0.5, color=['k', colors[0]])  
    ax2.set_xlim([0,2])
    ax2.set_ylim([0,18])
    ax2.set_xticks([0.5,1.5],['PC1','PC2'])
    ax2.set_yticklabels([])
    
    ax3.bar([0.5,1.5,2.5], [exp_var3[4],exp_var3[1],exp_var3_2[1]], width=0.5, color=['k', colors[0], colors[1]])  
    ax3.set_xlim([0,3])
    ax3.set_ylim([0,18])
    ax3.set_xticks([0.5,1.5,2.5],['PC5','PC2','PC3'])
    ax3.set_yticklabels([])


#### plot exp var
with plt.style.context(style_path):
    fig, axs = plt.subplots(1, 1, figsize=(1.4, 1.3), dpi=300, constrained_layout=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
   
    axs.plot(
    np.arange(len(exp_var1)), exp_var1,
    color='gray',        # 线的颜色
    marker='o',          # marker 形状
    markerfacecolor='gray',  # marker 填充颜色
    markeredgecolor='gray',  # marker 边框颜色
    markersize=2,         # marker 大小
    linewidth=0.7, zorder=1)
    
    axs.scatter(0, exp_var1[0], s=40, color='k', zorder=2)
    
    axs.set_xlim(-3, 40)
    axs.set_ylim(-1, 18)
    axs.set_xlabel('PCs')
    axs.set_ylabel('Exp.var')

with plt.style.context(style_path):
    fig, axs = plt.subplots(1, 1, figsize=(1.4, 1.3), dpi=300, constrained_layout=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
   
    axs.plot(
    np.arange(len(exp_var2)), exp_var2,
    color='gray',        # 线的颜色
    marker='o',          # marker 形状
    markerfacecolor='gray',  # marker 填充颜色
    markeredgecolor='gray',  # marker 边框颜色
    markersize=2,         # marker 大小
    linewidth=0.7, zorder=1)
    
    axs.scatter(0, exp_var2[0], s=40, color='k', zorder=2)
    axs.scatter(1, exp_var2[1], s=40, color=colors[0], zorder=2)
    
    axs.set_xlim(-3, 40)
    axs.set_ylim(-1, 18)
    axs.set_xlabel('PCs')
    axs.set_ylabel('Exp.var')


#%% 比较显著时间的早晚 L1 correct vs. error
path = r'F:\Human_chunking_Huashan_intraoperative\Result\bd_neuron_PCA\fig2'
file = path + '\\PCA_155drift_CorrectTrials_TopPCgen_AL5_AllPerm_PCmax5.mat'
data = sio.loadmat(file)

qt = 0
pc = 0
pctask = 0
plottask1 = 2

ch_sel = data['chan_idx_set'][0,qt][pctask,pc][:,0] - 1
### correct
fr = data['data_set'][0, plottask1][ch_sel,10:90] 
### error
# fr2 = data['data_set'][1, plottask1][ch_sel,10:90]
fr2 = data['data_set'][0, pctask][ch_sel,10:90]

maxidx = np.zeros([100,2])
for r in range(100):
    perm_idx = np.random.default_rng().choice(fr.shape[0], size=10, replace=False)
    maxidx[r,0] = fr[perm_idx,:].mean(0).argmax()
    maxidx[r,1] = fr2[perm_idx,:].mean(0).argmax()

with plt.style.context(style_path):
    import seaborn as sns
    
    fig, axs = plt.subplots(1, 1, figsize=(1.5, 1.2), dpi=300, constrained_layout=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    
    sns.kdeplot(x=maxidx[:,0], fill=True, color=colors[0])  # fill=True 表示填充曲线下面积
    sns.kdeplot(x=maxidx[:,1], fill=True, color='gray')  # fill=True 表示填充曲线下面积
    
    axs.set_xlim([40, 90])
stats.ttest_ind(maxidx[:,0], maxidx[:,1])
#%% plot cross task
## 'GeoRand'	'Geo4'	'Geo2'	'LanRand'	'LanRule'	'MusRand'	'Mus4'	'Mus2'
path = r'F:\Human_chunking_Huashan_intraoperative\Result\bd_neuron_PCA\Cross_task(fig5)'
# file = path + '\\PCA_155drift_AllTrials_TopPCgen_AL5_AllPerm'
# file = path + '\\PCA_155drift_AllTrials_TopPCgen_AL5_AllPerm_RespTrim'
file = path + '\\PCA_10subj_3tasks_160drift_AllTrials_TopPCgen_AL5_AllPerm_trim'
# file = path + '\\PCA_10subj_3tasks_160drift_AllTrials_Stacked_TopPCgen_AL5_AllPerm_trim'
data = sio.loadmat(file)

qt = 1
pc = 1
pctask = 1
plottask1 = 7
# plottask2 = 2
plot_correct = 0 # 0=correct, 1=error

ch_sel = data['chan_idx_set'][0,qt][pctask,pc][:,0] - 1
# ch_sel = np.append(ch_sel,[66,84])

### correct/error
fr = data['data_set'][plot_correct, plottask1][ch_sel,10:95]
# fr_rnd = data['data_set'][plot_correct, plottask2][ch_sel,10:95]

scaler = StandardScaler()
y = scaler.fit_transform(fr.T).T
# y_rnd = scaler.fit_transform(fr_rnd.T).T
  
#### plot
pval = data['pval_hold_set'][plot_correct,qt][pctask,pc,plottask1][0,:] ### correct
# reject, pval, alphacSidak, alphacBonf = multipletests(pval, alpha=0.05, method='fdr_bh')
# cw = 4
cw = 3
# pvalmsk = pval<0.0
pvalmsk = pval<0.025
pvalmsk2 = pval<0.005

# neuNum = y.shape[0]
with plt.style.context(style_path):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][0:]
    colors = ['gray'] + colors
    # fig, axsall = plt.subplots(2,1, figsize=(1.6, 1.7), dpi=300) # , constrained_layout=True
    fig = plt.figure(figsize=(1.5, 1.3), dpi=300)  # 整体图像大小（可按需改）
    plt.subplots_adjust(wspace=0.0, hspace=0.4)
    
    gs = GridSpec(2, 1, height_ratios=[1, 1])  # 上面是下面的3倍高度
    ax_top = fig.add_subplot(gs[0, 0])
    ax_bottom = fig.add_subplot(gs[1, 0])
    
    #### plot single neuron
    axs = ax_top
    cmap = plt.cm.viridis
    yy = y[~np.isnan(y[:,0]),:]
    # yy_rnd = y_rnd[~np.isnan(y_rnd[:,0]),:]
    neuNum = yy.shape[0]
    
    vmin, vmax = 0.5, 2
    h = axs.pcolormesh(yy, cmap = cmap, vmin=vmin, vmax=vmax)
    axs.vlines(np.array([20, 40, 60]), 0, yy.shape[0], colors='w', linestyle='--',linewidth=1)  
    axs.set_xticks(np.arange(0,80,10), [str(ii) for ii in range(1,9)])
    # axs.set_ylabel('Neuron')
    axs.set_ylabel(' ')
    
    for tt in range(8):
        axs.add_patch(Rectangle((tt*10, 0), 5, 23, edgecolor=None, facecolor="gray", alpha=0.5))
        
    # 将 colorbar 绘制到指定的轴 (cax) 中
    cax = fig.add_axes([0.95, 0.56, 0.02, 0.32])  # 放在主图右侧，宽度很窄
    cbar = fig.colorbar(h, cax=cax)
    cbar.set_label('FR (z-score)')
    
    #### plot mean
    axs = ax_bottom
    plotcolors = 'k'
    plotcolors = colors[1]
    bgcolor = colors[1]
    
    ymean = np.nanmean(yy, 0)
    y_sd = np.nanstd(yy, 0)/np.sqrt(neuNum)
    axs.plot(ymean, color=plotcolors)
    axs.fill_between(np.arange(len(ymean)), ymean-y_sd, ymean+y_sd, color=plotcolors, alpha=0.5)
    # 坐标轴背景色
    # axs.set_facecolor(bgcolor)
    # axs.patch.set_alpha(0.1)
    for tt in range(8):
        axs.add_patch(Rectangle((tt*10, -1.2), 5, 6, edgecolor=None, facecolor="gray", alpha=0.25))
    
    # 找出连续 >=3 个 1 的区间
    indices = []
    start = None
    for i, val in enumerate(pvalmsk):
        if val == 1:
            if start is None:
                start = i
        else:
            if start is not None:
                if i - start > cw:
                    indices.append((start, i))
                start = None
    # 如果最后到结尾还是连续的1
    if start is not None and len(pvalmsk) - start >= 0:
        indices.append((start, len(pvalmsk)-1))
    
    indices2 = []
    start = None
    for i, val in enumerate(pvalmsk2):
        if val == 1:
            if start is None:
                start = i
        else:
            if start is not None:
                if i - start > cw:
                    indices2.append((start, i))
                start = None
    # 如果最后到结尾还是连续的1
    if start is not None and len(pvalmsk2) - start >= 0:
        indices2.append((start, len(pvalmsk2)-1))
        
    # 在 y=1 画横线
    for (s, e) in indices:
        axs.hlines(y=ymean.max()+0.5, xmin=s, xmax=e, colors='gray', linewidth=1)
    for (s, e) in indices2: 
        axs.hlines(y=ymean.max()+0.5, xmin=s, xmax=e, colors='k', linewidth=1)
    
    axs.set_xticks(np.arange(0,80,10), [str(ii) for ii in range(1,9)])
    axs.set_ylim([-1, 1.5])
    # axs.set_ylabel('FR(z-score)')
    axs.set_ylabel(' ')
#%% 比较最早显著时刻，  L0 end vs L1/L2 end
path = r'F:\Human_chunking_Huashan_intraoperative\Result\bd_neuron_PCA\fig3'
file = path + '\\PCA_155drift_CorrectTrials_TopPCgen_AL5_AllPerm_StatExtend.mat'
data = sio.loadmat(file)
perm = data['perm_hold_set'][0,1]

qt = 1
pc1 = 0
pc2 = 0
pc3 = 0
plottask1 = 0
plottask2 = 1
plottask3 = 2

# pc1 = 1
# pc2 = 1
# pc3 = 2
# plottask1 = 0
# plottask2 = 1
# plottask3 = 2

### L0
ch_sel = data['chan_idx_set'][0,qt][plottask1,pc1][:,0] - 1
fr = data['data_set'][0, plottask1][ch_sel,10:90][:,:] 
scaler = StandardScaler()
fr = scaler.fit_transform(fr.T).T
### L1/L2
# ch_sel = data['chan_idx_set'][0,qt][plottask2,pc2][:,0] - 1
fr2 = data['data_set'][1, plottask2][ch_sel,10:90][:,:]
scaler = StandardScaler()
fr2 = scaler.fit_transform(fr2.T).T

# ch_sel = data['chan_idx_set'][0,qt][plottask3,pc3][:,0] - 1
fr3 = data['data_set'][0, plottask3][ch_sel,10:90][:,:]
scaler = StandardScaler()
fr3 = scaler.fit_transform(fr3.T).T

perm1 = perm[0,pc1,0]
perm2 = perm[1,pc1,1]
perm3 = perm[2,pc1,2]

maxidx = np.zeros([50,3])+np.nan
for r in range(50):
    perm_idx = np.random.default_rng().choice(fr.shape[0], size=10, replace=False)
    tmp = fr[perm_idx,:].mean(0)
    tmp2 = fr2[perm_idx,:].mean(0)
    tmp3 = fr3[perm_idx,:].mean(0)
    
    pval3t = np.zeros([80,3,1000]) 
    for i in range(80):
        pval3t[i,0,r] = np.sum(perm1[:,i]>tmp[i])/1000
        pval3t[i,1,r] = np.sum(perm2[:,i]>tmp2[i])/1000
        pval3t[i,2,r] = np.sum(perm3[:,i]>tmp3[i])/1000
    
    for j in range(3):
        pval = pval3t[:,j,r]
        pvalmsk = pval<0.005
        
        indices = []
        cw = 4
        start = None
        for i, val in enumerate(pvalmsk):
            if val == 1:
                if start is None:
                    start = i
            else:
                if start is not None:
                    if i - start > cw:
                        indices.append(start)
                    start = None
        # 如果最后到结尾还是连续的1
        if start is not None and len(pvalmsk) - start >= 0:
            indices.append(start)
            
        indices=np.array(indices)
        indices = indices[indices>60]
        if len(indices)>0:
            maxidx[r,j] = np.min(indices)
            
maxidx = maxidx*0.05

with plt.style.context(style_path):
    import seaborn as sns
    
    fig, axs = plt.subplots(1, 1, figsize=(1.4, 1.5), dpi=300, constrained_layout=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    
    df = pd.DataFrame({'x1':maxidx[:,0],'x2':maxidx[:,1],'x3':maxidx[:,2]})
    df_melted = df.melt(
            var_name='Variable', value_name='Value')
    
    # 画小提琴图
    color1 = ['gray', colors[0], colors[1]]
    axs = sns.violinplot(x='Variable', y='Value', ax=axs, data=df_melted)
    
    color1 = ['gray', colors[0], colors[1]]
    for patch, color in zip(axs.collections[::1], color1):  # 每个 violin 有 2 个 PolyCollection（轮廓和填充）
        patch.set_facecolor(color)
    # sns.kdeplot(x=maxidx[:,0], fill=True, color='gray')  # fill=True 表示填充曲线下面积
    # sns.kdeplot(x=maxidx[:,1], fill=True, color=colors[0])  # fill=True 表示填充曲线下面积
    axs.set_xticks([0,1,2],['L0','L1','L2'])
    axs.set_xlabel('Task')
    axs.set_ylabel('Sig. time (s)')
    
    axs.set_ylim([2.5, 4.3])
    
print(stats.ttest_ind(maxidx[:,2], maxidx[:,1]))
print(maxidx.mean(0))
#%% 比较最早显著时刻，  L1 end: correct vs. error
path = r'F:\Human_chunking_Huashan_intraoperative\Result\bd_neuron_PCA\fig3'
file = path + '\\PCA_155drift_CorrectTrials_TopPCgen_AL5_AllPerm_StatExtend.mat'
file_err = path + '\\PCA_155drift_ErrorTrials_perm_hold_cut.mat'
data = sio.loadmat(file)
perm = data['perm_hold_set'][0,1]

data_err = sio.loadmat(file_err)
permerr = data_err['temp2']

qt = 1
pc1 = 0
pc2 = 0
# pc3 = 0
plottask1 = 1
plottask2 = 1
# plottask3 = 2

# pc1 = 1
# pc2 = 1
# pc3 = 2
# plottask1 = 0
# plottask2 = 1
# plottask3 = 2

### L0
ch_sel = data['chan_idx_set'][0,qt][0,pc1][:,0] - 1
fr = data['data_set'][0, plottask1][ch_sel,10:90][:,:] 
scaler = StandardScaler()
fr = scaler.fit_transform(fr.T).T
### L1/L2
# ch_sel = data['chan_idx_set'][0,qt][plottask2,pc2][:,0] - 1
fr2 = data['data_set'][1, plottask2][ch_sel,10:90][:,:]
scaler = StandardScaler()
fr2 = scaler.fit_transform(fr2.T).T
fr2 = fr2

# # ch_sel = data['chan_idx_set'][0,qt][plottask3,pc3][:,0] - 1
# fr3 = data['data_set'][0, plottask3][ch_sel,10:90][:,:]
# scaler = StandardScaler()
# fr3 = scaler.fit_transform(fr3.T).T

perm1 = perm[1,pc1,1]
perm2 = permerr[1,pc1,1]
# perm3 = perm[2,pc1,2]

maxidx = np.zeros([50,3])+np.nan
for r in range(50):
    perm_idx = np.random.default_rng().choice(fr.shape[0], size=10, replace=False)
    tmp = fr[perm_idx,:].mean(0)
    tmp2 = fr2[perm_idx,:].mean(0)
    tmp3 = fr3[perm_idx,:].mean(0)
    
    pval3t = np.zeros([80,3,1000]) 
    for i in range(80):
        pval3t[i,0,r] = np.sum(perm1[:,i]>tmp[i])/1000
        pval3t[i,1,r] = np.sum(perm2[:,i]>tmp2[i])/1000
        # pval3t[i,2,r] = np.sum(perm3[:,i]>tmp3[i])/1000
    
    for j in range(2):
        pval = pval3t[:,j,r]
        pvalmsk = pval<0.005
        
        indices = []
        cw = 4
        start = None
        for i, val in enumerate(pvalmsk):
            if val == 1:
                if start is None:
                    start = i
            else:
                if start is not None:
                    if i - start > cw:
                        indices.append(start)
                    start = None
        # 如果最后到结尾还是连续的1
        if start is not None and len(pvalmsk) - start >= 0:
            indices.append(start)
            
        indices=np.array(indices)
        indices = indices[indices>60]
        if len(indices)>0:
            maxidx[r,j] = np.min(indices)
            
maxidx = maxidx*0.05

with plt.style.context(style_path):
    import seaborn as sns
    
    fig, axs = plt.subplots(1, 1, figsize=(1.4, 1.5), dpi=300, constrained_layout=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    
    df = pd.DataFrame({'x1':maxidx[:,0],'x2':maxidx[:,1]})
    df_melted = df.melt(
            var_name='Variable', value_name='Value')
    
    # 画小提琴图
    color1 = [colors[0], '#a6bddb']
    axs = sns.violinplot(x='Variable', y='Value', ax=axs, data=df_melted, width=0.5)
    
    for patch, color in zip(axs.collections[::1], color1):  # 每个 violin 有 2 个 PolyCollection（轮廓和填充）
        patch.set_facecolor(color)
    # sns.kdeplot(x=maxidx[:,0], fill=True, color='gray')  # fill=True 表示填充曲线下面积
    # sns.kdeplot(x=maxidx[:,1], fill=True, color=colors[0])  # fill=True 表示填充曲线下面积
    axs.set_xticks([0,1],['Correct','Error'])
    axs.set_xlabel('Sequence (L1)')
    axs.set_ylabel('Sig. time (s)')
    
    axs.set_ylim([2.5, 4.3])
    
print(stats.ttest_ind(maxidx[:,0], maxidx[:,1], nan_policy='omit'))
print(maxidx.mean(0))
#%% 比较最早显著时刻， L1 pc,  L1 vs L2 
path = r'F:\Human_chunking_Huashan_intraoperative\Result\bd_neuron_PCA\fig3'
file = path + '\\PCA_155drift_CorrectTrials_TopPCgen_AL5_AllPerm_StatExtend.mat'
data = sio.loadmat(file)
perm = data['perm_hold_set'][0,1]

qt = 1
pc1 = 1
pc2 = 1
pc3 = 1
plottask1 = 0
plottask2 = 1
plottask3 = 2

# pc1 = 1
# pc2 = 1
# pc3 = 2
# plottask1 = 0
# plottask2 = 1
# plottask3 = 2

ch_sel = data['chan_idx_set'][0,qt][plottask2,pc2][:,0] - 1
### L0
fr = data['data_set'][0, plottask1][ch_sel,10:90][:,:] 
scaler = StandardScaler()
fr = scaler.fit_transform(fr.T).T
### L1/L2
# ch_sel = data['chan_idx_set'][0,qt][plottask2,pc2][:,0] - 1
fr2 = data['data_set'][0, plottask2][ch_sel,10:90][:,:]
scaler = StandardScaler()
fr2 = scaler.fit_transform(fr2.T).T

# ch_sel = data['chan_idx_set'][0,qt][plottask3,pc3][:,0] - 1
fr3 = data['data_set'][0, plottask3][ch_sel,10:90][:,:]
scaler = StandardScaler()
fr3 = scaler.fit_transform(fr3.T).T

perm1 = perm[0,pc1,0]
perm2 = perm[1,pc1,1]
perm3 = perm[2,pc1,2]

maxidx = np.zeros([1000,3])+np.nan
for r in range(1000):
    perm_idx = np.random.default_rng().choice(fr.shape[0], size=10, replace=False)
    tmp = fr[perm_idx,:].mean(0)
    tmp2 = fr2[perm_idx,:].mean(0)
    tmp3 = fr3[perm_idx,:].mean(0)
    
    pval3t = np.zeros([80,3,1000]) 
    for i in range(80):
        pval3t[i,0,r] = np.sum(perm1[:,i]>tmp[i])/1000
        pval3t[i,1,r] = np.sum(perm2[:,i]>tmp2[i])/1000
        pval3t[i,2,r] = np.sum(perm3[:,i]>tmp3[i])/1000
    
    for j in range(3):
        pval = pval3t[:,j,r]
        pvalmsk = pval<0.005
        
        indices = []
        cw = 4
        start = None
        for i, val in enumerate(pvalmsk):
            if val == 1:
                if start is None:
                    start = i
            else:
                if start is not None:
                    if i - start > cw:
                        indices.append(start)
                    start = None
        # 如果最后到结尾还是连续的1
        if start is not None and len(pvalmsk) - start >= 0:
            indices.append(start)
            
        indices=np.array(indices)
        indices = indices[(indices>20)&(indices<60)]
        if len(indices)>0:
            maxidx[r,j] = np.min(indices)
            
maxidx = maxidx*0.05
idx = np.where((maxidx[:,1]>0) & (maxidx[:,2]>0))
idx = idx[:50]
maxidx = maxidx[idx]
#### L1 PC
with plt.style.context(style_path):
    import seaborn as sns
    
    fig, axs = plt.subplots(1, 1, figsize=(1.4, 1.5), dpi=300, constrained_layout=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    
    df = pd.DataFrame({'x2':maxidx[:,1],'x3':maxidx[:,2]})
    df_melted = df.melt(
            var_name='Variable', value_name='Value')
    
    # 画小提琴图
    color1 = [colors[0], colors[1]]
    axs = sns.violinplot(x='Variable', y='Value', ax=axs, data=df_melted)
    
    color1 = [ colors[0], colors[1]]
    for patch, color in zip(axs.collections[::1], color1):  # 每个 violin 有 2 个 PolyCollection（轮廓和填充）
        patch.set_facecolor(color)
    # sns.kdeplot(x=maxidx[:,0], fill=True, color='gray')  # fill=True 表示填充曲线下面积
    # sns.kdeplot(x=maxidx[:,1], fill=True, color=colors[0])  # fill=True 表示填充曲线下面积
    axs.set_xticks([0,1],['L1','L2'])
    axs.set_xlabel('Task')
    axs.set_ylabel('Sig. time (s)')
    
    axs.set_ylim([1, 3])
    
print(stats.ttest_ind(maxidx[:,2], maxidx[:,1]))
#%% 
# path = r'F:\Human_chunking_Huashan_intraoperative\Result\bd_neuron_PCA\fig3'
path = r'F:\Human_chunking_Huashan_intraoperative\Result\bd_neuron_PCA\fig2'
file = path + '\\PCA_155drift_CorrectTrials_TopPCgen_AL5_AllPerm(1)'
# file = path + '\\PCA_155drift_CorrectTrials_TopPCgen_AL5_AllPerm_PCmax5.mat'
# file = path + '\\PCA_155drift_CorrectTrials_Stacked_TopPCgen_AL5_AllPerm.mat'
# file = path + '\\PCA_155drift_AllTrials_Stacked_TopPCgen_AL5_AllPerm.mat'
data = sio.loadmat(file)


qt = 1
pc = 1
pctask = 2
plottask1 = 2
plottask2 = 2
plot_correct = 0 # 0=correct, 1=error
ch_sel = data['chan_idx_set'][0,qt][pctask,pc][:,0] - 1


path = r'F:\Human_chunking_Huashan_intraoperative\Data\Recording\PFC'
data_fr = pkl.load(open(path+'\\fr_beha_PFC(9sub_155)_errormark_drift.pkl', 'rb'))
neu_info = data_fr['neu_info']

neu_info = pd.concat(neu_info)
neu_info = neu_info.reset_index(drop=True)

tmp = []
for i in range(len(ch_sel)):
    tmp.append(neu_info.loc[ch_sel[i],'subID'])






per4 = [4/20, 6/67, 5/25, 1/16]
