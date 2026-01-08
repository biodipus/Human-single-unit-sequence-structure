# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 10:54:26 2025

@author: Wen
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

# sys.path.append("..") 
# from spca.spca import My_supervised_PCA

#%% plot cross rule
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import StandardScaler, MinMaxScaler

## 'GeoRand'	'Geo4'	'Geo2'	'LanRand'	'LanRule'	'MusRand'	'Mus4'	'Mus2'
path = r'F:\Human_chunking_Huashan_intraoperative\Result\bd_neuron_PCA\fig2'
# file = path + '\\PCA_155drift_AllTrials_TopPCgen_AL5_AllPerm_RespTrim_PCMax6.mat'
file = path + '\\PCA_10subj_3tasks_160drift_AllTrials_TopPCgen_AL5_AllPerm_trim.mat'

data = sio.loadmat(file)
qt = 1

pc = 0
dotask = 0
if dotask == 0:
    pctask = 0
    fr_task = 'fr_allsub_ir'
    reduce_task = 'fr_reduce_ir'
    fit_result = 'fit_result_a'

if dotask ==1:
    pctask = 1
    fr_task = 'fr_allsub_r4'
    reduce_task = 'fr_reduce_r4'
    fit_result = 'fit_result_b'

if dotask ==2:
    pctask = 2
    fr_task = 'fr_allsub_r2'
    reduce_task = 'fr_reduce_r2'
    fit_result = 'fit_result_c'
    
ch_sel = data['chan_idx_set'][0,qt][pctask,pc][:,0] - 1
pca_project = data['PC_output'][0, pctask][:, [pc]]
#%%
data_spca = pkl.load(open(r'F:\Human_chunking_Huashan_intraoperative\Result\spca\10sub_geo_fit(time40)_correct(1_3)_PermData.pkl','rb'))
trialNum = data_spca[fr_task][0].shape[1]

#%%
r4_activity = np.zeros([trialNum,2, 50])
for rep in range(50):
    for i in range(trialNum):
        r4_trial = data_spca[fr_task][rep][:,i,:]
        
        scaler = StandardScaler()
        y = scaler.fit_transform(r4_trial.T)
        tmp = np.dot(y, pca_project)
        r4_activity[i,0,rep] = tmp[15:25].max()
        # r4_activity[i,0,rep] = tmp[35:].max()
        
        #### 到第一个平面的距离
        plane1 = data_spca[fit_result][rep]['output']
        points = data_spca[reduce_task][rep][:3,i,20:].T
        # centroid = points.mean(axis=0)  # 平面上一点
        centered = points# - centroid
        
        normal = plane1['m2_normal1']     # 法向量 = 最小特征值对应方向
        axis1 = plane1['m2_1'][0]        # 平面内第一主方向
        axis2 = plane1['m2_1'][1]          # 平面内第二主方向
        
        residuals = np.abs(centered @ normal)   # 投影长度
        residuals /= np.linalg.norm(normal)
        residual_variance = np.mean(residuals**2)
        
        #### 前后两段的均值的距离
        r4_1 = data_spca[reduce_task][rep][:3,i,:5].mean(1)
        r4_2 = data_spca[reduce_task][rep][:3,i,20:25].mean(1)
        dis = np.sqrt(np.sum((r4_1 - r4_2)**2))
        
        ####
        r4_activity[i,1,rep] = residuals[:].mean()


# plt.scatter(r4_activity[:,0,0], r4_activity[:,1,0])
rval = np.zeros([50,1])
pval = np.zeros([50,1])
for rep in range(50):
    rval[rep,0], pval[rep,0] = stats.pearsonr(r4_activity[:,0,rep], r4_activity[:,1,rep])

plt.hist(rval)  

stats.ttest_1samp(rval, 0)  
# y_selch = y[:,ch_sel]
# plt.plot(y_selch.mean(1))
#%% plot 
with plt.style.context(style_path):
    from scipy.stats import gaussian_kde
    
    fig, axs = plt.subplots(1,1, figsize=(1.5, 1.2), dpi=300, constrained_layout=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = ['gray']
    
    dat = rval[:,0]
    # 画直方图
    axs.hist(dat, bins=10, density=True, alpha=0.4, color=colors[0], label='Histogram')
    
    kde = gaussian_kde(dat, bw_method=0.6)
    x = np.linspace(-0.8, 1.3, 200)
    y = kde(x)
    axs.plot(x, y, color=colors[0], linewidth=2, label='KDE')
    
    axs.scatter(dat.mean(), 2.2, marker='v', color='black', s=20, zorder=5)
    
    axs.set_xticks([-0.5,0,0.5,1])
    axs.set_xlabel('Corr. coefficient')
    axs.set_ylabel('Density')
    
    print(dat.mean())
#%% plot example regression     
idx = 17#np.argmax(dat)
act = r4_activity[:,:,idx]
with plt.style.context(style_path):
    fig, axs = plt.subplots(1,1, figsize=(1.5, 1.2), dpi=300, constrained_layout=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][dotask-1:]
    
    x1,x2 = act[:,0], act[:,1]
    axs.scatter(x1,x2)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(x1, x2)
    fit_line = slope * x1 + intercept
    
    axs.scatter(x1, x2, color=colors[0], edgecolor='None', s=60, label='Data')
    axs.plot(x1, fit_line, color='k', lw=2, label=f'Fit line (r={r_value:.2f})')
    
    axs.set_xlim([x1.min()-0.3, x1.max()+0.5])
    axs.set_ylim([x2.min()-0.3, x2.max()+0.2])
    
    axs.set_xlabel('Struture activity')
    axs.set_ylabel('Distance')
    
    print(rval[idx,0], pval[idx,0])
    
        




