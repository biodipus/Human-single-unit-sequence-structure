"""
Multi-label Supervised PCA
Modified for multi-label Y (Y has multiple label dimensions)
Y: rows are label dimensions, columns are samples
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

import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy.linalg import eigh, sqrtm, inv

#%%
class MultiLabelMultiClassSPCA:
    def __init__(self, n_components, label_weights=None,
                 center=True, scale=True, eps=1e-8):
        """
        Multi-label, multi-class Supervised PCA

        n_components : int
            Number of supervised principal components
        label_weights : list or None
            权重 α_l，用于不同 label 的重要性加权
        """
        self.n_components = n_components
        self.label_weights = label_weights
        self.center = center
        self.scale = scale
        self.eps = eps

        self.components_ = None
        self.explained_variance_ = None

    def _center_kernel(self, K):
        n = K.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        return H @ K @ H

    def fit(self, X, Y_list):
        """
        X      : (n_samples, n_features)
        Y_list : list of label arrays
                 每个元素 Y_l:
                   - shape (n_samples,)  → categorical
                   - 或 (n_samples, K_l) → 已 one-hot / 连续
        """
        n, p = X.shape

        # -------- X preprocessing --------
        if self.center or self.scale:
            scaler = StandardScaler(with_mean=self.center,
                                    with_std=self.scale)
            X = scaler.fit_transform(X)

        # Center X
        H = np.eye(n) - np.ones((n, n)) / n
        Xc = H @ X
        
        # -------- build label kernel --------
        if self.label_weights is None:
            self.label_weights = [1.0] * len(Y_list)

        L = np.zeros((n, n))

        for Y, w in zip(Y_list, self.label_weights):

            # case 1: categorical → one-hot
            if Y.ndim == 1:
                enc = OneHotEncoder(sparse_output=False, drop=None)
                Y_oh = enc.fit_transform(Y[:, None])
            else:
                Y_oh = Y

            K = Y_oh @ Y_oh.T
            Kc = self._center_kernel(K)

            L += w * Kc

        # -------- supervised covariance --------
        M = Xc.T @ L @ Xc
        Cx = Xc.T @ Xc + self.eps * np.eye(p)

        # -------- generalized eigenproblem --------
        Cx_inv_sqrt = inv(sqrtm(Cx))
        A = Cx_inv_sqrt @ M @ Cx_inv_sqrt

        eigvals, eigvecs = eigh(A)
        idx = np.argsort(eigvals)[::-1][:self.n_components]
        self.explained_variance_ = eigvals[idx]
        self.components_ = Cx_inv_sqrt @ eigvecs[:, idx]
        self.eigvecs = eigvecs[:, idx]

        return self

    def transform(self, X):
        return X @ self.components_

    def fit_transform(self, X, Y_list):
        self.fit(X, Y_list)
        return self.transform(X)

def pc_label_contributions(spca, X, Y_list):
    """
    返回 shape = (n_components, n_labels)
    """
    n = X.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    Xc = H @ X

    W = spca.components_
    n_pc = W.shape[1]
    n_label = len(Y_list)

    scores = np.zeros((n_pc, n_label))

    for l, Y in enumerate(Y_list):
        if Y.ndim == 1:
            from sklearn.preprocessing import OneHotEncoder
            Y = OneHotEncoder(sparse_output=False).fit_transform(Y[:, None])

        K = Y @ Y.T
        Kc = H @ K @ H

        for k in range(n_pc):
            wk = W[:, k]
            scores[k, l] = wk.T @ Xc.T @ Kc @ Xc @ wk

    # normalize per PC
    # scores /= scores.sum(axis=1, keepdims=True)
    return scores

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
            mu_c = z[idx].mean()
            var_exp[k] += (pc * (mu_c - mu_global) ** 2)

    return var_exp

def pc_multi_label_explained_variance(Z, Y_list):
    """
    return:
        var_matrix: (n_components, n_labels)
    """
    n_pc = Z.shape[1]
    n_label = len(Y_list)

    var_mat = np.zeros((n_pc, n_label))

    for l, Y in enumerate(Y_list):
        var_mat[:, l] = pc_label_explained_variance(Z, Y)

    return var_mat

def normalize_variance_by_pc(var_mat, Z):
    """
    Fraction of PC variance explained by each label
    """
    pc_var = Z.var(axis=0)[:, None]   # (n_components, 1)
    return var_mat / pc_var

def pc_label_predictable_variance(Z, Y, alpha=1.0):
    from sklearn.linear_model import Ridge

    """
    Z: (n_samples, n_components)
    Y: (n_samples, d_l) continuous or one-hot
    """
    n_pc = Z.shape[1]
    var_pred = np.zeros(n_pc)

    for k in range(n_pc):
        z = Z[:, k]
        reg = Ridge(alpha=alpha, fit_intercept=True)
        reg.fit(Y, z)
        z_hat = reg.predict(Y)
        var_pred[k] = np.var(z_hat)

    return var_pred
#%% 多label spca,采用

import numpy as np
from numpy import linalg as LA

class My_supervised_PCA:

    def __init__(self, n_components=None, kernel_on_labels=None):
        self.n_components = n_components
        self.U = None
        self.mean_of_X = None
        if kernel_on_labels is not None:
            self.kernel_on_labels = kernel_on_labels
        else:
            self.kernel_on_labels = "linear"

    def fit_transform(self, X, Y):
        self.fit(X, Y)
        X_transformed = self.transform(X, Y)
        return X_transformed, self.V, self.U, self.V_raw, self.U_raw

    def delta_kernel_multi_label(self, Y):
        """
        Compute a multi-label delta kernel.
        Y: shape (n_label_dims, n_samples)
        Returns B: (n_samples, n_samples), where B[i,j] = number of labels on which sample i and j agree.
        Optionally normalize by number of labels.
        """
        n_label_dims, n_samples = Y.shape
        B = np.zeros((n_samples, n_samples))
        
        # For each label dimension, compute agreement
        for d in range(n_label_dims):
            y_d = Y[d, :]  # shape (n_samples,)
            # Broadcast comparison: (n_samples, 1) == (1, n_samples)
            agreement = (y_d[:, None] == y_d[None, :]).astype(float)
            B += agreement
        
        # Optional: normalize by number of labels to keep B in [0,1]
        # B /= n_label_dims
        
        return B

    def fit(self, X, Y):
        # X: (n_features, n_samples)
        # Y: (n_label_dims, n_samples)
        self.mean_of_X = X.mean(axis=1).reshape((-1, 1))
        n = X.shape[1]  # number of samples
        H = np.eye(n) - (1.0 / n) * np.ones((n, n))
        
        # Use multi-label delta kernel
        B = self.delta_kernel_multi_label(Y=Y)
        
        # Centered supervised covariance matrix
        M = X @ H @ B @ H @ X.T  # (n_features, n_features)
        
        # Store raw eigenvalues/vectors before sorting
        eig_val_raw, eig_vec_raw = LA.eigh(M)
        self.V_raw = eig_val_raw.copy()
        self.U_raw = eig_vec_raw.copy()
        
        # Sort in descending order
        idx = eig_val_raw.argsort()[::-1]
        eig_val = eig_val_raw[idx]
        eig_vec = eig_vec_raw[:, idx]
        
        if self.n_components is not None:
            self.U = eig_vec[:, :self.n_components]
            self.V = eig_val[:self.n_components]
        else:
            self.U = eig_vec
            self.V = eig_val

    def transform(self, X, Y=None):
        # Note: no centering in original code! But SPCA usually centers X.
        # We follow your original: no explicit centering in transform
        X_transformed = self.U.T @ X
        return X_transformed

    def recon(self, transX, Y=None):
        reconDataMat = self.U @ transX + self.mean_of_X
        return reconDataMat

    def transform_outOfSample_all_together(self, X):
        return self.U.T @ X

    def get_projection_directions(self):
        return self.U

    def reconstruct(self, X, scaler=None, using_howMany_projection_directions=None):
        if using_howMany_projection_directions is not None:
            U = self.U[:, :using_howMany_projection_directions]
        else:
            U = self.U
        X_transformed = U.T @ X
        X_reconstructed = U @ X_transformed
        return X_reconstructed

    def reconstruct_outOfSample_all_together(self, X, scaler=None, using_howMany_projection_directions=None):
        if using_howMany_projection_directions is not None:
            U = self.U[:, :using_howMany_projection_directions]
        else:
            U = self.U
        X_transformed = U.T @ X
        X_reconstructed = U @ X_transformed
        return X_reconstructed

    def center_the_matrix(self, the_matrix, mode="double_center"):
        n_rows, n_cols = the_matrix.shape
        vector_one_left = np.ones((n_rows, 1))
        vector_one_right = np.ones((n_cols, 1))
        H_left = np.eye(n_rows) - (1.0 / n_rows) * vector_one_left @ vector_one_left.T
        H_right = np.eye(n_cols) - (1.0 / n_cols) * vector_one_right @ vector_one_right.T
        if mode == "double_center":
            return H_left @ the_matrix @ H_right
        elif mode == "remove_mean_of_rows_from_rows":
            return H_left @ the_matrix
        elif mode == "remove_mean_of_columns_from_columns":
            return the_matrix @ H_right
        else:
            return the_matrix
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
        
        idx = (info['acc'] == 1)# | (info['acc'] == 3)
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
#%%
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
        y1 = np.tile(np.repeat([1,2,3,4,0,0,0,0], ts_per_rank), int(x.shape[1])).reshape(-1,1)
        # y1 = np.tile(np.repeat([1,2,3,4,1,2,3,4], ts_per_rank), int(x.shape[1])).reshape(-1,1)
        y2 = np.tile(np.repeat([1,1,1,1,2,2,2,2], ts_per_rank), int(x.shape[1])).reshape(-1,1)
        y12 = np.concat([y1,y2], axis=1)
        
        # spca of T1
        # spca = MultiLabelMultiClassSPCA(n_components=167)
        spca = My_supervised_PCA(n_components=167)
        x_reduce, eigvals, eig_vec, _, _ = spca.fit_transform(x_norm, y12.T)
        # contrib = pc_label_contributions(spca, x_norm.T, [y1])
        
        # eig_vec = spca.eigvecs
        for jj in range(neuNum):
            pc1_scores = xtest_norm.T @ eig_vec[:, jj]  # 投影到 PC1
            exp_var[repeat,jj] = np.var(pc1_scores, ddof=1)
        
        exp_var[repeat,:] = exp_var[repeat,:]/exp_var[repeat,:].sum()
        # aa=spca.explained_variance_
        # var_mat = pc_label_predictable_variance(x_reduce.T, y1)
        # frac_var = normalize_variance_by_pc(var_mat, x_reduce.T)
        
        # reshape back to f*time*sample
        x_reduce3d = x_reduce.reshape(x_reduce.shape[0], shape_3d[1], shape_3d[2])
        
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
repetition = 10

# dotask, projecttask = 'geo_ir','geo_ir'
rank3d_value_gir, fea_imp_ir, exp_var_ir,fr_allsub_ir,fr_reduce_ir = spca_perm('geo_ir','geo_ir')
rank3d_value_gr4, fea_imp_r4, exp_var_r4,fr_allsub_r4,fr_reduce_r4 = spca_perm('geo_r4','geo_r4')
rank3d_value_gr2, fea_imp_r2, exp_var_r2,fr_allsub_r2,fr_reduce_r2 = spca_perm('geo_r2','geo_r2')

exp_var = [exp_var_ir, exp_var_r4, exp_var_r2]
exp_var_ir[:,:3].mean(0).sum()
exp_var_r4[:,:3].mean(0).sum()
exp_var_r2[:,:3].mean(0).sum()

# pkl.dump({'fr_reduce_ir':fr_reduce_ir, 'fr_reduce_r4':fr_reduce_r4,'fr_reduce_r2':fr_reduce_r2,'exp_var':exp_var}, 
#          open(r'F:\Human_chunking_Huashan_intraoperative\Result\spca_3Hyp_ExpVar\geo_HP3_correct(1_3).pkl','wb'))

#%% exp var比较
var_hp2 = pkl.load(open(r'F:\Human_chunking_Huashan_intraoperative\Result\spca_3Hyp_ExpVar\geo_HP2_correct(1_3).pkl','rb'))
var_hp2 = var_hp2['exp_var']

var_hp3 = pkl.load(open(r'F:\Human_chunking_Huashan_intraoperative\Result\spca_3Hyp_ExpVar\geo_HP3_correct(1_3).pkl','rb'))
var_hp3 = var_hp3['exp_var']

t,p = stats.ttest_ind(var_hp2[0][:,:3].mean(1), var_hp3[0][:,:3].mean(1))     
t,p = stats.ttest_ind(var_hp2[1][:,:3].mean(1), var_hp3[1][:,:3].mean(1))     
t,p = stats.ttest_ind(var_hp2[2][:,:3].mean(1), var_hp3[2][:,:3].mean(1))     


with plt.style.context(style_path):
    get_ipython().run_line_magic('matplotlib', 'inline')
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(1,1, figsize=(1.2, 1.5), dpi=300)
    
    # df = pd.DataFrame({'hp2':var_hp2[1][:,:3].sum(1), 'hp3':var_hp3[1][:,:3].sum(1)})
    df = pd.DataFrame({'hp2':var_hp2[2][:,:3].sum(1), 'hp3':var_hp3[2][:,:3].sum(1)})
    df_melted = df[['hp2','hp3',]].melt(
            var_name='Variable', value_name='Value')
    
    # 画小提琴图
    color1 = [colors[3],colors[3]]
    ax = sns.violinplot(x='Variable', y='Value', ax=ax, data=df_melted, width=0.5)
    
    for patch, color in zip(ax.collections[::1], color1):  # 每个 violin 有 2 个 PolyCollection（轮廓和填充）
        patch.set_facecolor(color)
    ax.set_ylim([0.1, 0.35])
    ax.set_xticklabels(['Factorized','Relation'], rotation=20)
    ax.set_xlabel('Model')
    ax.set_ylabel('Exp.var')
#%%
rank3d_value = rank3d_value_gr4
# fit_result = fit_result_a

plot_condition  = rank3d_value[:,:,:]
# output = fit_result[-1]['output']
# pc_exp = exp_var[0].mean(0)[:3]*100

ts_per_rank = plot_condition.shape[1]//8
with plt.style.context(style_path):
    get_ipython().run_line_magic('matplotlib', 'inline')
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
    
    # axislim = [-1, 1]
    # ax.set_xlim(axislim)
    # ax.set_ylim(axislim)
    # ax.set_zlim(axislim)
    
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
#%%
var_hp2 = pkl.load(open(r'F:\Human_chunking_Huashan_intraoperative\Result\spca_3Hyp_ExpVar\geo_HP2_correct(1_3).pkl','rb'))
fr_reduce = var_hp2['fr_reduce_r2']

# fr_reduce = fr_reduce_ir
# fit_result = fit_result_b

rep = 10 # r2:15 #7; lan_r4=32, 37
# pc_exp = geo_spca['exp_var'][2][rep]*100

tmp  = fr_reduce[rep][:3,:,:] # neu*trial*time
plot_condition = np.transpose(tmp, (1, 2, 0))

# output = fit_result[rep]['output']
# pc_exp = exp_var[0].mean(0)[:3]*100

ts_per_rank = plot_condition.shape[1]//8
with plt.style.context(style_path):
    from mpl_toolkits.mplot3d import Axes3D, art3d
    plt.rcParams['axes.linewidth'] = 0.3   # 对二维轴生效（若你也画 2D 图）
    plt.rcParams['xtick.major.width'] = 0.3      # X轴主刻度线粗细
    plt.rcParams['ytick.major.width'] = 0.3      # Y轴主刻度线粗细
    # plt.rcParams['ztick.major.width'] = 0.2      # Y轴主刻度线粗细

    get_ipython().run_line_magic('matplotlib', 'inline')
    # get_ipython().run_line_magic('matplotlib', 'qt5')
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
    # ax.view_init(elev=51, azim=-110)
    # ax.view_init(elev=7, azim=-50) # lanr4, rep=32; lanir, 10
    # ax.view_init(elev=36, azim=-45) # musr4, rep=32; lanir, 10
    ax.view_init(elev=25, azim=150) # musir, 6
    
    # axislim = [-8, 8]
    # ax.set_xlim([-8, 8])
    # ax.set_ylim(axislim)
    # ax.set_zlim(axislim)
    
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
#%%
np.random.seed(0)

n = 300
X = np.random.randn(n, 50)

# 多标签，多分类
Y1 = np.random.randint(0, 4, size=n)   # label 1
Y2 = np.random.randint(0, 3, size=n)   # label 2

Y_list = [Y1, Y2]

# ----- SPCA -----
spca = MultiLabelMultiClassSPCA(
    n_components=8,
    label_weights=[1.0, 1.0]
)

Z = spca.fit_transform(X, Y_list)

# ----- label-wise supervised contribution -----
contrib = pc_label_contributions(spca, X, Y_list)

# ----- label-wise explained variance -----
var_mat = pc_multi_label_explained_variance(Z, Y_list)
frac_var = normalize_variance_by_pc(var_mat, Z)

print("PC–label contribution:\n", contrib)
print("PC–label explained variance:\n", var_mat)
print("Fraction of PC variance explained:\n", frac_var)

