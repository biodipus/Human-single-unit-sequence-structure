%% PCA evaluation: AllTrials, top PC gen, with stats, 10 subj
clear all;
file_dir = 'F:\Human_chunking_Huashan_intraoperative\Data\Recording\PFC\Correct_error_meanFR';
% load(sprintf('%s\\kernel_match_deepdive\\kernel_match_15stim_zscore_155_drift.mat',file_dir),...
%     'task_labels');
load(sprintf('%s\\fr_beha_PFC(10sub_geo_160)_responded_drift.mat',file_dir),'fr*');
% load(sprintf('%s\\fr_beha_PFC(10sub_geo_160)_correct_drift.mat',file_dir),'fr*');

% load(sprintf('%s\\fr_beha_PFC(10sub_lan_160)_responded_drift.mat',file_dir),'fr*');
% load(sprintf('%s\\fr_beha_PFC(9sub_geo_155)_responded_drift.mat',file_dir),'fr*');

data_set = cell(1,3); 
task_labels = {'GeoRand','Geo4','Geo2'};
% task_labels = {'LanRand','Lan4'};
% task_labels = {'musRand','musr4','musr2'};
task_labels = task_labels(1:3);

% fr_gr2(167,70:85) = 20;
% fr_gr2(167,20:40) = 8;
% load('149.mat')
% fr_gr2(148:153,:)=repmat(aaa,6,1);

data_set{1} = fr_gir; data_set{2} = fr_gr4; data_set{3} = fr_gr2;
% data_set{1} = fr_lir; data_set{2} = fr_lr4; 
% data_set{1} = fr_mir; data_set{2} = fr_mr4; data_set{3} = fr_mr2;

block_mat_hold = cell(1,8);
data_sel = 11:90;
prop_set = [15];
PC_max = 3;

% alpha_thresh = 0.05; 
alpha_thresh = 0.025; 

consec_thresh = 4;

stat_flag = 1; % 1: all unit sel; 2: non-PC unit sel

stat_labels = {'AllPerm','DiffPerm'};
stat_str = stat_labels{stat_flag};
fig_pos = [59,520,1824,358];
c_range = [-4,4]; x_range = [0,81]; y_range = [-1.2,3.2];
xtick_labels = 0:1:4; xtick_range = 0.5:20:80.5;
time_bins = 80;
stat_line_pos = 3.1;
rng(42,"twister"); % for reproducibility

plot_flag = 1;

PC_output_dim1_labels = {'Coeff','Score','Latent','TSquared','EV','Mu'};
PC_output_dim2_label = {'Tasks'};

num_perm = 1000;
num_units = 167;
num_tasks = length(task_labels);
num_prop = length(prop_set);
PC_output = cell(6,num_tasks);

pval_hold_set = cell(1,num_prop);
chan_idx_set = cell(1,num_prop);
set_cell_labels = {'SourceBlock','PCs','TargetBlock'};

for iTask = 1:num_tasks
%     [coeff, score, latent, tsquared, explained, mu] = pca(block_mat_hold{iTask}');
    temp = data_set{iTask}(:,data_sel);
%     temp = temp(~isnan(temp(:,1)),:);
    temp = zscore(temp,0,2);
    block_mat_hold{iTask} = temp;
    [coeff, score, latent, tsquared, explained, mu] = pca(temp');
    PC_output{1,iTask} = coeff; PC_output{2,iTask} = coeff;
    PC_output{3,iTask} = latent; PC_output{4,iTask} = tsquared;
    PC_output{5,iTask} = explained; PC_output{6,iTask} = mu;
end
    
for iP = 1:num_prop
    perm_hold = cell(num_tasks,PC_max,num_tasks);
    pval_hold = cell(num_tasks,PC_max,num_tasks);
    chan_idx_hold = cell(num_tasks,PC_max);
           
    curr_prop = prop_set(iP);
    prop_val = round(num_units*(curr_prop/100));
    
    if plot_flag == 1
        fig_dir = sprintf('%s\\figs\\PCA_10subj_AllTrials_TopPropCut_%s_AL%d_CS%d\\Top_%d',...
            file_dir,stat_str,round(alpha_thresh*100),consec_thresh,curr_prop);
        png_dir = sprintf('%s\\pngs\\PCA_10subj_AllTrials_TopPropCut_%s_AL%d_CS%d\\Top_%d',...
            file_dir,stat_str,round(alpha_thresh*100),consec_thresh,curr_prop);
        if ~exist(fig_dir,'dir')
            mkdir(fig_dir)
        end
        if ~exist(png_dir,'dir')
            mkdir(png_dir)
        end
    end
        
    for iTask = 1:num_tasks
        curr_task = task_labels{iTask}; 
        coeff = PC_output{1,iTask};
        
        for iPC = 1:PC_max
            [~,idx] = sort(coeff(:,iPC),'descend');
            out = sort(idx(1:prop_val));
            chan_idx_hold{iTask,iPC} = out;
            
            % permutation stats
            num_hits = length(out);
            for iT = 1:num_tasks
                orig_vec = nanmean(block_mat_hold{iT}(out,:),1);
                perm_temp = nan(num_perm,time_bins);
                for iPerm = 1:num_perm
                    if stat_flag == 1
                        rand_sel = randi(num_units,[1,num_hits]);
                    elseif stat_flag == 2
                        temp_sel = randi(num_units-num_hits,[1,num_hits]);
                        temp_pool = setdiff(1:num_units,out);
                        rand_sel = temp_pool(temp_sel);
                    end
                    perm_vec = nanmean(block_mat_hold{iT}(rand_sel,:),1);
                    perm_temp(iPerm,:) = perm_vec;
                end
                perm_hold{iTask,iPC,iT} = perm_temp;
                
                pval_vec = sum(perm_temp > orig_vec, 1)./num_perm;
                pval_hold{iTask,iPC,iT} = pval_vec;
            end
            
            if plot_flag == 1
                h = figure; set(h,'Position',fig_pos);
                for iT = 1:num_tasks
                    subplot(2,num_tasks,iT);
                    imagesc(block_mat_hold{iT}(out,:));
                    caxis(c_range); xlim(x_range);
                    set(gca,'XTick',xtick_range); set(gca,'XTickLabel',xtick_labels);
                    if iT == 1
                        ylabel('PC Coeff-Sel Units');
                    end
                    
                    if iT == iTask
                        title(task_labels{iT},'Color','b');
                    else
                        title(task_labels{iT});
                    end
                    subplot(2,num_tasks,iT+num_tasks)
                    boundedline(1:time_bins,nanmean(block_mat_hold{iT}(out,:),1),nanstd(block_mat_hold{iT}(out,:),0,1)./sqrt(length(out)));
                    hline(0,'k--');
                    hold on; pvec = pval_hold{iTask,iPC,iT}; pvec_hits = find(pvec < alpha_thresh);
                    [class_types,class_tally,class_vec] = linear_consec_clust(pvec_hits);
                    for iClass = 1:max(class_types)
                        if class_tally(iClass) < consec_thresh
                            pvec_hits(class_vec == iClass) = 0;
                        end
                    end
                    pvec_hits(pvec_hits == 0) = [];
                    pvec_adj = nan(1,time_bins); pvec_adj(pvec_hits) = stat_line_pos;
                    plot(pvec_adj,'g','LineWidth',1.5);
                    xlim(x_range); ylim(y_range);
                    set(gca,'XTick',xtick_range); set(gca,'XTickLabel',xtick_labels);
                    if iT == 1
                        ylabel('Mean Norm FR');
                        xlabel('Time (s) from 1st stim on');
                    end
                end
%                 suptitle(sprintf('%s PC%d Common Sel, Top %d perc',curr_task,iPC,curr_prop));
                saveas(h,sprintf('%s\\gen_10subj_160drift_AT_%s_CommonSel_PC%d.fig',fig_dir,curr_task,iPC));
                saveas(h,sprintf('%s\\gen_10subj_160drift_AT_%s_CommonSel_PC%d.png',png_dir,curr_task,iPC));
                close(h);
            end
            
        end
    end
    
    pval_hold_set{iP} = pval_hold;
    chan_idx_set{iP} = chan_idx_hold;
end

% save(sprintf('%s\\PCA_finalize\\PCA_10subj_160drift_AllTrials_TopPCgen_AL%d_%s.mat',file_dir,round(alpha_thresh*100),stat_str),...
%     'task_labels','PC_output_dim1_labels','PC_output_dim2_label',...
%     'PC_max','num_perm','*_set','PC_output','set_cell_labels')

%% PCA evaluation: AllTrials Stacked, top PC gen, with stats, 10 subj
clear all;
file_dir = 'F:\Human_chunking_Huashan_intraoperative\Data\Recording\PFC\Correct_error_meanFR';
% load(sprintf('%s\\kernel_match_deepdive\\kernel_match_15stim_zscore_155_drift.mat',file_dir),...
%     'task_labels');
load(sprintf('%s\\fr_beha_PFC(10sub_geo_160)_responded_drift.mat',file_dir),'fr*');
data_set = cell(1,3); task_labels = {'GeoRand','Geo4','Geo2'};
task_labels = task_labels(1:3);
data_set{1} = fr_gir; data_set{2} = fr_gr4; data_set{3} = fr_gr2;


block_mat_hold = cell(1,8);
data_sel = 11:90;
prop_set = [15:5:15];
PC_max = 3;

alpha_thresh = 0.05;
% alpha_thresh = 0.01; 

consec_thresh = 3;

stat_flag = 1; % 1: all unit sel; 2: non-PC unit sel

stat_labels = {'AllPerm','DiffPerm'};
stat_str = stat_labels{stat_flag};
fig_pos = [59,520,1824,358];
c_range = [-4,4]; x_range = [0,81]; y_range = [-1.2,3.2];
xtick_labels = 0:1:4; xtick_range = 0.5:20:80.5;
time_bins = 80;
stat_line_pos = 3.1;
rng(42,"twister"); % for reproducibility

plot_flag = 1;

PC_output_dim1_labels = {'Coeff','Score','Latent','TSquared','EV','Mu'};
PC_output_dim2_label = {'Tasks'};


num_perm = 1000;
num_units = 167;
chan_idx_Comb = [1:num_units;1:num_units];
num_tasks = length(task_labels);
num_prop = length(prop_set);
PC_output = cell(6,num_tasks);

pval_hold_set = cell(1,num_prop);
chan_idx_set = cell(1,num_prop);
set_cell_labels = {'SourceBlock','PCs','TargetBlock'};

for iTask = 1:num_tasks
%     temp_mat = block_mat_hold{iTask};
    temp_mat = data_set{iTask}(:,data_sel);
    temp_mat = zscore(temp_mat,0,2);
    block_mat_hold{iTask} = temp_mat;
    temp_mat2 = [temp_mat(:,1:40);temp_mat(:,41:80)];
    [coeff, score, latent, tsquared, explained, mu] = pca(temp_mat2');
    PC_output{1,iTask} = coeff; PC_output{2,iTask} = coeff;
    PC_output{3,iTask} = latent; PC_output{4,iTask} = tsquared;
    PC_output{5,iTask} = explained; PC_output{6,iTask} = mu;
end
    
for iP = 1:num_prop
    perm_hold = cell(num_tasks,PC_max,num_tasks);
    pval_hold = cell(num_tasks,PC_max,num_tasks);
    chan_idx_hold = cell(num_tasks,PC_max);
           
    curr_prop = prop_set(iP);
    prop_val = round(num_units*(curr_prop/100));
    
    if plot_flag == 1
        fig_dir = sprintf('%s\\figs\\PCA_10subj_AllTrials_Stacked_TopPropCut_%s_AL%d_CS%d\\Top_%d',...
            file_dir,stat_str,round(alpha_thresh*100),consec_thresh,curr_prop);
        png_dir = sprintf('%s\\pngs\\PCA_10subj_AllTrials_Stacked_TopPropCut_%s_AL%d_CS%d\\Top_%d',...
            file_dir,stat_str,round(alpha_thresh*100),consec_thresh,curr_prop);
        if ~exist(fig_dir,'dir')
            mkdir(fig_dir)
        end
        if ~exist(png_dir,'dir')
            mkdir(png_dir)
        end
    end
        
    for iTask = 1:num_tasks
        curr_task = task_labels{iTask}; 
        coeff = PC_output{1,iTask};
        
        for iPC = 1:PC_max
            [~,idx] = sort(coeff(1:num_units,iPC)+coeff(num_units+1:end,iPC),'descend');
            out = sort(idx(1:prop_val));

            chan_idx_hold{iTask,iPC} = out;
            
            % permutation stats
            num_hits = length(out);
            for iT = 1:num_tasks
                orig_vec = nanmean(block_mat_hold{iT}(out,:),1);
                perm_temp = nan(num_perm,time_bins);
                for iPerm = 1:num_perm
                    if stat_flag == 1
                        rand_sel = randi(num_units,[1,num_hits]);
                    elseif stat_flag == 2
                        temp_sel = randi(num_units-num_hits,[1,num_hits]);
                        temp_pool = setdiff(1:num_units,out);
                        rand_sel = temp_pool(temp_sel);
                    end
                    perm_vec = nanmean(block_mat_hold{iT}(rand_sel,:),1);
                    perm_temp(iPerm,:) = perm_vec;
                end
                perm_hold{iTask,iPC,iT} = perm_temp;
                
                pval_vec = sum(perm_temp > orig_vec, 1)./num_perm;
                pval_hold{iTask,iPC,iT} = pval_vec;
            end
            
            if plot_flag == 1
                h = figure; set(h,'Position',fig_pos);
                for iT = 1:num_tasks
                    subplot(2,num_tasks,iT);
                    imagesc(block_mat_hold{iT}(out,:));
                    caxis(c_range); xlim(x_range);
                    set(gca,'XTick',xtick_range); set(gca,'XTickLabel',xtick_labels);
                    if iT == 1
                        ylabel('PC Coeff-Sel Units');
                    end
                    
                    if iT == iTask
                        title(task_labels{iT},'Color','b');
                    else
                        title(task_labels{iT});
                    end
                    subplot(2,num_tasks,iT+num_tasks)
                    boundedline(1:time_bins,nanmean(block_mat_hold{iT}(out,:),1),nanstd(block_mat_hold{iT}(out,:),0,1)./sqrt(length(out)));
                    hline(0,'k--');
                    hold on; pvec = pval_hold{iTask,iPC,iT}; pvec_hits = find(pvec < alpha_thresh);
                    [class_types,class_tally,class_vec] = linear_consec_clust(pvec_hits);
                    for iClass = 1:max(class_types)
                        if class_tally(iClass) < consec_thresh
                            pvec_hits(class_vec == iClass) = 0;
                        end
                    end
                    pvec_hits(pvec_hits == 0) = [];
                    pvec_adj = nan(1,time_bins); pvec_adj(pvec_hits) = stat_line_pos;
                    plot(pvec_adj,'g','LineWidth',1.5);
                    xlim(x_range); ylim(y_range);
                    set(gca,'XTick',xtick_range); set(gca,'XTickLabel',xtick_labels);
                    if iT == 1
                        ylabel('Mean Norm FR');
                        xlabel('Time (s) from 1st stim on');
                    end
                end
%                 suptitle(sprintf('%s PC%d Common Sel, Top %d perc',curr_task,iPC,curr_prop));
                saveas(h,sprintf('%s\\gen_10subj_160drift_AT_Stack_%s_CommonSel_PC%d.fig',fig_dir,curr_task,iPC));
                saveas(h,sprintf('%s\\gen_10subj_160drift_AT_Stack_%s_CommonSel_PC%d.png',png_dir,curr_task,iPC));
                close(h);
            end
            
        end
    end
    
    pval_hold_set{iP} = pval_hold;
    chan_idx_set{iP} = chan_idx_hold;
end

% save(sprintf('%s\\PCA_finalize\\PCA_10subj_160drift_AllTrials_Stacked_TopPCgen_AL%d_%s.mat',file_dir,round(alpha_thresh*100),stat_str),...
%     'task_labels','PC_output_dim1_labels','PC_output_dim2_label',...
%     'PC_max','num_perm','*_set','PC_output','set_cell_labels')
