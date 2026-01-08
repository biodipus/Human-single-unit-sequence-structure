idx = chan_idx_hold{2,1};
data_set{1} = fr_lir; data_set{2} = fr_lr4; 
data_set{1} = fr_mir; data_set{2} = fr_mr4; data_set{3} = fr_mr2;
temp = data_set{3}(idx,data_sel);
temp = zscore(temp,0,2);
figure();plot(nanmean(temp,1))