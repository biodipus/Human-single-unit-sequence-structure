function [class_types,class_tally,class_vec] = linear_consec_clust(vec)
    if isempty(vec)
        class_types = [];
        class_tally = 0;
        class_vec = [];
        return
    end
    if size(vec,1) > 1
        vec = vec';
    end
    t = diff(vec) == 1;
    y = [t,false];
    x = xor(y,[false,t]);
    ii = cumsum(~(x|y) + y.*x);
    class_types = unique(ii);
    num_class = length(class_types);
    class_tally = zeros(1,num_class);
    for iC = 1:num_class
        class_tally(iC) = sum(ii==iC);
    end
    class_vec = ii;
end