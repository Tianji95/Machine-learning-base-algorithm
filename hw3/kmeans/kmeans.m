function [idx, ctrs, iter_ctrs] = kmeans(X, K)
%KMEANS K-Means clustering algorithm
%
%   Input: X - data point features, n-by-p maxtirx.
%          K - the number of clusters
%
%   OUTPUT: idx  - cluster label
%           ctrs - cluster centers, K-by-p matrix.
%           iter_ctrs - cluster centers of each iteration, K-by-p-by-iter
%                       3D matrix.

% YOUR CODE HERE

max_iter = 1000;
[N, P] = size(X);
disp(N)
ctrs_rand = X(randperm(N),:);
ctrs = ctrs_rand(1:K,:);
idx = zeros(N,1);

iter_ctrs = zeros(K, P, max_iter);

for iter_idx = 1:max_iter
    new_ctrs = zeros(K,P);
    dist = pdist2(X, ctrs, 'euclidean');
    [~,pos] = min(dist,[],2);
    idx = pos;
    for k_idx = 1:K
        new_ctrs(k_idx,:) = sum(X(pos==k_idx,:))/sum(pos==k_idx);
    end
%     
%     
%     for data_idx = 1:N
%         dist = pdist2(X(data_idx,:), ctrs, 'euclidean');
%         [~,pos] = min(dist);
%         idx(data_idx,1) = pos;
%         total_num(pos,:) = total_num(pos,:) + X(data_idx,:);
%         total_count(pos,1) = total_count(pos,1) + 1;
%     end
%     new_ctrs = total_num ./repmat(total_count, 1, P);
    if all(ctrs == new_ctrs)
        iter_ctrs(:,:,iter_idx) = ctrs;
        iter_ctrs = iter_ctrs(:,:,1:iter_idx);
        break
    else
        iter_ctrs(:,:,iter_idx) = ctrs;
        ctrs = new_ctrs;
    end
%     disp(ctrs);
end
disp(size(iter_ctrs))




end
