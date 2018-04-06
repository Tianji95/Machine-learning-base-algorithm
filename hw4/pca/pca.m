function [eigvector, eigvalue] = PCA(data)
%PCA	Principal Component Analysis
%
%             Input:
%               data       - Data matrix. Each row vector of fea is a data point.
%
%             Output:
%               eigvector - Each column is an embedding function, for a new
%                           data point (row vector) x,  y = x*eigvector
%                           will be the embedding result of x.
%               eigvalue  - The sorted eigvalue of PCA eigen-problem.
%

% YOUR CODE HERE
X_mean = mean(data);

disp(X_mean);
[N, P] = size(data);
S = zeros(P, P);
for row_idx=1:N
    miner = data(row_idx,:) - X_mean;
    S = S + miner' * miner;
end
S = S / N;
[eigVec, eigVal] = eig(S);

[eigvalue, eigPos] = sort(sum(eigVal),'descend');
eigvector = eigVec(:,eigPos);
% disp(eigvector);
% disp(eigvalue);
end