function W = knn_graph(X, k, threshold)
%KNN_GRAPH Construct W using KNN graph
%   Input: X - data point features, n-by-p maxtirx.
%          k - number of nn.
%          threshold - distance threshold.
%
%   Output:W - adjacency matrix, n-by-n matrix.

% YOUR CODE HERE

[N, ~] = size(X);
W = pdist2(X, X, 'euclidean');
%W = exp(-abs(W/2/std2(W)));
[~, sort_pos] = sort(W);
W(sort_pos(k+1:N)) = 0;
W( W > threshold) = 0;
W(W==0) = 100;

end
