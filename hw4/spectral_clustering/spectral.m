function idx = spectral(W, k)
%SPECTRUAL spectral clustering
%   Input:
%     W: Adjacency matrix, N-by-N matrix
%     k: number of clusters
%   Output:
%     idx: data point cluster labels, n-by-1 vector.

% YOUR CODE HERE
D = diag(sum(W));
L = D-W;
opt = struct('issym', true, 'isreal', true);
[V,~] = eigs(L, D, k,'SM', opt);
idx = kmeans(V, k);



end
