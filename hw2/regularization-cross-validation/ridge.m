function w = ridge(X, y, lambda)
%RIDGE Ridge Regression.
%
%   INPUT:  X: training sample features, P-by-N matrix.
%           y: training sample labels, 1-by-N row vector.
%           lambda: regularization parameter.
%
%   OUTPUT: w: learned parameters, (P+1)-by-1 column vector.
%

% YOUR CODE HERE
[P,N] = size(X);
I = eye(P);
w = pinv(X*X'+lambda*I)*X*y';
w = [1;w];

end
