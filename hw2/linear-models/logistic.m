function w = logistic(X, y)
%LR Logistic Regression.
%
%   INPUT:  X:   training sample features, P-by-N matrix.
%           y:   training sample labels, 1-by-N row vector.
%
%   OUTPUT: w    : learned parameters, (P+1)-by-1 column vector.
%

% YOUR CODE HERE




maxIter = 500;
learningRate = 0.001;
[P,N] = size(X);
w = ones(P+1,1);
X = [ones(1,N);X];


for i=1:N
    if (y(i)==-1)
        y(i)=0;
    end
end

for iter=1:maxIter
     output = sigmf((w'*X),[1 0])-y;
     w = w - learningRate * X * output';
end

end
