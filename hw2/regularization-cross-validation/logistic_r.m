function w = logistic_r(X, y, lambda)
%LR Logistic Regression.
%
%   INPUT:  X:   training sample features, P-by-N matrix.
%           y:   training sample labels, 1-by-N row vector.
%           lambda: regularization parameter.
%
%   OUTPUT: w    : learned parameters, (P+1)-by-1 column vector.
%

% YOUR CODE HERE

learningRate = 0.001;
[P,N] = size(X);
w = ones(P+1,1);
X = [ones(1,N);X];
maxIter = 3000;

for i=1:N
    if (y(i)==-1)
        y(i)=0;
    end
end

for iter=1:maxIter
     before = w;
     output = sigmf((w'*X),[1 0])-y;
     w = w*(1-learningRate * lambda) - learningRate * X * output';
%      if abs(w'*w-before'*before) < eposi
%          break
%      end
end

end
