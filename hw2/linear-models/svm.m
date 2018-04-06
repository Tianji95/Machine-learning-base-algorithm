function [w, num] = svm(X, y)
%SVM Support vector machine.
%
%   INPUT:  X: training sample features, P-by-N matrix.
%           y: training sample labels, 1-by-N row vector.
%
%   OUTPUT: w:    learned perceptron parameters, (P+1)-by-1 column vector.
%           num:  number of support vectors
%

% YOUR CODE HERE

[P,N] = size(X);
H=eye(P+1);
H(P+1,P+1)=0;
f=zeros(P+1,1);
X=[ones(1,N);X];
A = -[y.*X(1,:);y.*X(2,:);y.*X(3,:)]';
b = -ones(N,1);
w=quadprog(H,f,A,b);
num = 0;
for i=1:N
    if (w'*X(:,i)<=1)
        num = num+1;
    end
end

end
