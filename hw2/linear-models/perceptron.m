function [w, iter] = perceptron(X, y)
%PERCEPTRON Perceptron Learning Algorithm.
%
%   INPUT:  X: training sample features, P-by-N matrix.
%           y: training sample labels, 1-by-N row vector.
%
%   OUTPUT: w:    learned perceptron parameters, (P+1)-by-1 column vector.
%           iter: number of iterations
%

% YOUR CODE HERE
iter = 0;
[P,N] = size(X);
X=[ones(1,N);X];%%bias
maxIter = 10000;
w = zeros(P+1,1);
for iter_index = 1:maxIter
    iter = iter + 1;
    isSolution = 1;
    for col_index = 1:N
        if sign(w'*X(:,col_index))~=y(col_index)
            w = w + y(col_index)*X(:,col_index);
            isSolution = 0;
        end
    end
    if isSolution==1
        break
    end
    
end

end
