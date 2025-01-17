function p = posterior(x)
%POSTERIOR Two Class Posterior Using Bayes Formula
%
%   INPUT:  x, features of different class, C-By-N vector
%           C is the number of classes, N is the number of different feature
%
%   OUTPUT: p,  posterior of each class given by each feature, C-By-N matrix
%

[C, N] = size(x);
l = likelihood(x);
total = sum(sum(x));
p = zeros(C, N);

for line_index=1:C
    prior = sum(x(line_index,:))/total;
    p(line_index,:)=l(line_index,:)*prior;
end
end
