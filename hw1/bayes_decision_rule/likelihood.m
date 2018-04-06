function l = likelihood(x)
%LIKELIHOOD Different Class Feature Liklihood 
%
%   INPUT:  x, features of different class, C-By-N vector
%           C is the number of classes, N is the number of different feature
%
%   OUTPUT: l,  likelihood of each feature(from smallest feature to biggest feature) given by each class, C-By-N matrix
%

[C, N] = size(x);
l = zeros(C, N);
for line_index = 1:C
    total_num = sum(x(line_index,:));
    l(line_index,:) = x(line_index,:)/total_num;
end
%TODO
end
