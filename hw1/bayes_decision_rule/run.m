% You can use this skeleton or write your own.
% You are __STRONGLY__ suggest to run this script section-by-section using Ctrl+Enter.
% See http://www.mathworks.cn/cn/help/matlab/matlab_prog/run-sections-of-programs.html for more details.

%%load data
load('data');
all_x = cat(2, x1_train, x1_test, x2_train, x2_test);
range = [min(all_x), max(all_x)];
train_x = get_x_distribution(x1_train, x2_train, range);
test_x = get_x_distribution(x1_test, x2_test, range);

%% Part1 likelihood: 
l = likelihood(train_x);

bar(range(1):range(2), l');
xlabel('x');
ylabel('P(x|\omega)');
axis([range(1) - 1, range(2) + 1, 0, 0.5]);

% l_test = likelihood(test_x);
% bar(range(1):range(2), l_test');
% xlabel('x');
% ylabel('P(x|\omega)');
% axis([range(1) - 1, range(2) + 1, 0, 0.5]);

[C, N] = size(test_x);
test_error_num = 0;
for col_index = 1:N
    row=find(l(:,col_index)==max(l(:,col_index)));
    feature_error_num = sum(test_x(:,col_index)) - test_x(row,col_index);
    test_error_num = test_error_num + feature_error_num;
end
disp(test_error_num);

%TODO
%compute the number of all the misclassified x using maximum likelihood decision rule

%% Part2 posterior:
p = posterior(train_x);

bar(range(1):range(2), p');
xlabel('x');
ylabel('P(\omega|x)');
axis([range(1) - 1, range(2) + 1, 0, 1.2]);

[C, N] = size(test_x);
test_error_num = 0;
for col_index = 1:N
    row=find(p(:,col_index)==max(p(:,col_index)));
    feature_error_num = sum(test_x(:,col_index)) - test_x(row,col_index);
    test_error_num = test_error_num + feature_error_num;
end
disp(test_error_num);

%TODO
%compute the number of all the misclassified x using optimal bayes decision rule

%% Part3 risk:
risk = [0, 1; 2, 0];
riskVal = 0;

for col_index = 1:N
    if 2 * p(1,col_index)>  p(2,col_index);
        riskVal = riskVal + p(2,col_index)*risk(1,2)*sum(test_x(:,col_index));
    else
        riskVal = riskVal + p(1,col_index)*risk(2,1)*sum(test_x(:,col_index));
    end
end
disp(riskVal)

%TODO
%get the minimal risk using optimal bayes decision rule and risk weights
