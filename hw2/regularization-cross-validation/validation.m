%% Ridge Regression
load('digit_train', 'X', 'y');

%minimum = min(X);
%ranges = max(X)-minimum;
mean_square = std(X);
[P,N] = size(X);
mean = sum(X)/P;
X = (X - repmat(mean, P, 1))./repmat(mean_square, P, 1);

% Do feature normalization
% ...

% Do LOOCV

lambdas = [1e-3, 1e-2, 1e-1, 0, 1, 1e1, 1e2, 1e3];
%lambdas = [0];
lambda = 0;

min_E_val = N;
min_w = zeros(P+1, 1);
for i = 1:length(lambdas)
    E_val = 0;
    for j = 1:size(X, 2)
        if j==1
            X_ = X(:,j+1:size(X,2)); y_ = y(j+1:size(X,2));
        elseif j == size(X, 2)
            X_ = X(:,1:j-1); y_ = y(1:j-1);
        else
            X_ = [X(:,1:j-1) X(:,j+1:size(X,2))]; y_ = [y(1:j-1) y(j+1:size(X,2))]; % take point j out of X
        end
        w = ridge(X_, y_, lambdas(i));
        if sign(w'*[1;X(:,j)])~=y(j)
            E_val = E_val + 1;
        end
    end
    if min_E_val > E_val
        lambda = lambdas(i);
        min_E_val = E_val;
        min_w = w;
    end
    
    % Update lambda according validation error
end

E_train = 0;
E_test = 0;

X_train = [ones(1,N);X];
for col_index = 1:N
    if sign(min_w'*X_train(:,col_index)) ~= y(col_index)
        E_train = E_train + 1;
    end
end
E_train = E_train/N;


% Compute training error

load('digit_test', 'X_test', 'y_test');

%minimum = min(X_test);
mean_square = std(X_test);
%ranges = max(X_test)-minimum;
[P,N] = size(X_test);
mean = sum(X_test)/P;
X_test = (X_test - repmat(mean, P, 1))./repmat(mean_square, P, 1);
X_test = [ones(1,N);X_test];

for col_index = 1:N
    if sign(min_w'*X_test(:,col_index)) ~= y_test(col_index)
        E_test = E_test + 1;
    end
end
E_test = E_test / N;

w_square = min_w' * min_w;


fprintf('E_train is %f, E_test is %f.\n', E_train, E_test);
fprintf('the Best lambda is %f\n', lambda);
fprintf('the sum of w squares is %f\n', w_square);

% Do feature normalization
% ...
% Compute test error
%% Logistic

load('digit_train', 'X', 'y');

minimum = min(X);
ranges = max(X)-minimum;
[P,N] = size(X);
mean = sum(X)/P;
X = (X - repmat(mean, P, 1))./repmat(ranges, P, 1);

% Do feature normalization
% ...

% Do LOOCV

lambdas = [1e-3, 1e-2, 1e-1, 0, 1, 1e1, 1e2, 1e3];
%lambdas = [0];
lambda = 0;

min_E_val = N;
min_w = zeros(P+1, 1);
for i = 1:length(lambdas)
    E_val = 0;
    for j = 1:size(X, 2)
        if j==1
            X_ = X(:,j+1:size(X,2)); y_ = y(j+1:size(X,2));
        elseif j == size(X, 2)
            X_ = X(:,1:j-1); y_ = y(1:j-1);
        else
            X_ = [X(:,1:j-1) X(:,j+1:size(X,2))]; y_ = [y(1:j-1) y(j+1:size(X,2))]; % take point j out of X
        end
        w = logistic_r(X_, y_, lambdas(i));
        if sign(w'*[1;X(:,j)])~=y(j)
            E_val = E_val + 1;
        end
    end
    if min_E_val > E_val
        lambda = lambdas(i);
        min_E_val = E_val;
        min_w = w;
    end
    
    % Update lambda according validation error
end


E_train = 0;
E_test = 0;

X_train = [ones(1,N);X];
for col_index = 1:N
    if sign(min_w'*X_train(:,col_index)) ~= y(col_index)
        E_train = E_train + 1;
    end
end
E_train = E_train/N;


% Compute training error

load('digit_test', 'X_test', 'y_test');

minimum = min(X_test);
ranges = max(X_test)-minimum;
[P,N] = size(X_test);
mean = sum(X_test)/P;
X_test = (X_test - repmat(mean, P, 1))./repmat(ranges, P, 1);
X_test = [ones(1,N);X_test];
for col_index = 1:N
    if sign(min_w'*X_test(:,col_index)) ~= y_test(col_index)
        E_test = E_test + 1;
    end
end
E_test = E_test / N;

w_square = min_w' * min_w;

fprintf('logistic r-cv \n');
fprintf('E_train is %f, E_test is %f.\n', E_train, E_test);
fprintf('the Best lambda is %f\n', lambda);
fprintf('the sum of w squares is %f\n', w_square);

