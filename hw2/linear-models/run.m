% You can use this skeleton or write your own.
% You are __STRONGLY__ suggest to run this script section-by-section using Ctrl+Enter.
% See http://www.mathworks.cn/cn/help/matlab/matlab_prog/run-sections-of-programs.html for more details.

%% Part1: Preceptron
nRep = 1000; % number of replicates
nTrain = 10; % number of training data
sumIter = 0;
E_train = 0;
E_test = 0;
for i = 1:nRep
    [X, y, w_f] = mkdata(nTrain*2);
    X_train = X(:,1:nTrain);
    y_train = y(1:nTrain);
    X_test = X(:,nTrain+1:nTrain*2);
    y_test = y(nTrain+1:nTrain*2);
   
    [w_g, iter] = perceptron(X_train, y_train);
    
    [P,N] = size(X_train);
    X_train = [ones(1,N);X_train];
    for col_index = 1:N
        thisError = sign(w_g'*X_train(:,col_index))*y(col_index);
        if thisError < 0
            E_train = E_train + 1;
        end
    end
    
    [P,N] = size(X_test);
    X_test= [ones(1,N);X_test];
    for col_index = 1:N
        thisError = sign(w_g'*X_test(:,col_index))*y_test(col_index);
        if thisError < 0
            E_test = E_test + 1;
        end
    end
    
    sumIter = sumIter+iter;
    % Compute training, testing error
    % Sum up number of iterations
end

avgIter = sumIter/nRep;
E_test = E_test/(nRep*nTrain);
E_train = E_train/(nRep*nTrain);

fprintf('E_train is %f, E_test is %f.\n', E_train, E_test);
fprintf('Average number of iterations is %d.\n', avgIter);
plotdata(X, y, w_f, w_g, 'Pecertron');

%% Part2: Preceptron: Non-linearly separable case
nTrain = 100; % number of training data
[X, y, w_f] = mkdata(nTrain, 'noisy');
[w_g, iter] = perceptron(X, y);

plotdata(X, y, w_f, w_g, 'Pecertron');
%% Part3: Linear Regression
nRep = 1000; % number of replicates
nTrain = 100; % number of training data
E_train = 0;
E_test = 0;

for i = 1:nRep
    E_train_temp = 0;
    E_test_temp = 0;
    [X, y, w_f] = mkdata(nTrain*2);
    X_train = X(:,1:nTrain);
    y_train = y(1:nTrain);
    X_test = X(:,nTrain+1:nTrain*2);
    y_test = y(nTrain+1:nTrain*2);
    
    w_g = linear_regression(X_train, y_train);
    
    [P,N] = size(X_train);
    X_train = [ones(1,N);X_train];
    for col_index = 1:N
        thisPredict = sign(w_g'*X_train(:,col_index));
        if y_train(col_index)~=thisPredict
            E_train_temp = E_train_temp + 1;
        end
    end
    
    
    [P,N] = size(X_test);
    X_test= [ones(1,N);X_test];
    for col_index = 1:N
        thisPredict = sign(w_g'*X_test(:,col_index));
        if thisPredict ~= y_test(col_index)
            E_test_temp = E_test_temp + 1;
        end
    end
    % Compute training, testing error
    E_test = E_test + E_test_temp/N;
    E_train = E_train + E_train_temp/N;
end

E_test = E_test/(nRep);
E_train = E_train/(nRep);
fprintf('E_train is %f, E_test is %f.\n', E_train, E_test);
plotdata(X, y, w_f, w_g, 'Linear Regression');

%% Part4: Linear Regression: noisy
nRep = 1000; % number of replicates
nTrain = 100; % number of training data
E_train = 0;
E_test = 0;
for i = 1:nRep
    E_train_temp = 0;
    E_test_temp = 0;
    [X, y, w_f] = mkdata(nTrain*2, 'noisy');
    X_train = X(:,1:nTrain);
    y_train = y(1:nTrain);
    X_test = X(:,nTrain+1:nTrain*2);
    y_test = y(nTrain+1:nTrain*2);
    
    w_g = linear_regression(X_train, y_train);
    
    [P,N] = size(X_train);
    X_train = [ones(1,N);X_train];
    for col_index = 1:N
        thisPredict = sign(w_g'*X_train(:,col_index));
        if y_train(col_index)~=thisPredict
            E_train_temp = E_train_temp + 1;
        end
    end
    
    
    [P,N] = size(X_test);
    X_test= [ones(1,N);X_test];
    for col_index = 1:N
        thisPredict = sign(w_g'*X_test(:,col_index));
        if thisPredict ~= y_test(col_index)
            E_test_temp = E_test_temp + 1;
        end
    end
    E_test = E_test + E_test_temp/N;
    E_train = E_train + E_train_temp/N;

    % Compute training, testing error
end

E_test = E_test/(nRep);
E_train = E_train/(nRep);

fprintf('E_train is %f, E_test is %f.\n', E_train, E_test);
plotdata(X, y, w_f, w_g, 'Linear Regression: noisy');

%% Part5: Linear Regression: poly_fit
load('poly_train', 'X', 'y');
load('poly_test', 'X_test', 'y_test');
w = linear_regression(X, y);

E_train = 0;
E_test = 0;

[P,N_train] = size(X);
X_train = [ones(1,N_train);X];
for col_index = 1:N_train
    thisPredict = sign(w'*X_train(:,col_index));
    if y(col_index)~=thisPredict
        E_train = E_train + 1;
    end
end


[P,N_test] = size(X_test);
X_test_temp= [ones(1,N_test);X_test];
for col_index = 1:N_test
    thisPredict = sign(w'*X_test_temp(:,col_index));
    if thisPredict ~= y_test(col_index)
        E_test = E_test + 1;
    end
end
E_test = E_test/N_test;
E_train = E_train/N_train;

% Compute training, testing error
fprintf('E_train is %f, E_test is %f.\n', E_train, E_test);

% poly_fit with transform
X_t = [X(1,:);X(2,:);X(1,:).*X(2,:);X(1,:).*X(1,:);X(2,:).*X(2,:)]; % CHANGE THIS LINE TO DO TRANSFORMATION
X_test_t = [X_test(1,:);X_test(2,:);X_test(1,:).*X_test(2,:);X_test(1,:).*X_test(1,:);X_test(2,:).*X_test(2,:)]; % CHANGE THIS LINE TO DO TRANSFORMATION
w = linear_regression(X_t, y);


E_train = 0;
E_test = 0;

[P,N_train] = size(X_t);
X_t = [ones(1,N_train);X_t];
for col_index = 1:N_train
    thisPredict = sign(w'*X_t(:,col_index));
    if y(col_index)~=thisPredict
        E_train = E_train + 1;
    end
end


[P,N_test] = size(X_test_t);
X_test_t= [ones(1,N_test);X_test_t];

for col_index = 1:N_test
    thisPredict = sign(w'*X_test_t(:,col_index));
    if thisPredict ~= y_test(col_index)
        E_test = E_test + 1;
    end
end
E_test = E_test/N_test;
E_train = E_train/N_train;
% Compute training, testing error
fprintf('E_train is %f, E_test is %f.\n', E_train, E_test);


%% Part6: Logistic Regression
nRep = 100; % number of replicates
nTrain = 100; % number of training data

E_train = 0;
E_test = 0;

for i = 1:nRep
    [X, y, w_f] = mkdata(nTrain*2);
    X_train = X(:,1:nTrain);
    y_train = y(1:nTrain);
    X_test = X(:,nTrain+1:nTrain*2);
    y_test = y(nTrain+1:nTrain*2);
    E_train_temp = 0;
    E_test_temp = 0;
    
    w_g = logistic(X_train, y_train);
    [P,N] = size(X_train);
    X_train = [ones(1,N);X_train];
    for col_index = 1:N
        thisPredict = sign(w_g'*X_train(:,col_index));
        if y_train(col_index)~=thisPredict
            E_train_temp = E_train_temp + 1;
        end
    end
    
    [P,N] = size(X_test);
    X_test = [ones(1,N);X_test];
    for col_index=1:N
        thisPredict = sign(w_g'*X_test(:,col_index));
        if y_test(col_index)~=thisPredict
            E_test_temp = E_test_temp + 1;
        end
    end
    E_test = E_test + E_test_temp/N;
    E_train = E_train + E_train_temp/N;
    % Compute training, testing error
end
E_test = E_test/nRep;
E_train = E_train/nRep;

fprintf('E_train is %f, E_test is %f.\n', E_train, E_test);
plotdata(X, y, w_f, w_g, 'Logistic Regression');

%% Part7: Logistic Regression: noisy
nRep = 100; % number of replicates
nTrain = 100; % number of training data
nTest = 10000; % number of training data

E_train = 0;
E_test = 0;

for i = 1:nRep
    [X, y, w_f] = mkdata(nTrain+nTest, 'noisy');
    X_train = X(:,1:nTrain);
    y_train = y(1:nTrain);
    X_test = X(:,nTrain+1:nTrain+nTest);
    y_test = y(nTrain+1:nTrain+nTest);
    E_train_temp = 0;
    E_test_temp = 0;
    
    w_g = logistic(X_train, y_train);
    [P,N_train] = size(X_train);
    X_train = [ones(1,N_train);X_train];
    for col_index = 1:N_train
        thisPredict = sign(w_g'*X_train(:,col_index));
        if y_train(col_index)~=thisPredict
            E_train_temp = E_train_temp + 1;
        end
    end
    
    [P,N_test] = size(X_test);
    X_test = [ones(1,N_test);X_test];
    for col_index=1:N_test
        thisPredict = sign(w_g'*X_test(:,col_index));
        if y_test(col_index)~=thisPredict
            E_test_temp = E_test_temp + 1;
        end
    end
    E_test = E_test + E_test_temp/N_test;
    E_train = E_train + E_train_temp/N_train;
    % Compute training, testing error
end
E_test = E_test/nRep;
E_train = E_train/nRep;


fprintf('E_train is %f, E_test is %f.\n', E_train, E_test);
plotdata(X(:,1:nTrain), y(1:nTrain), w_f, w_g, 'Logistic Regression: noisy');

%% Part8: SVM
nRep = 1000; % number of replicates
nTrain = 100; % number of training data

E_train = 0;
E_test = 0;
total_num = 0;

for i = 1:nRep
    [X, y, w_f] = mkdata(nTrain*2);
    
    X_train = X(:,1:nTrain);
    y_train = y(1:nTrain);
    X_test = X(:,nTrain+1:nTrain+nTrain);
    y_test = y(nTrain+1:nTrain+nTrain);
    E_train_temp = 0;
    E_test_temp = 0;
    
    [w_g, num_sc] = svm(X_train, y_train);
    [P,N_train] = size(X_train);
    X_train = [ones(1,N_train);X_train];
    for col_index = 1:N_train
        thisPredict = sign(w_g'*X_train(:,col_index));
        if y_train(col_index)~=thisPredict
            E_train_temp = E_train_temp + 1;
        end
    end
    
    [P,N_test] = size(X_test);
    X_test = [ones(1,N_test);X_test];
    for col_index=1:N_test
        thisPredict = sign(w_g'*X_test(:,col_index));
        if y_test(col_index)~=thisPredict
            E_test_temp = E_test_temp + 1;
        end
    end
    E_test = E_test + E_test_temp/N_test;
    E_train = E_train + E_train_temp/N_train;
    
    total_num = total_num + num_sc;
    % Compute training, testing error
    % Sum up number of support vectors
end

E_test = E_test/nRep;
E_train = E_train/nRep;
total_num = total_num / nRep;
fprintf('E_train is %f, E_test is %f., number of support vectors is %f\n', E_train, E_test, total_num);
plotdata(X, y, w_f, w_g, 'SVM');
