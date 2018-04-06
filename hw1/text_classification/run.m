%ham_train contains the occurrences of each word in ham emails. 1-by-N vector
ham_train = csvread('ham_train.csv');
%spam_train contains the occurrences of each word in spam emails. 1-by-N vector
spam_train = csvread('spam_train.csv');
%N is the size of vocabulary.
N = size(ham_train, 2);
%There 9034 ham emails and 3372 spam emails in the training samples
num_ham_train = 9034;
num_spam_train = 3372;
%Do smoothing
x = [ham_train;spam_train] + 1;

%ham_test contains the occurences of each word in each ham test email. P-by-N vector, with P is number of ham test emails.
load ham_test.txt;
ham_test_tight = spconvert(ham_test);
ham_test = sparse(size(ham_test_tight, 1), size(ham_train, 2));
ham_test(:, 1:size(ham_test_tight, 2)) = ham_test_tight;
%spam_test contains the occurences of each word in each spam test email. Q-by-N vector, with Q is number of spam test emails.
load spam_test.txt;
spam_test_tight = spconvert(spam_test);
spam_test = sparse(size(spam_test_tight, 1), size(spam_train, 2));
spam_test(:, 1:size(spam_test_tight, 2)) = spam_test_tight;


%get top 10 ratio words
ratio_matrix = zeros(1,N);
for col_index = 1:N
    ratio_matrix(1,col_index) = x(2,col_index)/num_spam_train /(x(1,col_index)/num_ham_train);
end
[sort_ratio_matrix,sort_ratio_index] = sort(ratio_matrix,'descend');
disp(sort_ratio_matrix(1,1:10));
disp(sort_ratio_index(1,1:10));

%homework
lglikelihood_data = likelihood(x);%2-by-N matrix
train_pspam = num_spam_train/(num_spam_train+num_ham_train);

test_error = 0;
ham_num = 0;
spam_num = 0;

[P,] = size(ham_test,1);
[Q,] = size(spam_test,1);
for ham_index=1:P
    test_pham = ham_test(ham_index,:)*lglikelihood_data(1,:)';
    test_pspam = ham_test(ham_index,:)*lglikelihood_data(2,:)';
    if test_pham+log(1-train_pspam)>test_pspam+log(train_pspam)
        ham_num = ham_num + 1;
    else
        spam_num = spam_num + 1;
        test_error = test_error+1;
    end
end

disp(ham_num)
disp(spam_num)
ham_num = 0;
spam_num = 0;
for spam_index=1:Q
    test_pham = spam_test(spam_index,:)*lglikelihood_data(1,:)';
    test_pspam = spam_test(spam_index,:)*lglikelihood_data(2,:)';
    if test_pham+log(1-train_pspam)<test_pspam+log(train_pspam)
        spam_num = spam_num + 1;
    else
        ham_num = ham_num + 1;
        test_error = test_error+1;
    end
end
disp(ham_num)
disp(spam_num)

accuracy = 1-test_error/(P+Q);
disp(accuracy)







%TODO
%Implement a ham/spam email classifier, and calculate the accuracy of your classifier
