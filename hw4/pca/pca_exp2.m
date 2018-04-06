load('ORL_data', 'fea_Train', 'gnd_Train', 'fea_Test', 'gnd_Test');

% YOUR CODE HERE
% 1. Feature preprocessing
% 2. Run PCA
[eigvec, eigval] = pca(fea_Train);
% 3. Visualize eigenface
show_face(eigvec')
% 4. Project data on to low dimensional space
% 5. Run KNN in low dimensional space
options = [];
dimension = [8,16,32,64,128];
K_in_knn = 1;

for di_idx = 1:length(dimension)
    fea_Train_after = fea_Train * eigvec(:,1:dimension(di_idx));
    fea_Test_after = fea_Test * eigvec(:,1:dimension(di_idx));
    y_predict = knn(fea_Test_after', fea_Train_after', gnd_Train', K_in_knn)';
    errorRate = sum(y_predict~=gnd_Test)/N;
    fprintf('K_in_knn is %d, dimension is %d, errorRate is %f\n',K_in_knn, dimension(di_idx), errorRate);
end
    
% [eigvec, eigval] = LDA(gnd_Train,options,fea_Train);
% fea_Train_after = fea_Train * eigvec;
% fea_Test_after = fea_Test * eigvec;
% y_predict = knn(fea_Test_after', fea_Train_after', gnd_Train', K_in_knn)';
% errorRate = sum(y_predict~=gnd_Test)/N;
% fprintf('Using LDA, K_in_knn is %d, errorRate is %f\n',K_in_knn, errorRate);

% 6. Recover face images form low dimensional space, visualize them
figure
show_face(fea_Train)
for di_idx = 1:length(dimension)
    fea_Train_after = fea_Train * eigvec(:,1:dimension(di_idx));
    recover = fea_Train_after * eigvec(:,1:dimension(di_idx))';
    figure
    show_face(recover)
end
