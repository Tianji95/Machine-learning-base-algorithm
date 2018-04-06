load('TDT2_data', 'fea', 'gnd');

MaxIter = 20;
k = 5;
total_accuracy_sp = 0;
total_mIhat_sp = 0;
for iter = 1:MaxIter
    options = [];
    options.NeighborMode = 'KNN';
    options.k = 5;
    options.WeightMode = 'HeatKernel';
    W = constructW(fea, options);
    res_sp = spectral(W, k);
    res_sp = bestMap(gnd, res_sp);
    total_accuracy_sp = total_accuracy_sp + length(find(gnd == res_sp))/length(gnd);
    total_mIhat_sp = total_mIhat_sp + MutualInfo(gnd, res_sp); 
end
accuracy_sp = total_accuracy_sp / MaxIter;
mIhat_sp = total_mIhat_sp / MaxIter;

fprintf('spectral clustering: accuracy:%f, MutualInfo: %f\n', accuracy_sp, mIhat_sp);

[label, center] = litekmeans(fea, k, 'MaxIter', MaxIter);
res_km = bestMap(gnd, label);
accuracy_km = length(find(gnd == res_km))/length(gnd);
mIhat_km = MutualInfo(gnd, res_km);
fprintf('kmeans: accuracy:%f, MutualInfo: %f\n', accuracy_km, mIhat_km);



% YOUR CODE HERE