load kmeans_data;
[idx, ctrs, iter_ctrs] = kmeans(X,2);
kmeans_plot(X, idx, ctrs, iter_ctrs);
