load digit_data;

[idx, ctrs, iter_ctrs] = kmeans(X,20);
show_digit(ctrs);