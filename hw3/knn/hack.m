function digits = hack(img_name)
%HACK Recognize a CAPTCHA image
%   Inputs:
%       img_name: filename of image
%   Outputs:
%       digits: 1x5 matrix, 5 digits in the input CAPTCHA image.

load('hack_data');
K = 1;
X_test = extract_image(img_name);
digits = knn(X_test, X, y, K);
% YOUR CODE HERE




end