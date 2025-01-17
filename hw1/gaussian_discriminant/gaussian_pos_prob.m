function p = gaussian_pos_prob(X, Mu, Sigma, Phi)
%GAUSSIAN_POS_PROB Posterior probability of GDA.
%   p = GAUSSIAN_POS_PROB(X, Mu, Sigma) compute the posterior probability
%   of given N data points X using Gaussian Discriminant Analysis where the
%   K gaussian distributions are specified by Mu, Sigma and Phi.
%
%   Inputs:
%       'X'     - M-by-N matrix, N data points of dimension M.
%       'Mu'    - M-by-K matrix, mean of K Gaussian distributions.
%       'Sigma' - M-by-M-by-K matrix (yes, a 3D matrix), variance matrix of
%                   K Gaussian distributions.
%       'Phi'   - 1-by-K matrix, prior of K Gaussian distributions.
%
%   Outputs:
%       'p'     - N-by-K matrix, posterior probability of N data points
%                   with in K Gaussian distributions.

N = size(X, 2);
K = length(Phi);
p = zeros(N, K);

for data_index = 1:N
    total_P = 0;
    for gaussian_index = 1:K
        total_P = total_P + Phi(gaussian_index) * 1/(2*pi*sqrt(det(Sigma(:,:,gaussian_index))))*exp(-1/2*(X(:,data_index)-Mu(:,gaussian_index))'/Sigma(:,:,gaussian_index)*(X(:,data_index)-Mu(:,gaussian_index)));
    end
    
    for gaussian_index = 1:K
        p(data_index,gaussian_index) = Phi(gaussian_index) * 1/(2*pi*sqrt(det(Sigma(:,:,gaussian_index))))*exp(-1/2*(X(:,data_index)-Mu(:,gaussian_index))'/Sigma(:,:,gaussian_index)*(X(:,data_index)-Mu(:,gaussian_index)))/total_P;
    end
end


% Your code HERE
