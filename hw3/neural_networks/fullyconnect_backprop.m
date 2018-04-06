function [weight_grad, bias_grad, out_sensitivity] = fullyconnect_backprop(in_sensitivity,  in, weight)
%The backpropagation process of fullyconnect
%   input parameter:
%       in_sensitivity  : the sensitivity from the upper layer, shape: 
%                       : [number of images, number of outputs in feedforward]
%       in              : the input in feedforward process, shape: 
%                       : [number of images, number of inputs in feedforward]
%       weight          : the weight matrix of this layer, shape: 
%                       : [number of inputs in feedforward, number of outputs in feedforward]
%
%   output parameter:
%       weight_grad     : the gradient of the weights, shape: 
%                       : [number of inputs in feedforward, number of outputs in feedforward]
%       out_sensitivity : the sensitivity to the lower layer, shape: 
%                       : [number of images, number of inputs in feedforward]
%
% Note : remember to divide by number of images in the calculation of gradients.

% TODO


[img_cnt, input_cnt] = size(in);
[~,output_cnt] = size(weight);
weight_grad = (in' * in_sensitivity)/img_cnt;
out_sensitivity = zeros(img_cnt, input_cnt);
bias_grad = sum(in_sensitivity)'/img_cnt;

out_sensitivity = in_sensitivity * weight'; 
%这里是hidden层，out_sensitivity = in.* (in_sensitivity * weight');
%这里乘上in是要在下一层乘的，例如input->hidden->output中，*in是在input做的，hidden只需要做后面一部分
% for img_idx = 1:img_cnt
%     out_sensitivity(img_idx,:) = in(img_idx,:).* (in_sensitivity * weight');%sum(repmat(in_sensitivity(img_idx,:),input_cnt,1) .* weight,2)';
% end

end

