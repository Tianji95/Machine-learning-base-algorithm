function img = hack_pca(filename)
% Input: filename -- input image file name/path
% Output: img -- image without rotation

img_r = double(imread(filename))./255;
[N, P] = size(img_r);
pixel_count = sum(sum(img_r < 0.8));
reshape_point= zeros(pixel_count, 2);
pixel_count = 1;
for row_idx = 1:N
    for col_idx = 1:P
        if img_r(row_idx, col_idx) < 0.8
            reshape_point(pixel_count,1) = row_idx;
            reshape_point(pixel_count,2) = col_idx;
            pixel_count = pixel_count + 1;
        end
    end
end
[eigvector, eigvalue] = pca(reshape_point);
A = eigvector;
img_after = A'*reshape_point';
disp(size(img_after))
Max = floor(max(max(img_after)));
Min = min(min(img_after));
show_img_after = ones(floor(Max-Min)+2, floor(Max-Min)+2);

for pixel = 1:pixel_count-1
    show_img_after(Max-(floor(img_after(2,pixel)-Min)+1),floor(img_after(1,pixel)-Min)+1) = img_r(reshape_point(pixel,1), reshape_point(pixel,2));
end

figure
imshow(show_img_after);
% img = (img_r*A)*A';
% figure
% imshow(uint8(img));


% YOUR CODE HERE

end