X = [extract_image('images/1.png'),extract_image('images/2.png'),extract_image('images/3.png'),extract_image('images/4.png'),extract_image('images/5.png')];
y = [6,6,6,5,8, 1,4,7,1,2, 2,4,5,7,0, 1,8,8,7,5, 1,8,6,4,3];

save hack_data X y;
for i=10:300
    y_test = hack(['images/',num2str(i),'.png']);
    disp(y_test);
end