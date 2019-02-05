function D = depth_read (filename)
% loads depth map D from png file
% for details see readme.txt

I = imread(filename);
D = double(I)/256;
D(I==0) = -1;

