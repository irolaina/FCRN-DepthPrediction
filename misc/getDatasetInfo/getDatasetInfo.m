%% Matlab Init
clc
clear all
close all

addpath 'align_figure'

%% Kitti 2012
kitti2012_image = imread('/media/nicolas/Nícolas/datasets/kitti/stereo/stereo2012/data_stereo_flow/training/colored_0/000000_10.png');
kitti2012_depth = imread('/media/nicolas/Nícolas/datasets/kitti/stereo/stereo2012/data_stereo_flow/training/disp_noc/000000_10.png');
kitti2012_depth_true = double(kitti2012_depth)/256.0;

%% Kitti 2015
kitti2015_image = imread('/media/nicolas/Nícolas/datasets/kitti/stereo/stereo2015/data_scene_flow/training/image_2/000000_10.png');
kitti2015_depth = imread('/media/nicolas/Nícolas/datasets/kitti/stereo/stereo2015/data_scene_flow/training/disp_noc_0/000000_10.png');
kitti2015_depth_true = double(kitti2015_depth)/256.0;

%% Kitti Depth
kittiDepth_image = imread('/media/nicolas/Nícolas/datasets/kitti/raw_data/data/city/2011_09_26_drive_0001_sync/image_02/data/0000000005.png');
kittiDepth_depth = imread('/media/nicolas/Nícolas/datasets/kitti/depth/depth_prediction/data/train/2011_09_26_drive_0001_sync/proj_depth/groundtruth/image_02/0000000005.png');
kittiDepth_depth_true = double(kittiDepth_depth)/256.0;

%% KittiContinuous (Residential)
kitticontinuous_image = imread('/media/nicolas/Nícolas/datasets/kitti/continuous/residential/imgs/residential_2011_09_26_drive_0019_sync_0000000000.png');
kitticontinuous_depth = imread('/media/nicolas/Nícolas/datasets/kitti/continuous/residential/dispc/residential_2011_09_26_drive_0019_sync_0000000000.png');
kitticontinuous_depth_true = double(kitticontinuous_depth)/3.0;

%% NYU Depth v2
nyu_image = imread('/media/nicolas/Nícolas/datasets/nyu-depth-v2/data/images/training/basement/00489_colors.png');
nyu_depth = imread('/media/nicolas/Nícolas/datasets/nyu-depth-v2/data/images/training/basement/00489_depth.png');
nyu_depth_true = double(nyu_depth)/1000.0; % According to the convert.py file

%% ApolloScape
apollo_image = imread('/media/nicolas/Nícolas/datasets/apolloscape/data/ColorImage/Record001/Camera 5/170927_063811892_Camera_5.jpg');
apollo_depth = imread('/media/nicolas/Nícolas/datasets/apolloscape/data/Depth/Record001/Camera 5/170927_063811892_Camera_5.png');

apollo_depth(apollo_depth == 65535) = 0;
apollo_depth_true = double(apollo_depth)/200.0;

%% Plot
h1 = figure('Name','Kitti 2012','NumberTitle','on');
subplot(3,1,1), imshow(kitti2012_image), title('image')
subplot(3,1,2), imshow(kitti2012_depth), title('depth'), c = colorbar; c.Label.String = '(uint16)';
subplot(3,1,3), image(kitti2012_depth_true), title('true depth'), c = colorbar; c.Label.String = 'Disparity'; axis image

h2 = figure('Name','Kitti 2015','NumberTitle','on');
subplot(3,1,1), imshow(kitti2015_image), title('image')
subplot(3,1,2), imshow(kitti2015_depth), title('depth'), c = colorbar; c.Label.String = '(uint16)';
subplot(3,1,3), image(kitti2015_depth_true), title('true depth'), c = colorbar; c.Label.String = 'Disparity'; axis image

h3 = figure('Name','KittiDepth','NumberTitle','on');
subplot(3,1,1), imshow(kittiDepth_image), title('image')
subplot(3,1,2), imshow(kittiDepth_depth), title('depth'), c = colorbar; c.Label.String = '(uint16)';
subplot(3,1,3), image(kittiDepth_depth_true), title('true depth'), c = colorbar; c.Label.String = 'Depth (in meters)'; axis image

h4 = figure('Name','KittiContinuous (Residential)','NumberTitle','on');
subplot(3,1,1), imshow(kitticontinuous_image), title('image')
subplot(3,1,2), imshow(kitticontinuous_depth), title('depth'), c = colorbar; c.Label.String = '(uint8)';
subplot(3,1,3), image(kitticontinuous_depth_true), title('true depth'), c = colorbar; c.Label.String = 'Depth (in meters)'; axis image

h5 = figure('Name','NYU Depth v2','NumberTitle','on');
subplot(3,1,1), imshow(nyu_image), title('image')
subplot(3,1,2), imshow(nyu_depth), title('depth'), c = colorbar; c.Label.String = '(uint16)';
subplot(3,1,3), image(nyu_depth_true), title('true depth'), c = colorbar; c.Label.String = 'Depth (in meters)'; axis image % FIXME: Colorbar range

h6 = figure('Name','Apolloscape','NumberTitle','on');
subplot(3,1,1), imshow(apollo_image), title('image')
subplot(3,1,2), imshow(apollo_depth), title('depth'), c = colorbar; c.Label.String = '(uint16)';
subplot(3,1,3), image(apollo_depth_true), title('true depth'), c = colorbar; c.Label.String = 'Depth (in meters)'; axis image

align_figure([h1 h2 h3 h4 h5]); 