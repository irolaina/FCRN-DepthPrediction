function evaluateMake3D

% Evaluation of depth prediction on Make3D dataset.

% -------------------------------------------------------------------------
% Setup MatConvNet
% -------------------------------------------------------------------------

% Set your matconvnet path here:
matconvnet_path = '../../matconvnet-1.0-beta20';
setupMatConvNet(matconvnet_path);

% -------------------------------------------------------------------------
% Options
% -------------------------------------------------------------------------

opts.dataDir = fullfile(pwd, 'Make3D');     % working directory
opts.interp = 'nearest';    % interpolation method applied during resizing
opts.imageSize = [460,345]; % desired image size for evaluation

netOpts.gpu = true;     % set to true to enable GPU support
netOpts.plot = true;    % set to true to visualize the predictions during inference

% -------------------------------------------------------------------------
% Prepate data
% -------------------------------------------------------------------------

imdb = get_Make3D(opts);
net = get_model(opts);

% Test set
testSet.images = imdb.images(:,:,:, imdb.set == 2);
testSet.depths = imdb.depths(:,:, imdb.set == 2);

% resize images to input resolution (equal to round(opts.imageSize/2))
testSet.images = imresize(testSet.images, net.meta.normalization.imageSize(1:2), opts.interp);    
% resize depth to opts.imageSize resolution
testSet.depths = imresize(testSet.depths, opts.imageSize, opts.interp);     

% -------------------------------------------------------------------------
% Evaluate network
% -------------------------------------------------------------------------

% Get predictions
predictions = DepthMapPrediction(testSet, net, netOpts);
predictions = squeeze(predictions); % remove singleton dimensions
predictions = imresize(predictions, [size(testSet.depths,1), size(testSet.depths,2)], 'bilinear'); %rescale

% Error calculation
c1_mask = testSet.depths > 0 & testSet.depths < 70;
errors = error_metrics(predictions, testSet.depths, c1_mask);

% Save results
fprintf('\nsaving predictions...');
save(fullfile(opts.dataDir, 'results.mat'), 'predictions', 'errors', '-v7.3');
fprintf('done!\n');


function imdb = get_Make3D(opts)
% -------------------------------------------------------------------------
% Download required data (test only)
% -------------------------------------------------------------------------

opts.dataDirImages = fullfile(opts.dataDir, 'data', 'Test134');
opts.dataDirDepths = fullfile(opts.dataDir, 'data', 'Gridlaserdata');

% Download test set
if ~exist(opts.dataDirImages, 'dir')
    fprintf('downloading Make3D testing images (~190 MB)...');
    mkdir(opts.dataDirImages);
    untar('http://www.cs.cornell.edu/~asaxena/learningdepth/Test134.tar.gz', fileparts(opts.dataDirImages));
    fprintf('done.\n');
end

if ~exist(opts.dataDirDepths, 'dir')
    fprintf('downloading Make3D testing depth maps (~22 MB)...');
    mkdir(opts.dataDirDepths);   
    untar('http://www.cs.cornell.edu/~asaxena/learningdepth/Test134Depth.tar.gz', fileparts(opts.dataDirDepths));
    fprintf('done.\n');
end

fprintf('preparing testing data...');
img_files = dir(fullfile(opts.dataDirImages, 'img-*.jpg'));
depth_files = dir(fullfile(opts.dataDirDepths, 'depth_sph_corr-*.mat'));

% Verify that the correct number of files has been found
assert(numel(img_files)==134, 'Incorrect number of Make3D test images. \n');
assert(numel(depth_files)==134, 'Incorrect number of Make3D test depths. \n');

% Read dataset files and store necessary information to imdb structure
for i = 1:numel(img_files)
    imdb.images(:,:,:,i) = single(imread(fullfile(opts.dataDirImages, img_files(i).name)));   % get RGB image
    gt = load(fullfile(opts.dataDirDepths, depth_files(i).name));
    imdb.depths(:,:,i) = single(gt.Position3DGrid(:,:,4));  % get depth channel
    imdb.set(i) = 2;
end
fprintf(' done!\n');



function net = get_model(opts)
% -------------------------------------------------------------------------
% Download trained models
% -------------------------------------------------------------------------

opts.dataDir = fullfile(opts.dataDir, 'models');
if ~exist(opts.dataDir, 'dir'), mkdir(opts.dataDir); end

filename = fullfile(opts.dataDir, 'Make3D_ResNet-UpProj.mat');
if ~exist(filename, 'file')
    url = 'http://campar.in.tum.de/files/rupprecht/depthpred/Make3D_ResNet-UpProj.zip';
    fprintf('downloading trained model: %s\n', url);
    unzip(url, opts.dataDir);
end

net = load(filename);

