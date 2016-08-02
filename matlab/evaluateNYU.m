function evaluateNYU

% Evaluation of depth prediction on NYU Depth v2 dataset. 

% -------------------------------------------------------------------------
% Setup MatConvNet
% -------------------------------------------------------------------------

% Set your matconvnet path here:
matconvnet_path = '../../matconvnet-1.0-beta20';
setupMatConvNet(matconvnet_path);

% -------------------------------------------------------------------------
% Options
% -------------------------------------------------------------------------

opts.dataDir = fullfile(pwd, 'NYU');    % working directory
opts.interp = 'nearest';                % interpolation method applied during resizing

netOpts.gpu = true;     % set to true to enable GPU support      
netOpts.plot = true;    % set to true to visualize the predictions during inference          

% -------------------------------------------------------------------------
% Prepate data
% -------------------------------------------------------------------------

imdb = get_NYUDepth_v2(opts);
net = get_model(opts);

% Test set
testSet.images = imdb.images(:,:,:, imdb.set == 2);
testSet.depths = imdb.depths(:,:, imdb.set == 2);

% Prepare input for evaluation through the network, in accordance to the
% way the model was trained for the NYU dataset. No processing is applied
% to the ground truth. 
meta = net.meta.normalization;  % information about input
res = meta.imageSize(1:2) + 2*meta.border; 
testSet.images = imresize(testSet.images, res, opts.interp);  % resize
testSet.images = testSet.images(1+meta.border(1):end-meta.border(1), 1+meta.border(2):end-meta.border(2), :, :);  % center crop

% -------------------------------------------------------------------------
% Evaluate network 
% -------------------------------------------------------------------------

% Get predictions
predictions = DepthMapPrediction(testSet, net, netOpts);
predictions = squeeze(predictions); % remove singleton dimensions
predictions = imresize(predictions, [size(testSet.depths,1), size(testSet.depths,2)], 'bilinear'); %rescale

% Error calculation 
errors = error_metrics(predictions, testSet.depths, []);

% Save results
fprintf('\nsaving predictions...');
save(fullfile(opts.dataDir, 'results.mat'), 'predictions', 'errors', '-v7.3');
fprintf('done!\n');



function imdb = get_NYUDepth_v2(opts)
% -------------------------------------------------------------------------
% Download required data 
% -------------------------------------------------------------------------

opts.dataDir = fullfile(opts.dataDir, 'data');
if ~exist(opts.dataDir, 'dir'), mkdir(opts.dataDir); end

% Download dataset
filename = fullfile(opts.dataDir, 'nyu_depth_v2_labeled.mat');
if ~exist(filename, 'file')
    url = 'http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat';
    fprintf('downloading dataset (~2.8 GB): %s\n', url);
    websave(filename, url);
end

% Download official train/test split
filename_splits = fullfile(opts.dataDir, 'splits.mat');
if ~exist(filename_splits, 'file')
    url_split = 'http://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat';
    fprintf('downloading train/test split: %s\n', url_split);
    websave(filename_splits, url_split);
end

% Load dataset and splits
fprintf('loading data to workspace...');
data = load(filename);
splits = load(filename_splits);

% Store necessary information to imdb structure
imdb.images = single(data.images);  %(no mean subtraction has been performed)
imdb.depths = single(data.depths);  %depth filled-in values
imdb.set(splits.trainNdxs) = 1;     %training indices (ignored for inference)
imdb.set(splits.testNdxs) = 2;      %testing indices (on which evaluation is performed)
fprintf(' done!\n');



function net = get_model(opts)
% -------------------------------------------------------------------------
% Download trained models
% -------------------------------------------------------------------------

opts.dataDir = fullfile(opts.dataDir, 'models');
if ~exist(opts.dataDir, 'dir'), mkdir(opts.dataDir); end

filename = fullfile(opts.dataDir, 'NYU_ResNet-UpProj.mat');
if ~exist(filename, 'file')
    url = 'http://campar.in.tum.de/files/rupprecht/depthpred/NYU_ResNet-UpProj.zip';
    fprintf('downloading trained model: %s\n', url);
    unzip(url, opts.dataDir);
end

net = load(filename);

