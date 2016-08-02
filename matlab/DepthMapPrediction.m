function pred = DepthMapPrediction(imdb, net, varargin)

% Depth prediction (inference) using a trained model.
% Inputs (imdb) can be either from the NYUDepth_v2 or Make3D dataset, along
% with the corresponding trained model (net). Additionally, the evaluation
% can be run for any single image. MatConvNet library has to be already
% setup for this function to work properly.
% -------------------------------------------------------------------------
% Inputs:
%   - imdb: a structure with fields 'images' and 'depths' in the case of
%           the benchmark datasets with known ground truth. imdb could
%           alternatively be any single RGB image of size NxMx3 in [0,255]
%           or a tensor of D input images NxMx3xD.
%   - net:  a trained model of type struct (suitable to be converted to a 
%           DagNN object and successively processed using the DagNN 
%           wrapper). For testing on arbitrary images, use NYU model for 
%           indoor and Make3D model for outdoor scenes respectively.
% -------------------------------------------------------------------------

opts.gpu = false;           % Set to true (false) for GPU (CPU only) support 
opts.plot = false;          % Set to true to visualize the predictions during inference
opts = vl_argparse(opts, varargin);

% Set network properties
net = dagnn.DagNN.loadobj(net);
net.mode = 'test';
out = net.getVarIndex('prediction');
if opts.gpu
    net.move('gpu');
end

% Check input
if isa(imdb, 'struct')
    % case of benchmark datasets (NYU, Make3D)
    images = imdb.images;
    groundTruth = imdb.depths;
else
    % case of arbitrary image(s)
    images = imdb;
    images = imresize(images, net.meta.normalization.imageSize(1:2));
    groundTruth = [];
end

% Get output size for initialization
varSizes = net.getVarSizes({'data', net.meta.normalization.imageSize});  % get variable sizes
pred = zeros(varSizes{out}(1), varSizes{out}(2), varSizes{out}(3), size(images, 4));    % initiliaze 

if opts.plot, figure(); end

fprintf('predicting...\n');
for i = 1:size(images, 4)
    % get input image
    im = single(images(:,:,:,i));
    if opts.gpu
        im = gpuArray(im);
    end
    
    % run the CNN
    inputs = {'data', im};
    net.eval(inputs) ;
    
    % obtain prediction
    pred(:,:,i) = gather(net.vars(out).value);
    
    % visualize results
    if opts.plot
        colormap jet
        if ~isempty(groundTruth)
            subplot(1,3,1), imagesc(uint8(images(:,:,:,i))), title('RGB Input'), axis off
            subplot(1,3,2), imagesc(groundTruth(:,:,i)), title('Depth Ground Truth'), axis off
            subplot(1,3,3), imagesc(pred(:,:,i)), title('Depth Prediction'), axis off
        else
            subplot(1,2,1), imagesc(uint8(images(:,:,:,i))), title('RGB Input'), axis off
            subplot(1,2,2), imagesc(pred(:,:,i)), title('Depth Prediction'), axis off
        end
        drawnow;
    end
end
