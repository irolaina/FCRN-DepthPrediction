function results = error_metrics(pred, gt, mask)

% Compute error metrics on benchmark datasets
% -------------------------------------------------------------------------

% make sure predictions and ground truth have same dimensions
if size(pred) ~= size(gt)
    pred = imresize(pred, [size(gt,1), size(gt,2)], 'bilinear');  
end

if isempty(mask)
    n_pxls = numel(gt);
else
    n_pxls = sum(mask(:));  % average over valid pixels only
end

fprintf('\n Errors computed over the entire test set \n');
fprintf('------------------------------------------\n');

% Mean Absolute Relative Error
rel = abs(gt(:) - pred(:)) ./ gt(:);    % compute errors
rel(~mask) = 0;                         % mask out invalid ground truth pixels
rel = sum(rel) / n_pxls;                % average over all pixels
fprintf('Mean Absolute Relative Error: %4f\n', rel);

% Root Mean Squared Error
rms = (gt(:) - pred(:)).^2;
rms(~mask) = 0;
rms = sqrt(sum(rms) / n_pxls);
fprintf('Root Mean Squared Error: %4f\n', rms);

% LOG10 Error
lg10 = abs(log10(gt(:)) - log10(pred(:)));
lg10(~mask) = 0;
lg10 = sum(lg10) / n_pxls ; 
fprintf('Mean Log10 Error: %4f\n', lg10);

results.rel = rel;
results.rms = rms;
results.log10 = lg10;
