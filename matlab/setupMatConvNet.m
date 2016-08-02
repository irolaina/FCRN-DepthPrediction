function setupMatConvNet(matconvnet_path)

% check path
if ~exist(fullfile(matconvnet_path, 'matlab', 'vl_setupnn.m'), 'file')
    error('Count not find MatConvNet in "%s"!\nPlease point matcovnet_path to the correct directory.\n', matconvnet_path);
end

% check if it is the right version (beta-20)
mcnv = getMatConvNetVersion(matconvnet_path);
if ~strcmp(mcnv, '1.0-beta20')
    error('Your MatConvNet version (%s) is not the required version 1.0-beta20. Please download and compile the right version.', mcnv);
end

% if everything is fine, then set up
run(fullfile(matconvnet_path, 'matlab', 'vl_setupnn.m'));



function versionName = getMatConvNetVersion(matconvnet_path)

fid = fopen(fullfile(matconvnet_path, 'Makefile'), 'rt');
s = textscan(fid, '%s', 'delimiter', '\n');
fclose(fid);
idxs = find(~cellfun(@isempty,strfind(s{1}, 'VER = ')));
mcnVersion = s{1}(idxs(1));
versionName = mcnVersion{1}(7:end);