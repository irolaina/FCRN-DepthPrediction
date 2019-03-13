###########################################################################
# THE KITTI VISION BENCHMARK: DEPTH PREDICTION/COMPLETION BENCHMARKS 2017 #
#       based on our publication Sparsity Invariant CNNs (3DV 2017)       #
#                                                                         #
#           Jonas Uhrig     Nick Schneider     Lukas Schneider            #
#           Uwe Franke       Thomas Brox        Andreas Geiger            #
#                                                                         #
#          Daimler R&D Sindelfingen       University of Freiburg          #
#          KIT Karlsruhe         ETH Zürich         MPI Tübingen          #
#                                                                         #
###########################################################################

This file describes the 2017 KITTI depth completion and single image depth
prediction benchmarks, consisting of 93k training and 1.5k test images.
Ground truth has been acquired by accumulating 3D point clouds from a
360 degree Velodyne HDL-64 Laserscanner and a consistency check using
stereo camera pairs. Please have a look at our publications for details.

Dataset description:
====================

If you unzip all downloaded files from the KITTI vision benchmark website
into the same base directory, your folder structure will look like this:

|-- devkit
|-- test_depth_completion_anonymous
  |-- image
    |-- 0000000000.png
    |-- ...
    |-- 0000000999.png
  |-- velodyne_raw
    |-- 0000000000.png
    |-- ...
    |-- 0000000999.png
|-- test_depth_prediction_anonymous
  |-- image
    |-- 0000000000.png
    |-- ...
    |-- 0000000999.png
|-- train
  |-- 2011_xx_xx_drive_xxxx_sync
    |-- proj_depth
      |-- groundtruth           # "groundtruth" describes our annotated depth maps
        |-- image_02            # image_02 is the depth map for the left camera
          |-- 0000000005.png    # image IDs start at 5 because we accumulate 11 frames
          |-- ...               # .. which is +-5 around the current frame ;)
        |-- image_03            # image_02 is the depth map for the right camera
          |-- 0000000005.png
          |-- ...
      |-- velodyne_raw          # this contains projected and temporally unrolled
        |-- image_02            # raw Velodyne laser scans
          |-- 0000000005.png
          |-- ...
        |-- image_03
          |-- 0000000005.png
          |-- ...
  |-- ... (all drives of all days in the raw KITTI dataset)
|-- val
  |-- (same as in train)
|-- val_selection_cropped       # 1000 images of size 1216x352, cropped and manually
  |-- groundtruth_depth         # selected frames from from the full validation split
    |-- 2011_xx_xx_drive_xxxx_sync_groundtruth_depth_xxxxxxxxxx_image_0x.png
    |-- ...
  |-- image
    |-- 2011_xx_xx_drive_xxxx_sync_groundtruth_depth_xxxxxxxxxx_image_0x.png
    |-- ...
  |-- velodyne_raw
    |-- 2011_xx_xx_drive_xxxx_sync_groundtruth_depth_xxxxxxxxxx_image_0x.png
    |-- ...

For train and val splits, the mapping from the KITTI raw dataset to our
generated depth maps and projected raw laser scans can be extracted. All
files are uniquely identified by their recording date, the drive ID as well
as the camera ID (02 for left, 03 for right camera).

Submission instructions:
========================

NOTE: WHEN SUBMITTING RESULTS, PLEASE STORE THEM IN THE SAME DATA FORMAT IN
WHICH THE GROUND TRUTH DATA IS PROVIDED (SEE BELOW), USING THE FILE NAMES
0000000000.png TO 0000000999.png (DEPTH COMPLETION) OR 0000000499.png (DEPTH
PREDICTION). CREATE A ZIP ARCHIVE OF THEM AND STORE YOUR RESULTS IN YOUR
ZIP'S ROOT FOLDER:

|-- zip
  |-- 0000000000.png
  |-- ...
  |-- 0000000999.png

Data format:
============

Depth maps (annotated and raw Velodyne scans) are saved as uint16 PNG images,
which can be opened with either MATLAB, libpng++ or the latest version of
Python's pillow (from PIL import Image). A 0 value indicates an invalid pixel
(ie, no ground truth exists, or the estimation algorithm didn't produce an
estimate for that pixel). Otherwise, the depth for a pixel can be computed
in meters by converting the uint16 value to float and dividing it by 256.0:

disp(u,v)  = ((float)I(u,v))/256.0;
valid(u,v) = I(u,v)>0;

Evaluation Code:
================

For transparency we have included the benchmark evaluation code in the
sub-folder 'cpp' of this development kit. It can be compiled by running
the 'make.sh' script. Run it using two arguments:

./evaluate_depth gt_dir prediction_dir

Note that gt_dir is most likely '../../val_selection_cropped/groundtruth_depth'
if you unzipped all files in the same base directory. We also included a sample
result of our proposed approach for the validation split ('predictions/sparseConv_val').
