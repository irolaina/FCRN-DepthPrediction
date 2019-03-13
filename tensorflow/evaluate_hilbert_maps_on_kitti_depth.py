#!/usr/bin/python3
# -*- coding: utf-8 -*-
# ===========
#  Libraries
# ===========
import os
import sys
import time
import warnings

import imageio
import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure, img_as_uint

# Custom Libraries
from modules import metrics
from modules.args import args
from modules.utils import settings

# =========================
#  [Test] Framework Config
# =========================
SAVE_TEST_DISPARITIES = True  # Default: True
showImages = False
# eval_tool = 'monodepth'
eval_tool = 'kitti_depth'

# ==================
#  Global Variables
# ==================
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu  # Limits TensorFlow to see only the specified GPU.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")  # Suppress Warnings


def read_text_file(filename):
    dataset_path = '/media/nicolas/nicolas_seagate/datasets/kitti/'

    print("\n[Dataloader] Loading '%s'..." % filename)
    try:
        data = np.genfromtxt(filename, dtype='str', delimiter='\t')
        # print(data.shape)

        # Parsing Data
        depth_continuous_filenames = list(data[:, 0])
        depth_semidense_filenames = list(data[:, 1])

        timer = -time.time()
        depth_continuous_filenames = [dataset_path + filename for filename in depth_continuous_filenames]
        depth_semidense_filenames = [dataset_path + filename for filename in depth_semidense_filenames]
        timer += time.time()
        print('time:', timer, 's\n')

    except OSError:
        raise OSError("Could not find the '%s' file." % filename)

    return depth_continuous_filenames, depth_semidense_filenames


def read_hilbert_maps_depth_image(filename):
    return imageio.imread(filename).astype('float32') / 3.0


def read_kitti_depth_depth_image(filename):
    return imageio.imread(filename).astype('float32') / 256.0


def imsave_as_uint16_png(filename, image_float32):
    # Converts the Predictions Images from float32 to uint16 and Saves as PNG Images

    image_uint16 = img_as_uint(exposure.rescale_intensity(image_float32, out_range='float'))
    imageio.imsave(filename, image_uint16)


def evaluate_hilbert_maps_on_kitti_depth():
    # Loads split file containing Hilbert Maps and KITTI Depth filenames
    hilbert_maps_filenames, kitti_depth_filenames = read_text_file(
        'data/new_splits/eigen_split_based_on_kitti_depth/eigen_test_kitti_depth_aligned_with_kitti_continuous_files.txt')
    assert len(kitti_depth_filenames) == len(hilbert_maps_filenames)

    print(len(kitti_depth_filenames), len(hilbert_maps_filenames))

    # Read Images
    hilbert_maps_depths = []
    kitti_depth_depths = []
    num_test_images = len(kitti_depth_filenames)

    fig, (ax1, ax2) = plt.subplots(2, 1)
    for i, (hilbert_maps_filename, kitti_depth_filename) in enumerate(
            list(zip(hilbert_maps_filenames, kitti_depth_filenames))):
        hilbert_maps_depth = read_hilbert_maps_depth_image(hilbert_maps_filename)  # Continuous
        kitti_depth_depth = read_kitti_depth_depth_image(kitti_depth_filename)  # Semi-Dense

        hilbert_maps_depths.append(hilbert_maps_depth)
        kitti_depth_depths.append(kitti_depth_depth)

        if showImages:
            ax1.imshow(hilbert_maps_depth)
            ax2.imshow(kitti_depth_depth)
            plt.draw()
            plt.pause(0.001)

        # Saves the Test Predictions as uint16 PNG Images
        if SAVE_TEST_DISPARITIES or eval_tool == 'monodepth':
            imsave_as_uint16_png(settings.output_tmp_pred_dir + 'pred' + str(i) + '.png', hilbert_maps_depth)
            imsave_as_uint16_png(settings.output_tmp_gt_dir + 'gt' + str(i) + '.png', kitti_depth_depth)

        print('{}/{}'.format(i, num_test_images))

    # Invokes Evaluation Tools
    if eval_tool == 'monodepth':
        metrics.evaluation_tool_monodepth(hilbert_maps_depths, kitti_depth_depths)
    elif eval_tool == 'kitti_depth':
        metrics.evaluation_tool_kitti_depth(num_test_images)
    else:
        raise SystemError("Invalid 'eval_tool' selected. Choose one of the options: 'monodepth' or 'kitti_depth'.")


# ======
#  Main
# ======
if __name__ == '__main__':
    evaluate_hilbert_maps_on_kitti_depth()

    print("\nDone.")
    sys.exit()
