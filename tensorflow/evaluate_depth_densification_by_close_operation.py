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
import cv2

# =========================
#  [Test] Framework Config
# =========================
SAVE_TEST_DISPARITIES = True  # Default: True
showImages = True

# ==================
#  Global Variables
# ==================
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu  # Limits TensorFlow to see only the specified GPU.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")  # Suppress Warnings


def read_text_file(filename, dataset_path='/media/nicolas/nicolas_seagate/datasets/kitti/'):

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

def read_depth_image(filename, div=1.0):
    return imageio.imread(filename).astype('float32') / div


def imsave_as_uint16_png(filename, image_float32):
    # Converts the Predictions Images from float32 to uint16 and Saves as PNG Images

    image_uint16 = img_as_uint(exposure.rescale_intensity(image_float32, out_range='float'))
    imageio.imsave(filename, image_uint16)


def evaluate_densification():
    # Loads split file containing Input and Output filenames
    # input_filenames, output_filenames = read_text_file('data/new_splits/eigen_split_based_on_kitti_depth/eigen_test_kitti_depth_aligned_with_kitti_continuous_files.txt')
    input_filenames = ['/home/nicolas/Downloads/depth_interpolation/close/0000000005_close_k_2.png']
    output_filenames = ['/home/nicolas/Downloads/depth_interpolation/close/0000000005.png']

    assert len(input_filenames) == len(output_filenames)
    print(len(input_filenames), len(output_filenames))

    # Read Images
    input_depths, output_depths = [], []
    num_test_images = len(output_filenames)

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1)
    for i, (input_filename, output_filename) in enumerate(list(zip(input_filenames, output_filenames))):
        close_depth = read_depth_image(input_filename, 256.0)
        kitti_depth_depth = read_depth_image(output_filename, 256.0)

        # Fix Data shift caused by close operation
        # TODO: Esta correção só precisa ser feita se o kernel utilizado no close é par.
        rows, cols = close_depth.shape
        tx, ty = -1, -1 # Offsets
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        close_depth = cv2.warpAffine(close_depth, M, (cols, rows))

        artefacts = close_depth - kitti_depth_depth
        real_proof = kitti_depth_depth+artefacts
        real_proof2 = (close_depth - real_proof)*1000
        print(real_proof2)

        print(close_depth[close_depth > 0.0].size,
              kitti_depth_depth[kitti_depth_depth > 0.0].size,
              artefacts[artefacts > 0.0].size)  # Number of Valid Pixels

        print(close_depth.shape)
        print(kitti_depth_depth.shape)
        print(np.min(close_depth), np.max(close_depth))

        if showImages:
            ax1.imshow(close_depth)
            ax1.set_title('close(k=2)')
            ax2.imshow(kitti_depth_depth)
            ax2.set_title('KITTI Depth')
            ax3.imshow(artefacts)
            ax3.set_title('Artefacts')
            ax4.imshow(real_proof)
            ax4.set_title('KITTI Depth + Artefacts')
            ax5.imshow(real_proof2)
            ax5.set_title('(KITTI Depth + Artefacts)-close(k=2)')

            # plt.draw()
            # plt.pause(0.001)
            plt.show()

        # Saves the Test Predictions as uint16 PNG Images
        if SAVE_TEST_DISPARITIES or args.eval_tool == 'monodepth':
            imsave_as_uint16_png(settings.output_tmp_pred_dir + 'pred' + str(i) + '.png', close_depth)
            imsave_as_uint16_png(settings.output_tmp_gt_dir + 'gt' + str(i) + '.png', kitti_depth_depth)

        input_depths.append(close_depth)
        output_depths.append(kitti_depth_depth)

        print('{}/{}'.format(i+1, num_test_images))


    # Invokes Evaluation Tools
    if args.eval_tool == 'monodepth':
        metrics.evaluation_tool_monodepth(input_depths, output_depths)
    elif args.eval_tool == 'kitti_depth':
        metrics.evaluation_tool_kitti_depth(num_test_images)
    else:
        raise SystemError("Invalid 'eval_tool' selected. Choose one of the options: 'monodepth' or 'kitti_depth'.")


# ======
#  Main
# ======
if __name__ == '__main__':
    evaluate_densification()

    print("\nDone.")
    sys.exit()
