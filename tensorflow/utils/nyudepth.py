# ===========
#  Libraries
# ===========
import glob
import os
import numpy as np
import tensorflow as tf
import sys

from .size import Size

# ==================
#  Global Variables
# ==================
LOG_INITIAL_VALUE = 1


# ===================
#  Class Declaration
# ===================
# NYU Depth v2
# Image: (480, 640, 3) ?
# Depth: (480, 640)    ?
class NyuDepth(object):
    def __init__(self, machine):
        if machine == 'olorin':
            self.dataset_path = ''
        elif machine == 'xps':
            self.dataset_path = "/media/nicolas/Nícolas/datasets/nyu-depth-v2/images/"

        self.name = 'nyudepth'

        self.image_size = Size(480, 640, 3)
        self.depth_size = Size(480, 640, 1)

        self.image_replace = [b'_colors.png', b'']
        self.depth_replace = [b'_depth.png', b'']

        # Data Range/Plot ColorSpace
        self.vmin = None
        self.vmax = None
        self.log_vmin = None
        self.log_vmax = None

        print("[Dataloader] NyuDepth object created.")

    def getFilenamesLists(self, mode):
        image_filenames = []
        depth_filenames = []

        if mode == 'train':
            dataset_path_aux = self.dataset_path + "training/*/"
        elif mode == 'test':
            dataset_path_aux = self.dataset_path + "testing/*/"
        else:
            sys.exit()

        # Finds input images and labels inside list of folders.
        for folder in glob.glob(dataset_path_aux):
            # print(folder)
            os.chdir(folder)

            for file in glob.glob('*_colors.png'):
                # print(file)
                image_filenames.append(folder + file)

            for file in glob.glob('*_depth.png'):
                # print(file)
                depth_filenames.append(folder + file)

            # print()

        # Alphabelly Sort the List of Strings
        image_filenames.sort()
        depth_filenames.sort()

        return image_filenames, depth_filenames

    def getFilenamesTensors(self, image_filenames, depth_filenames):
        tf_image_filenames = tf.constant(image_filenames)
        tf_depth_filenames = tf.constant(depth_filenames)

        return tf_image_filenames, tf_depth_filenames
