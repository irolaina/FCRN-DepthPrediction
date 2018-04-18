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
# KittiRaw Residential Continuous
# Image: (375, 1242, 3) uint8
# Depth: (375, 1242)    uint8
class KittiRaw(object):
    def __init__(self, machine):
        if machine == 'olorin':
            self.dataset_path = ''
        elif machine == 'xps':
            self.dataset_path = "../../data/residential_continuous/"

        self.name = 'kittiraw'

        self.image_size = Size(375, 1242, 3)
        self.depth_size = Size(375, 1242, 1)

        self.image_filenames = None
        self.depth_filenames = None

        self.image_replace = [b'/imgs/', b'']
        self.depth_replace = [b'/dispc/', b'']

        # Data Range/Plot ColorSpace
        self.vmin = 0
        self.vmax = 240
        self.log_vmin = np.log(self.vmin + LOG_INITIAL_VALUE)
        self.log_vmax = np.log(self.vmax)

        print("[Dataloader] KittiRaw object created.")

    def getFilenamesLists(self, mode):
        if mode == 'train':
            dataset_path_aux = self.dataset_path + "training/"
        elif mode == 'test':
            dataset_path_aux = self.dataset_path + "testing/"
        else:
            sys.exit()

        # Finds input images and labels inside list of folders.
        self.image_filenames = glob.glob(dataset_path_aux + "imgs/*")
        self.depth_filenames = glob.glob(dataset_path_aux + "dispc/*")

        return self.image_filenames, self.depth_filenames

    def getFilenamesTensors(self):
        self.tf_image_filenames = tf.constant(self.image_filenames)
        self.tf_depth_filenames = tf.constant(self.depth_filenames)

        return self.tf_image_filenames, self.tf_depth_filenames
