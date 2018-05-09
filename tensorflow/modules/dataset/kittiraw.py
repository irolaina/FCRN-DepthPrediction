# ===========
#  Libraries
# ===========
import glob
import os
import numpy as np
import tensorflow as tf
import sys
import time

from ..size import Size
from ..filenames import FilenamesHandler

# ==================
#  Global Variables
# ==================
LOG_INITIAL_VALUE = 1


# ===========
#  Functions
# ===========


# ===================
#  Class Declaration
# ===================
# KittiRaw Residential Continuous
# Image: (375, 1242, 3) uint8
# Depth: (375, 1242)    uint8
class KittiRaw(FilenamesHandler):
    def __init__(self, machine):
        super().__init__()
        if machine == 'olorin':
            self.dataset_path = ''
        elif machine == 'xps':
            self.dataset_path = "../../data/residential_continuous/"

        self.name = 'kittiraw'

        self.image_size = Size(375, 1242, 3)
        self.depth_size = Size(375, 1242, 1)

        self.image_replace = [b'/imgs/', b'']
        self.depth_replace = [b'/dispc/', b'']

        # Data Range/Plot ColorSpace
        self.vmin = 0
        self.vmax = 240
        self.log_vmin = np.log(self.vmin + LOG_INITIAL_VALUE)
        self.log_vmax = np.log(self.vmax)

        print("[Dataloader] KittiRaw object created.")

    def getFilenamesLists(self, mode):
        image_filenames = []
        depth_filenames = []

        file = 'data/' + self.name + '_' + mode + '.txt'

        if os.path.exists(file):
            data = self.loadList(file)

            # Parsing Data
            image_filenames = list(data[:, 0])
            depth_filenames = list(data[:, 1])
        else:
            print("[Dataloader] '%s' doesn't exist..." % file)
            print("[Dataloader] Searching files using glob (This may take a while)...")

            # Finds input images and labels inside list of folders.
            start = time.time()
            image_filenames = glob.glob(self.dataset_path + mode + "ing/imgs/*")
            depth_filenames = glob.glob(self.dataset_path + mode + "ing/dispc/*")

            # TODO: Adicionar Sequential Search

            print("time: %f s" % (time.time() - start))

            # TODO: Fazer shuffle
            # TODO: Eu acho que não precisa mais disso
            # Alphabelly Sort the List of Strings
            image_filenames.sort()
            depth_filenames.sort()

            self.saveList(image_filenames, depth_filenames, mode)

        return image_filenames, depth_filenames
