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
# NYU Depth v2
# TODO: Add info
# Image: (480, 640, 3) ?
# Depth: (480, 640)    ?
class NyuDepth(FilenamesHandler):
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

        # Data Range/Plot ColorSpace # TODO: Terminar
        self.vmin = None
        self.vmax = None
        self.log_vmin = None
        self.log_vmax = None

        print("[Dataloader] NyuDepth object created.")

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
            for folder in glob.glob(self.dataset_path + mode + "ing/*/"):
                # print(folder)
                os.chdir(folder)

                for file in glob.glob('*_colors.png'):
                    # print(file)
                    image_filenames.append(folder + file)

                for file in glob.glob('*_depth.png'):
                    # print(file)
                    depth_filenames.append(folder + file)

            # TODO: Adicionar Sequential Search

            print("time: %f s" % (time.time() - start))

            # TODO: Fazer shuffle
            # TODO: Eu acho que não precisa mais disso
            # Alphabelly Sort the List of Strings
            image_filenames.sort()
            depth_filenames.sort()

            self.saveList(image_filenames, depth_filenames, mode)

        return image_filenames, depth_filenames
