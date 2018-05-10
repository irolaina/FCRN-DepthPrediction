# ========
#  README
# ========
# Kitti Stereo 2015
# Image: (375, 1242, 3) uint8
# Depth: (375, 1242)    uint16

# Dataset Guidelines
# disp(u,v)  = ((float)I(u,v))/256.0;
# valid(u,v) = I(u,v)>0;


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
class Kitti2015(FilenamesHandler):
    def __init__(self, machine):
        super().__init__()
        if machine == 'olorin':
            self.dataset_path = ''
        elif machine == 'xps':
            self.dataset_path = "/media/nicolas/Nícolas/datasets/kitti/stereo/stereo2015/data_scene_flow/"

        self.name = 'kitti2015'

        self.image_size = Size(375, 1242, 3)
        self.depth_size = Size(375, 1242, 1)

        print("[Dataloader] Kitti2015 object created.")

    def getFilenamesLists(self, mode):  # FIXME
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

        # TODO: Adicionar Sequential Search
        # TODO: Fazer shuffle
        # TODO: Eu acho que não precisa mais disso
        # Alphabelly Sort the List of Strings
        image_filenames.sort()
        depth_filenames.sort()

        # self.saveLists(image_filenames, depth_filenames, self.name, mode) # FIXME: Doesn't Save

        return image_filenames, depth_filenames
