# ===========
#  Libraries
# ===========
import glob
import os
import tensorflow as tf
import sys
import numpy as np

from ..size import Size

# ==================
#  Global Variables
# ==================
LOG_INITIAL_VALUE = 1


# ===================
#  Class Declaration
# ===================
# Kitti Stereo 2015
# TODO: Add info
# Image: (?, ?, 3) ?
# Depth: (?, ?)    ?
class Kitti2015(object):
    def __init__(self, machine):
        if machine == 'olorin':
            self.dataset_path = ''
        elif machine == 'xps':
            self.dataset_path = "/media/nicolas/Nícolas/datasets/kitti/stereo/stereo2015/data_scene_flow/"

        self.name = 'kitti2015'

        self.image_size = Size(376, 1241, 3)  # TODO: Validar
        self.depth_size = Size(376, 1226, 1)  # TODO: Validar

        self.image_replace = [b'_colors.png', b'']  # TODO: Terminar
        self.depth_replace = [b'_depth.png', b'']  # TODO: Terminar

        # Data Range/Plot ColorSpace
        self.vmin = None
        self.vmax = None
        self.log_vmin = None
        self.log_vmax = None

        print("[Dataloader] Kitti2015 object created.")

    # TODO: Terminar
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

        # self.saveLists(image_filenames, depth_filenames) # FIXME: Doesn't Save

        return image_filenames, depth_filenames
