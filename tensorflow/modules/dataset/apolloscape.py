# ===========
#  Libraries
# ===========
import glob
import os
import numpy as np
import tensorflow as tf
import sys

from ..size import Size

# ==================
#  Global Variables
# ==================
LOG_INITIAL_VALUE = 1


# ===================
#  Class Declaration
# ===================
# Apollo Scape
# TODO: Add Info
# Image: (2710, 3384, 3) ?
# Depth: (2710, 3384)    ?
class Apolloscape(object):
    def __init__(self, machine):
        if machine == 'olorin':
            self.dataset_path = ''
        elif machine == 'xps':
            self.dataset_path = "/media/nicolas/Nícolas/datasets/apolloscape/"

        self.name = 'apolloscape'

        self.image_size = Size(2710, 3384, 3)
        self.depth_size = Size(2710, 3384, 1)

        self.image_filenames = []
        self.depth_filenames = []

        self.image_replace = [b'/ColorImage/', b'']
        self.depth_replace = [b'/Depth/', b'']

        # Data Range/Plot ColorSpace # TODO: Terminar
        self.vmin = None
        self.vmax = None
        self.log_vmin = None
        self.log_vmax = None

        print("[Dataloader] Apolloscape object created.")

    def getFilenamesLists(self, mode):
        # if mode == 'train':
        #     dataset_path_aux = self.dataset_path + "training/*/"
        # elif mode == 'test':
        #     dataset_path_aux = self.dataset_path + "testing/*/"
        # else:
        #     sys.exit()

        # Finds input images and labels inside list of folders.
        self.image_filenames = glob.glob(self.dataset_path + "ColorImage/*/*/*")  # ...ColorImage/Record*/Camera */*.png
        self.depth_filenames = glob.glob(self.dataset_path + "Depth/*/*/*")  # ...Depth/Record*/Camera */*.png

        print(self.image_filenames[0])
        print(self.depth_filenames[0])
        print(len(self.image_filenames))
        print(len(self.depth_filenames))
        input("apollo")

        return self.image_filenames, self.depth_filenames
