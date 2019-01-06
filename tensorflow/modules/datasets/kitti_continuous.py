# ========
#  README
# ========
# KittiContinuous
# Uses Depth Maps: measures distances [close - LOW values, far - HIGH values]
# Image: (375, 1242, 3) uint8
# Depth: (375, 1242)    uint8

# -----
# Dataset Guidelines by Vitor Guizilini
# -----
# Raw Depth image to Depth (meters):
# depth(u,v) = ((float)I(u,v))/3.0;
# valid(u,v) = I(u,v)>0;
# -----


# ===========
#  Libraries
# ===========
import glob
import os

import numpy as np

from .dataset import Dataset


# ===================
#  Class Declaration
# ===================
class KittiContinuous(Dataset):
    def __init__(self, *args, **kwargs):
        super(KittiContinuous, self).__init__(*args, **kwargs)

    def getFilenamesLists(self, mode, test_split='', test_file_path=''):
        file = self.get_file_path(mode, test_split, test_file_path)

        if os.path.exists(file):
            image_filenames, depth_filenames = self.read_text_file(file, self.dataset_path)
        else:
            raise(DeprecationWarning)
            input("aki")

        return image_filenames, depth_filenames, file
