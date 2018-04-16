# ===========
#  Libraries
# ===========
import numpy as np
import tensorflow as tf

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
    def __init__(self):
        self.dataset_path = ''  # TODO: Terminar
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
        self.log_vmin = np.log(self.vmin+LOG_INITIAL_VALUE)
        self.log_vmax = np.log(self.vmax)

        print("[Dataloader] KittiRaw object created.")

    # TODO: Terminar
    def getFilenamesLists(self):
        return self.image_filenames, self.depth_filenames

    def getFilenamesTensors(self):
        search_image_files_str = "../../data/residential_continuous/training/imgs/*.png"
        search_depth_files_str = "../../data/residential_continuous/training/dispc/*.png"

        self.tf_image_filenames = tf.train.match_filenames_once(search_image_files_str)
        self.tf_depth_filenames = tf.train.match_filenames_once(search_depth_files_str)

        return self.tf_image_filenames, self.tf_depth_filenames