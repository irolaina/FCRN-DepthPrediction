# ===========
#  Libraries
# ===========
import numpy as np
import tensorflow as tf
import os
import glob

from .size import Size
from .kitti import Kitti
from .kittiraw import KittiRaw
from .nyudepth import NyuDepth

# ==================
#  Global Variables
# ==================
LOSS_LOG_INITIAL_VALUE = 0.1


# ===========
#  Functions
# ===========


# ===================
#  Class Declaration
# ===================
class Dataloader_new():
    def __init__(self, args):
        # Detects which dataset was selected and creates the 'datasetObj'.
        self.selectedDataset = args.dataset
        # print(selectedDataset)

        if self.selectedDataset == 'kittiraw_residential_continuous':
            datasetObj = KittiRaw() # TODO: Terminar
            pass

        elif self.selectedDataset == 'nyudepth':
            datasetObj = NyuDepth() # TODO: Terminar
            pass

        else:
            print("[Dataset] The typed dataset '%s' is invalid. Check the list of supported datasets." % self.selectedDataset)
            raise SystemExit

        # Collects Dataset Info
        self.dataset_name = datasetObj.name
        self.dataset_path = datasetObj.dataset_path
        self.image_size = datasetObj.image_size
        self.depth_size = datasetObj.depth_size

        print("[Dataset] dataloader object created.")

    # TODO: Ler outros Datasets
    def getTrainInputs(self, args):
        if args.machine == 'olorin':
            # KittiRaw Residential Continuous
            # Image: (375, 1242, 3) uint8
            # Depth: (375, 1242)    uint8
            if args.dataset == 'kittiraw_residential_continuous':
                # TODO: Migrar informações para os handlers de cada dataset
                search_image_files_str = "../../mestrado_code/monodeep/data/residential_continuous/training/imgs/*.png"
                search_depth_files_str = "../../mestrado_code/monodeep/data/residential_continuous/training/dispc/*.png"

                tf_image_filenames = tf.train.match_filenames_once(search_image_files_str)
                tf_depth_filenames = tf.train.match_filenames_once(search_depth_files_str)

        elif args.machine == 'xps':
            # KittiRaw Residential Continuous
            # Image: (375, 1242, 3) uint8
            # Depth: (375, 1242)    uint8
            if args.dataset == 'kittiraw_residential_continuous':
                # TODO: Migrar informações para os handlers de cada dataset
                search_image_files_str = "../../data/residential_continuous/training/imgs/*.png"
                search_depth_files_str = "../../data/residential_continuous/training/dispc/*.png"

                image_filenames = None
                depth_filenames = None

                tf_image_filenames = tf.train.match_filenames_once(search_image_files_str)
                tf_depth_filenames = tf.train.match_filenames_once(search_depth_files_str)

            # NYU Depth v2
            # Image: (480, 640, 3) ?
            # Depth: (480, 640)    ?
            elif args.dataset == 'nyudepth':
                image_filenames = []
                depth_filenames = []

                # TODO: Migrar informações para os handlers de cada dataset
                root_folder = "/media/nicolas/Nícolas/datasets/nyu-depth-v2/images/training/"

                # Finds input images and labels inside list of folders.
                for folder in glob.glob(root_folder + "*/"):
                    print(folder)
                    os.chdir(folder)

                    for file in glob.glob('*_colors.png'):
                        print(file)
                        image_filenames.append(folder + file)

                    for file in glob.glob('*_depth.png'):
                        print(file)
                        depth_filenames.append(folder + file)

                    print()

                print("Summary - Training Inputs")
                print("image_filenames: ", len(image_filenames))
                print("depth_filenames: ", len(depth_filenames))

                tf_image_filenames = tf.placeholder(tf.string)
                tf_depth_filenames = tf.placeholder(tf.string)

        return image_filenames, depth_filenames, tf_image_filenames, tf_depth_filenames


