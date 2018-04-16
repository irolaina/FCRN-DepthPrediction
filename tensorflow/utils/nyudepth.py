# ===========
#  Libraries
# ===========
import glob
import os
import tensorflow as tf

from .size import Size


# ===================
#  Class Declaration
# ===================
# NYU Depth v2
# Image: (480, 640, 3) ?
# Depth: (480, 640)    ?
class NyuDepth(object):
    def __init__(self):
        self.dataset_path = "/media/nicolas/Nícolas/datasets/nyu-depth-v2/images/training/"
        self.name = 'nyudepth'

        self.image_size = Size(480, 640, 3)
        self.depth_size = Size(480, 640, 1)

        self.image_filenames = []
        self.depth_filenames = []

        self.image_replace = [b'_colors.png', b'']
        self.depth_replace = [b'_depth.png', b'']

        # Data Range/Plot ColorSpace
        self.vmin = None
        self.vmax = None
        self.log_vmin = None
        self.log_vmax = None

        print("[Dataloader] NyuDepth object created.")

    def getFilenamesLists(self):
        # Finds input images and labels inside list of folders.
        for folder in glob.glob(self.dataset_path + "*/"):
            # print(folder)
            os.chdir(folder)

            for file in glob.glob('*_colors.png'):
                # print(file)
                self.image_filenames.append(folder + file)

            for file in glob.glob('*_depth.png'):
                # print(file)
                self.depth_filenames.append(folder + file)

            # print()

        print("\nSummary")
        print("image_filenames: ", len(self.image_filenames))
        print("depth_filenames: ", len(self.depth_filenames))

        return self.image_filenames, self.depth_filenames

    def getFilenamesTensors(self):
        self.tf_image_filenames = tf.constant(self.image_filenames)
        self.tf_depth_filenames = tf.constant(self.depth_filenames)

        return self.tf_image_filenames, self.tf_depth_filenames