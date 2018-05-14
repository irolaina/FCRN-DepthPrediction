# ========
#  README
# ========
# Apollo Scape
# Uses Depth Maps: measures distances [close - LOW values, far - HIGH values] in meters
# Image: (2710, 3384, 3) uint8
# Depth: (2710, 3384)    uint16

# Dataset Guidelines
# Depth image format:
# In the depth image, the depth value is save as unsigned short int format. It can be easily read in OpenCV as:
#   cv::Mat depth_u16 = cv::imread ( depth_path, CV_LOAD_IMAGE_ANYDEPTH);
#
# The absolute depth value in meter can be obtained as:
#   double depth_value = depth_u16.at(row, col) / 200.00;

# FIXME: Algumas imagens de profundidade parecem estar incorretas. Ex: apolloscapes_test.txt linhas: 46 73 78

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
class Apolloscape(FilenamesHandler):
    def __init__(self, machine):
        super().__init__()
        if machine == 'olorin':
            self.dataset_path = ''
        elif machine == 'xps':
            self.dataset_path = "/media/nicolas/Nícolas/datasets/apolloscape/data/"

        self.name = 'apolloscape'

        self.image_size = Size(2710, 3384, 3)
        self.depth_size = Size(2710, 3384, 1)

        print("[Dataloader] Apolloscape object created.")

    def getFilenamesLists(self, mode):
        image_filenames = []
        depth_filenames = []

        file = 'data/' + self.name + '_' + mode + '.txt'
        ratio = 0.8

        if os.path.exists(file):
            data = self.loadList(file)

            # Parsing Data
            image_filenames = list(data[:, 0])
            depth_filenames = list(data[:, 1])
        else:
            print("[Dataloader] '%s' doesn't exist..." % file)
            print("[Dataloader] Searching files using glob (This may take a while)...")

            # Finds input images and labels inside list of folders.
            image_filenames_tmp = glob.glob(self.dataset_path + "ColorImage/*/*/*")  # ...ColorImage/Record*/Camera */*.jpg
            depth_filenames_tmp = glob.glob(self.dataset_path + "Depth/*/*/*")  # ...Depth/Record*/Camera */*.png

            image_filenames_aux = [os.path.splitext(os.path.split(image)[1])[0] for image in image_filenames_tmp]
            depth_filenames_aux = [os.path.splitext(os.path.split(depth)[1])[0] for depth in depth_filenames_tmp]

            n, m = len(image_filenames_aux), len(depth_filenames_aux)

            # Sequential Search. This kind of search ensures that the images are paired!
            start = time.time()
            for j, depth in enumerate(depth_filenames_aux):
                print("%d/%d" % (j + 1, m))  # Debug
                for i, image in enumerate(image_filenames_aux):
                    if image == depth:
                        image_filenames.append(image_filenames_tmp[i])
                        depth_filenames.append(depth_filenames_tmp[j])

            n2, m2 = len(image_filenames), len(depth_filenames)
            assert (n2 == m2), "Houston we've got a problem."  # Length must be equal!
            print("time: %f s" % (time.time() - start))

            # Shuffles
            s = np.random.choice(n2, n2, replace=False)
            image_filenames = list(np.array(image_filenames)[s])
            depth_filenames = list(np.array(depth_filenames)[s])

            # Splits Train/Test Subsets
            divider = int(n2 * ratio)

            if mode == 'train':
                image_filenames = image_filenames[:divider]
                depth_filenames = depth_filenames[:divider]
            elif mode == 'test':
                # Defines Testing Subset
                image_filenames = image_filenames[divider:]
                depth_filenames = depth_filenames[divider:]

            n3, m3 = len(image_filenames), len(depth_filenames)

            print('%s_image_set: %d/%d' % (mode, n3, n2))
            print('%s_depth_set: %d/%d' % (mode, m3, m2))

            # Debug
            # filenames = list(zip(image_filenames[:10], depth_filenames[:10]))
            # for i in filenames:
            #     print(i)
            # input("enter")

            self.saveList(image_filenames, depth_filenames, self.name, mode)

        return image_filenames, depth_filenames
