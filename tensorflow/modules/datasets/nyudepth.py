# ========
#  README
# ========
# NYU Depth v2
# Uses Depth Maps: measures distances [close - LOW values, far - HIGH values]
# Kinect's maxDepth: 0~10m

# Image: (480, 640, 3) uint8
# Depth: (480, 640)    uint16

# -----
# Official Dataset Guidelines
# -----
# According to the NYU's Website the Labeled Dataset:
# images – HxWx3xN matrix of RGB images where H and W are the height and width, respectively, and N is the number of images.
# depths – HxWxN matrix of in-painted depth maps where H and W are the height and width, respectively and N is the number of images. The values of the depth elements are in meters.

# Raw Depth image to Depth (meters):
# depthParam1 = 351.3;
# depthParam2 = 1092.5;
# maxDepth = 10;

# depth_true = depthParam1./(depthParam2 - swapbytes(depth));
# depth_true(depth_true > maxDepth) = maxDepth;
# depth_true(depth_true < 0) = 0;
# ------

# -----
# Dataset Guidelines - Custom
# -----
# 1) Download the 'nyu_depth_v2_labeled.mat' and 'splits.mat' files from NYU Depth Dataset V2 website.
# 2) Uses the 'convert.py' script from https://github.com/deeplearningais/curfil/wiki/Training-and-Prediction-with-the-NYU-Depth-v2-Dataset
#   This script decompresses the information in the *.mat files to generate *.png images.
#   The above script loads the dictionary 'depth', which is given in meters, and multiplies by 1000.0 before dumping it on the PNG format.
# 3) Then, for retrieving the information from *_depth.png (uint16) to meters:
#   depth_true = ((float) depth)/1000.0
# -----


# ===========
#  Libraries
# ===========
import glob
import os
import time

import numpy as np

from ..filenames import FilenamesHandler
from .dataset import Dataset
from ..filenames import join_dataset_path


# ==================
#  Global Variables
# ==================


# ===========
#  Functions
# ===========


# ===================
#  Class Declaration
# ===================
class NyuDepth(Dataset, FilenamesHandler):
    def __init__(self, *args, **kwargs):
        super(NyuDepth, self).__init__(*args, **kwargs)

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

            image_filenames = join_dataset_path(image_filenames, self.dataset_path)
            depth_filenames = join_dataset_path(depth_filenames, self.dataset_path)
        else:
            print("[Dataloader] '%s' doesn't exist..." % file)
            print("[Dataloader] Searching files using glob (This may take a while)...")

            # Finds input images and labels inside list of folders.
            image_filenames_tmp = []
            depth_filenames_tmp = []

            image_filenames_aux = []
            depth_filenames_aux = []
            for folder in glob.glob(self.dataset_path + mode + "ing/*/"):
                # print(folder)
                os.chdir(folder)

                for image in glob.glob('*_colors.png'):
                    # print(file)
                    image_filenames_tmp.append(folder + image)
                    image_filenames_aux.append(os.path.split(image)[1].replace('_colors.png', ''))

                for depth in glob.glob('*_depth.png'):
                    # print(file)
                    depth_filenames_tmp.append(folder + depth)
                    depth_filenames_aux.append(os.path.split(depth)[1].replace('_depth.png', ''))

            n, m = len(image_filenames_aux), len(depth_filenames_aux)

            # Sequential Search. This kind of search ensures that the images are paired!
            print("[Dataloader] Checking if RGB and Depth images are paired... ")

            start = time.time()
            for j, depth in enumerate(depth_filenames_aux):
                print("%d/%d" % (j + 1, m))  # Debug
                for i, image in enumerate(image_filenames_aux):
                    if image == depth:
                        image_filenames.append(image_filenames_tmp[i])
                        depth_filenames.append(depth_filenames_tmp[j])

            n2, m2 = len(image_filenames), len(depth_filenames)
            if not n2 == m2:
                print("[AssertionError] Length must be equal!")
                raise AssertionError()
            print("time: %f s" % (time.time() - start))

            # Shuffles
            s = np.random.choice(n2, n2, replace=False)
            image_filenames = list(np.array(image_filenames)[s])
            depth_filenames = list(np.array(depth_filenames)[s])

            # Debug
            # filenames = list(zip(image_filenames[:10], depth_filenames[:10]))
            # for i in filenames:
            #     print(i)
            # input("enter")

            image_filenames_dump = [image.replace(self.dataset_path, '') for image in image_filenames]
            depth_filenames_dump = [depth.replace(self.dataset_path, '') for depth in depth_filenames]

            self.saveList(image_filenames_dump, depth_filenames_dump, self.name, mode)

        return image_filenames, depth_filenames
