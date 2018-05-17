# FIXME: Refatorar
# ========
#  README
# ========
# Kitti Stereo 2012
# Uses Disparity Maps: measures pixels displacements [close - HIGH values, far - LOW values]
# Image: (370, 1226, 3) uint8
# Depth: (370, 1226)    uint16

# Dataset Guidelines
# disp(u,v)  = ((float)I(u,v))/256.0;
# valid(u,v) = I(u,v)>0;

# depth_png = np.array(Image.open(filename), dtype=int)
# assert(np.max(depth_png) > 255)
# depth = depth_png.astype(np.float) / 256.
# depth[depth_png == 0] = -1.

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
class Kitti2012(FilenamesHandler):
    def __init__(self, machine):
        super().__init__()
        if machine == 'olorin':
            self.dataset_path = ''
        elif machine == 'xps':
            self.dataset_path = "/media/nicolas/Nícolas/datasets/kitti/stereo/stereo2012/data_stereo_flow/"

        self.name = 'kitti2012'

        self.image_size = Size(370, 1226, 3)
        self.depth_size = Size(370, 1226, 1)

        print("[Dataloader] Kitti2012 object created.")

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
            image_filenames_tmp = glob.glob(self.dataset_path + mode + "ing/colored_0/*")
            depth_filenames_tmp = glob.glob(self.dataset_path + mode + "ing/disp_occ/*")

            image_filenames_aux = [os.path.split(image)[1] for image in image_filenames_tmp]
            depth_filenames_aux = [os.path.split(depth)[1] for depth in depth_filenames_tmp]

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
            assert (n2 == m2), "Houston we've got a problem."  # Length must be equal!
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

            self.saveList(image_filenames, depth_filenames, self.name, mode)

        input("terminar")

        return image_filenames, depth_filenames
