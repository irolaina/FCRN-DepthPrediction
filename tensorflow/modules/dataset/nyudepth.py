# ========
#  README
# ========
# NYU Depth v2
# Image: (480, 640, 3) uint8
# Depth: (480, 640)    uint16
# Kinect maxDepth: 10m
# TODO: Encontrar dataset guidelines para extrair o valor de depth da imagem uint16 para este dataset

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
class NyuDepth(FilenamesHandler):
    def __init__(self, machine):
        super().__init__()
        if machine == 'olorin':
            self.dataset_path = ''
        elif machine == 'xps':
            self.dataset_path = "/media/nicolas/Nícolas/datasets/nyu-depth-v2/images/"

        self.name = 'nyudepth'

        self.image_size = Size(480, 640, 3)
        self.depth_size = Size(480, 640, 1)

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
        else:
            print("[Dataloader] '%s' doesn't exist..." % file)
            print("[Dataloader] Searching files using glob (This may take a while)...")

            # Finds input images and labels inside list of folders.
            start = time.time()
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

            self.saveList(image_filenames, depth_filenames, self.name, mode)

        return image_filenames, depth_filenames
