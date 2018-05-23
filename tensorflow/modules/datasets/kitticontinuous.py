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
class KittiContinuous(FilenamesHandler):
    def __init__(self, machine):
        super().__init__()
        if machine == 'olorin':
            self.dataset_path = ''
        elif machine == 'xps':
            self.dataset_path = "/media/nicolas/Nícolas/datasets/kitti/raw_data/data/"

        self.name = 'kitticontinuous'

        # self.image_size = Size(375, 1242, 3)
        # self.depth_size = Size(375, 1242, 1)

        print("[Dataloader] KittiContinuous object created.")

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

            timer = -time.time()
            image_filenames = [self.dataset_path + image for image in image_filenames]
            depth_filenames = [self.dataset_path + depth for depth in depth_filenames]
            timer += time.time()
            print(timer)
        else:
            print("[Dataloader] '%s' doesn't exist..." % file)
            print("[Dataloader] Searching files using glob (This may take a while)...")

            # Finds input images and labels inside list of folders.
            image_filenames_tmp = glob.glob(self.dataset_path + "*/proc2/imgs/*.png") + glob.glob(self.dataset_path + "*/proc_kitti/imgs/*.png")
            depth_filenames_tmp = glob.glob(self.dataset_path + "*/proc2/disp2/*.png") + glob.glob(self.dataset_path + "*/proc_kitti/disp2/*.png")

            # print(image_filenames_tmp)
            # print(len(image_filenames_tmp))
            # input("image_filenames_tmp")
            # print(depth_filenames_tmp)
            # print(len(depth_filenames_tmp))
            # input("depth_filenames_tmp")

            image_filenames_aux = [os.path.splitext(os.path.split(image)[1])[0] for image in image_filenames_tmp]
            depth_filenames_aux = [os.path.splitext(os.path.split(depth)[1])[0] for depth in depth_filenames_tmp]

            # print(image_filenames_aux)
            # print(len(image_filenames_aux))
            # input("image_filenames_aux")
            # print(depth_filenames_aux)
            # print(len(depth_filenames_aux))
            # input("depth_filenames_aux")

            n, m = len(image_filenames_aux), len(depth_filenames_aux)

            # Sequential Search. This kind of search ensures that the images are paired!
            print("[Dataloader] Checking if RGB and Depth images are paired... ")

            start = time.time()
            for j, depth in enumerate(depth_filenames_aux):
                print("%d/%d" % (j + 1, m))  # Debug
                for i, image in enumerate(image_filenames_aux):
                    if image == depth:
                        image_filenames.append(image_filenames_tmp[i].replace(self.dataset_path, ''))
                        depth_filenames.append(depth_filenames_tmp[j].replace(self.dataset_path, ''))

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
