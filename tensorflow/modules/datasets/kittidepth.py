# ========
#  README
# ========
# Kitti Depth Prediction
# Uses Depth Maps: measures distances [close - LOW values, far - HIGH values]
# Image: (375, 1242, 3) uint8
# Depth: (375, 1242)    uint16

# -----
# Dataset Guidelines
# -----
# Raw Depth image to Depth (meters):
# depth(u,v) = ((float)I(u,v))/256.0;
# valid(u,v) = I(u,v)>0;
# -----

# FIXME: Este dataset possui conjunto de validação. O conjunto de tests não possui ground truth. Criar lista de validação

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
class KittiDepth(FilenamesHandler):
    def __init__(self, machine):
        super().__init__()
        if machine == 'olorin':
            self.dataset_path = ''
        elif machine == 'xps':
            self.dataset_path = "/media/nicolas/Nícolas/datasets/kitti/"

        self.name = 'kittidepth'

        self.image_size = Size(375, 1242, 3)
        self.depth_size = Size(375, 1242, 1)

        print("[Dataloader] KittiDepth object created.")

    # FIXME: PAREI Aqui
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

            # Workaround # FIXME: Temporary
            if mode == 'test':
                mode = 'val'

            # Finds input images and labels inside list of folders.
            image_filenames_tmp = glob.glob(self.dataset_path + 'raw_data/data/*/*/image_0*/data/*.png')
            depth_filenames_tmp = glob.glob(self.dataset_path + 'depth/depth_prediction/data/' + mode + '/*/proj_depth/groundtruth/image_0*/*.png')

            # TODO: Remover
            # print(image_filenames_tmp)
            # print(len(image_filenames_tmp))
            # input("oi")
            # print(depth_filenames_tmp)
            # print(len(depth_filenames_tmp))
            # input("oi2")

            image_filenames_aux = [image.replace(self.dataset_path, '').split(os.sep) for image in image_filenames_tmp]
            depth_filenames_aux = [depth.replace(self.dataset_path, '').split(os.sep) for depth in depth_filenames_tmp]

            image_idx = [3, 4, 6]
            depth_idx = [4, 7, 8]

            image_filenames_aux = ['/'.join([image[i] for i in image_idx]) for image in image_filenames_aux]
            depth_filenames_aux = ['/'.join([depth[i] for i in depth_idx]) for depth in depth_filenames_aux]

            # TODO: Remover
            # print(image_filenames_aux)
            # print(len(image_filenames_aux))
            # input("oi3")
            # print(depth_filenames_aux)
            # print(len(depth_filenames_aux))
            # input("oi4")

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

        return image_filenames, depth_filenames
