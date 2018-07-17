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


# ===========
#  Libraries
# ===========
import glob
import os
import time

import numpy as np

from ..filenames import FilenamesHandler
from .dataset import Dataset


# ==================
#  Global Variables
# ==================


# ===========
#  Functions
# ===========


# ===================
#  Class Declaration
# ===================
class KittiDepth(Dataset, FilenamesHandler):
    def __init__(self, *args, **kwargs):
        super(KittiDepth, self).__init__(*args, **kwargs)

        print("[Dataloader] KittiDepth object created.")

    def getFilenamesLists(self, mode):
        image_filenames = []
        depth_filenames = []

        # Workaround # FIXME: Temporary
        if mode == 'test':
            mode = 'val'

        file = 'data/' + self.name + '_' + mode + '.txt'

        if os.path.exists(file):
            image_filenames, depth_filenames = self.loadInputList(file, self.dataset_path)
        else:
            print("[Dataloader] '%s' doesn't exist..." % file)
            print("[Dataloader] Searching files using glob (This may take a while)...")

            # Finds input images and labels inside list of folders.
            image_filenames_tmp = glob.glob(self.dataset_path + 'raw_data/2011_*/*/image_02/data/*.png') + glob.glob(self.dataset_path + 'raw_data/2011_*/*/image_03/data/*.png')
            depth_filenames_tmp = glob.glob(self.dataset_path + 'depth/depth_prediction/data/' + mode + '/*/proj_depth/groundtruth/image_02/*.png') + glob.glob(self.dataset_path + 'depth/depth_prediction/data/' + mode + '/*/proj_depth/groundtruth/image_03/*.png')

            # print(image_filenames_tmp)
            # print(len(image_filenames_tmp))
            # input("image_filenames_tmp")
            # print(depth_filenames_tmp)
            # print(len(depth_filenames_tmp))
            # input("depth_filenames_tmp")

            image_filenames_aux = [image.replace(self.dataset_path, '').split(os.sep) for image in image_filenames_tmp]
            depth_filenames_aux = [depth.replace(self.dataset_path, '').split(os.sep) for depth in depth_filenames_tmp]

            image_idx = [2, 3, 5]
            depth_idx = [4, 7, 8]

            image_filenames_aux = ['/'.join([image[i] for i in image_idx]) for image in image_filenames_aux]
            depth_filenames_aux = ['/'.join([depth[i] for i in depth_idx]) for depth in depth_filenames_aux]

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
                        image_filenames.append(image_filenames_tmp[i])
                        depth_filenames.append(depth_filenames_tmp[j])

            n2, m2 = len(image_filenames), len(depth_filenames)

            n2 = 1
            m2 = 3

            if not n2 == m2:
                print("[AssertionError] Length must be equal!")
                raise AssertionError()

            input("aki")

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
