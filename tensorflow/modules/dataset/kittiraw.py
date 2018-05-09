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
# KittiRaw Residential Continuous
# Image: (375, 1242, 3) uint8
# Depth: (375, 1242)    uint8
# FIXME: SEPAREI MANUALMENTE O CONJUNTO DE TESTE E TREINAMENTO, DEVERIA DEIXAR A SEPARAÇÃO ONLINE, COMO FIZ COM O APOLLO, A NÃO SER QUE O VITOR APLICOU O MÉTODO DELE EM IMAGENS QUE JÁ POSSUIAM A SEPARAÇÃO TRAIN/TEST
class KittiRaw(FilenamesHandler):
    def __init__(self, machine):
        super().__init__()
        if machine == 'olorin':
            self.dataset_path = ''
        elif machine == 'xps':
            self.dataset_path = "../../data/residential_continuous/"

        self.name = 'kittiraw'

        self.image_size = Size(375, 1242, 3)
        self.depth_size = Size(375, 1242, 1)

        self.image_replace = [b'/imgs/', b'']
        self.depth_replace = [b'/dispc/', b'']

        # Data Range/Plot ColorSpace
        self.vmin = 0
        self.vmax = 240
        self.log_vmin = np.log(self.vmin + LOG_INITIAL_VALUE)
        self.log_vmax = np.log(self.vmax)

        print("[Dataloader] KittiRaw object created.")

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
            image_filenames_tmp = glob.glob(self.dataset_path + mode + "ing/imgs/*")
            depth_filenames_tmp = glob.glob(self.dataset_path + mode + "ing/dispc/*")

            image_filename_aux = [os.path.split(image)[1] for image in image_filenames_tmp]
            depth_filename_aux = [os.path.split(depth)[1] for depth in depth_filenames_tmp]

            n, m = len(image_filename_aux), len(depth_filename_aux)

            # Sequential Search. This kind of search ensures that the images are paired!
            start = time.time()
            for j, depth in enumerate(depth_filename_aux):
                print("%d/%d" % (j + 1, m))  # Debug
                for i, image in enumerate(image_filename_aux):
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
