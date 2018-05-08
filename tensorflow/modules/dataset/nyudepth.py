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
# NYU Depth v2
# TODO: Add info
# Image: (480, 640, 3) ?
# Depth: (480, 640)    ?
class NyuDepth(object):
    def __init__(self, machine):
        if machine == 'olorin':
            self.dataset_path = ''
        elif machine == 'xps':
            self.dataset_path = "/media/nicolas/Nícolas/datasets/nyu-depth-v2/images/"

        self.name = 'nyudepth'

        self.image_size = Size(480, 640, 3)
        self.depth_size = Size(480, 640, 1)

        self.image_replace = [b'_colors.png', b'']
        self.depth_replace = [b'_depth.png', b'']

        # Data Range/Plot ColorSpace # TODO: Terminar
        self.vmin = None
        self.vmax = None
        self.log_vmin = None
        self.log_vmax = None

        print("[Dataloader] NyuDepth object created.")

    def getFilenamesLists(self, mode):
        image_filenames = []
        depth_filenames = []

        file = 'data/' + self.name + '_' + mode + '.txt'

        if mode == 'train':
            if os.path.exists(file):
                data = self.loadList(file)

                # Parsing Data
                image_filenames = data[:, 0]
                depth_filenames = data[:, 1]
            else:
                print("[Dataloader] '%s' doesn't exist..." % file)
                print("[Dataloader] Searching files using glob (This may take a while)...")

                # Finds input images and labels inside list of folders.
                start = time.time()
                for folder in glob.glob(self.dataset_path + "training/*/"):
                    # print(folder)
                    os.chdir(folder)

                    for file in glob.glob('*_colors.png'):
                        # print(file)
                        image_filenames.append(folder + file)

                    for file in glob.glob('*_depth.png'):
                        # print(file)
                        depth_filenames.append(folder + file)

                print("time: %f s" % (time.time() - start))

                self.saveList(image_filenames, depth_filenames, mode)

        elif mode == 'test':
            if os.path.exists(file):
                data = self.loadList(file)

                # Parsing Data
                image_filenames = data[:, 0]
                depth_filenames = data[:, 1]
            else:
                print("[Dataloader] '%s' doesn't exist..." % file)
                print("[Dataloader] Searching files using glob (This may take a while)...")

                # Finds input images and labels inside list of folders.
                for folder in glob.glob(self.dataset_path + "testing/*/"):
                    # print(folder)
                    os.chdir(folder)

                    for file in glob.glob('*_colors.png'):
                        # print(file)
                        image_filenames.append(folder + file)

                    for file in glob.glob('*_depth.png'):
                        # print(file)
                        depth_filenames.append(folder + file)

                self.saveList(image_filenames, depth_filenames, mode)
        else:
            sys.exit()

        # TODO: Fazer shuffle

        # TODO: Eu acho que não precisa mais disso
        # Alphabelly Sort the List of Strings
        image_filenames.sort()
        depth_filenames.sort()

        return image_filenames, depth_filenames

    def loadList(self, filename):
        print("\n[Dataloader] Loading '%s'..." % filename)
        try:
            data = np.genfromtxt(filename, dtype='str', delimiter='\t')
            # print(data.shape)
        except OSError:
            print("[OSError] Could not find the '%s' file." % filename)
            sys.exit()

        return data

    def saveList(self, image_filenames, depth_filenames, mode):
        # Column-Concatenation of Lists of Strings
        filenames = list(zip(image_filenames, depth_filenames))
        filenames = np.array(filenames)

        # Saving the 'filenames' variable to *.txt
        root_path = os.path.abspath(os.path.join(__file__, "../../.."))
        relative_path = 'data/' + self.name + '_' + mode + '.txt'
        save_file_path = os.path.join(root_path, relative_path)

        np.savetxt(save_file_path, filenames, delimiter='\t', fmt='%s')

        print("[Dataset] '%s' file saved." % save_file_path)
