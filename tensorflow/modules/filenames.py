# ===========
#  Libraries
# ===========
import os
import sys
import time

import numpy as np


# ==================
#  Global Variables
# ==================


# ===========
#  Functions
# ===========


# ===================
#  Class Declaration
# ===================
class FilenamesHandler(object):
    def __init__(self):
        pass

    @staticmethod
    def loadList(filename):
        print("\n[Dataloader] Loading '%s'..." % filename)
        try:
            data = np.genfromtxt(filename, dtype='str', delimiter='\t')
            # print(data.shape)
        except OSError:
            print("[OSError] Could not find the '%s' file." % filename)
            sys.exit()

        return data

    @staticmethod
    def saveList(image_filenames, depth_filenames, name, mode):
        # Column-Concatenation of Lists of Strings
        filenames = list(zip(image_filenames, depth_filenames))
        filenames = np.array(filenames)

        # Saving the 'filenames' variable to *.txt
        root_path = os.path.abspath(os.path.join(__file__, "../.."))  # This line may cause 'FileNotFindError'
        relative_path = 'data/' + name + '_' + mode + '.txt'
        save_file_path = os.path.join(root_path, relative_path)

        # noinspection PyTypeChecker
        np.savetxt(save_file_path, filenames, fmt='%s', delimiter='\t')

        print("\n[Dataset] '%s' file saved." % save_file_path)

def join_dataset_path(filenames, dataset_path):
    timer = -time.time()
    filenames = [dataset_path + depth for depth in filenames]
    timer += time.time()
    print('time:', timer, 's\n')

    return filenames