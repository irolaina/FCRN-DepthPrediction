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
        self.image_filenames = []
        self.depth_filenames = []

    @staticmethod
    def read_text_file(filename, dataset_path):
        print("\n[Dataloader] Loading '%s'..." % filename)
        try:
            data = np.genfromtxt(filename, dtype='str', delimiter='\t')
            # print(data.shape)
        except OSError:
            print("[OSError] Could not find the '%s' file." % filename)
            sys.exit()

        # Parsing Data
        image_filenames = list(data[:, 0])
        depth_filenames = list(data[:, 1])

        image_filenames = join_dataset_path(image_filenames, dataset_path)
        depth_filenames = join_dataset_path(depth_filenames, dataset_path)

        return image_filenames, depth_filenames

    @staticmethod
    def search_pairs(image_filenames_tmp, depth_filenames_tmp,
                     image_filenames_aux, depth_filenames_aux):  # TODO: Preciso realmente ter essas duas variaveis? Podem ser unificadas?
        image_filenames = []
        depth_filenames = []

        # print(image_filenames_tmp)
        # print(len(image_filenames_tmp))
        # input("image_filenames_tmp")
        # print(depth_filenames_tmp)
        # print(len(depth_filenames_tmp))
        # input("depth_filenames_tmp")

        # print(image_filenames_aux)
        # print(len(image_filenames_aux))
        # input("image_filenames_aux")
        # print(depth_filenames_aux)
        # print(len(depth_filenames_aux))
        # input("depth_filenames_aux")

        _, m = len(image_filenames_aux), len(depth_filenames_aux)

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

        return image_filenames, depth_filenames, n2, m2

    @staticmethod
    def saveList(image_filenames, depth_filenames, name, mode, dataset_path):
        # TODO: add comemnt
        image_filenames_dump = [image.replace(dataset_path, '') for image in image_filenames]
        depth_filenames_dump = [depth.replace(dataset_path, '') for depth in depth_filenames]

        # Column-Concatenation of Lists of Strings
        filenames = list(zip(image_filenames_dump, depth_filenames_dump))
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
    filenames = [dataset_path + filename for filename in filenames]
    timer += time.time()
    print('time:', timer, 's\n')

    return filenames
