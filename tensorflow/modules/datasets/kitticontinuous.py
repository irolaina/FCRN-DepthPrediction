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
import time

import numpy as np

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
class KittiContinuous(Dataset):
    def __init__(self, *args, **kwargs):
        super(KittiContinuous, self).__init__(*args, **kwargs)


    def getFilenamesLists(self, mode, test_split='', test_file_path=''):
        file = self.get_file_path(mode, test_split, test_file_path)
        ratio = 0.8

        if os.path.exists(file):
            image_filenames, depth_filenames = self.read_text_file(file, self.dataset_path)
        else:
            # TODO: Acredito que dê pra remover as variaveis abaixo
            image_filenames = []
            depth_filenames = []

            print("[Dataloader] '%s' doesn't exist..." % file)
            print("[Dataloader] Searching files using glob (This may take a while)...")

            # Finds input images and labels inside list of folders.
            image_filenames_tmp, depth_filenames_tmp = [], []

            try:
                selected_category = self.name.split('_')[1]
                scenes = np.genfromtxt("data/kitti_scenes/" + selected_category + ".txt", dtype='str', delimiter='\t')

                print()
                print(scenes)
                print(len(scenes))
                print()

                for scene in scenes:
                    print(scene)
                    # print(scene.split("_drive")[0])
                    print(self.dataset_path + scene.split("_drive")[0] + "/" + scene + "/proc_kitti_nick/imgs/*.png")
                    image_filenames_tmp += glob.glob(
                        self.dataset_path + scene.split("_drive")[0] + "/" + scene + "/proc_kitti_nick/imgs/*.png")
                    depth_filenames_tmp += glob.glob(
                        self.dataset_path + scene.split("_drive")[0] + "/" + scene + "/proc_kitti_nick/disp2/*.png")

                    print(len(image_filenames_tmp))
                    # print(len(depth_filenames_tmp))

            except IndexError:
                image_filenames_tmp = glob.glob(self.dataset_path + "2011_*/*/proc_kitti_nick/imgs/*.png")
                depth_filenames_tmp = glob.glob(self.dataset_path + "2011_*/*/proc_kitti_nick/disp2/*.png")

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

            # TODO: Acredito que dê pra mover a chamada dessa função para fora
            self.saveList(image_filenames, depth_filenames, self.name, mode, self.dataset_path)

        return image_filenames, depth_filenames, file
