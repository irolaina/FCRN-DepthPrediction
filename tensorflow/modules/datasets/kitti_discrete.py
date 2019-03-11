# ========
#  README
# ========
# KittiDiscrete
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

from .dataset import Dataset


# ===================
#  Class Declaration
# ===================
class KittiDiscrete(Dataset):
    def __init__(self, **kwargs):
        super(KittiDiscrete, self).__init__(**kwargs)

    def get_filenames_lists(self, mode, test_split='', test_file_path=''):
        file_path = self.get_file_path(mode, test_split, test_file_path)

        if os.path.exists(file_path):
            image_filenames, depth_filenames = self.read_text_file(file_path, self.dataset_path)
        else:
            # TODO: Este trecho deverá estar de acordo com o que foi desenvolvido em generate_new_split_kitti_discrete.py
            raise SystemError

            print("[Dataloader] '%s' doesn't exist..." % file_path)
            print("[Dataloader] Searching files using glob (This may take a while)...")

            # Finds input images and labels inside the list of folders.
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
                        self.dataset_path + scene.split("_drive")[0] + "/" + scene + "/proc_kitti_nick/disp1/*.png")

                    print(len(image_filenames_tmp))
                    # print(len(depth_filenames_tmp))

            except IndexError:
                image_filenames_tmp = glob.glob(self.dataset_path + "2011_*/*/proc_kitti_nick/imgs/*.png")
                depth_filenames_tmp = glob.glob(self.dataset_path + "2011_*/*/proc_kitti_nick/disp1/*.png")

            image_filenames_aux = [os.path.splitext(os.path.split(image)[1])[0] for image in image_filenames_tmp]
            depth_filenames_aux = [os.path.splitext(os.path.split(depth)[1])[0] for depth in depth_filenames_tmp]

            # TODO: Add Comment
            image_filenames, depth_filenames, n2, m2 = self.search_pairs(image_filenames_tmp, depth_filenames_tmp,
                                                                         image_filenames_aux, depth_filenames_aux)

            # Splits Train/Test Subsets
            divider = int(n2 * self.ratio)

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
            self.save_list(image_filenames, depth_filenames, self.name, mode, self.dataset_path)

        # Debug
        # print(image_filenames[0], depth_filenames[0])
        # input("enter")

        return image_filenames, depth_filenames
