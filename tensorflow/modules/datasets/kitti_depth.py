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

from .dataset import Dataset


# ===================
#  Class Declaration
# ===================
class KittiDepth(Dataset):
    def __init__(self, *args, **kwargs):
        super(KittiDepth, self).__init__(*args, **kwargs)

    def getFilenamesLists(self, mode, test_split='', test_file_path=''):
        # Workaround # FIXME: Temporary
        if mode == 'test':
            mode = 'val'

        file = self.get_file_path(mode, test_split, test_file_path)

        if os.path.exists(file):
            image_filenames, depth_filenames = self.read_text_file(file, self.dataset_path)
        else:
            print("[Dataloader] '%s' doesn't exist..." % file)
            print("[Dataloader] Searching files using glob (This may take a while)...")

            # Finds input images and labels inside the list of folders.
            image_filenames_tmp = glob.glob(self.dataset_path + 'raw_data/2011_*/*/image_02/data/*.png') + glob.glob(self.dataset_path + 'raw_data/2011_*/*/image_03/data/*.png')
            depth_filenames_tmp = glob.glob(self.dataset_path + 'depth/depth_prediction/data/' + mode + '/*/proj_depth/groundtruth/image_02/*.png') + glob.glob(self.dataset_path + 'depth/depth_prediction/data/' + mode + '/*/proj_depth/groundtruth/image_03/*.png')

            image_filenames_aux = [image.replace(self.dataset_path, '').split(os.sep) for image in image_filenames_tmp]
            depth_filenames_aux = [depth.replace(self.dataset_path, '').split(os.sep) for depth in depth_filenames_tmp]

            image_idx = [2, 3, 5]
            depth_idx = [4, 7, 8]

            image_filenames_aux = ['/'.join([image[i] for i in image_idx]) for image in image_filenames_aux]
            depth_filenames_aux = ['/'.join([depth[i] for i in depth_idx]) for depth in depth_filenames_aux]

            # TODO: Add Comment
            image_filenames, depth_filenames, _, _ = self.search_pairs(image_filenames_tmp, depth_filenames_tmp,
                                                                       image_filenames_aux, depth_filenames_aux)

            # Debug
            # filenames = list(zip(image_filenames[:10], depth_filenames[:10]))
            # for i in filenames:
            #     print(i)
            # input("enter")

            # TODO: Acredito que dê pra mover a chamada dessa função para fora
            self.saveList(image_filenames, depth_filenames, self.name, mode, self.dataset_path)

        return image_filenames, depth_filenames, file
