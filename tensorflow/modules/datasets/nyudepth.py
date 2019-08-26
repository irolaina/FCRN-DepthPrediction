# ========
#  README
# ========
# NYU Depth v2
# Uses Depth Maps: measures distances [close - LOW values, far - HIGH values]
# Kinect's maxDepth: 0~10m

# Image: (480, 640, 3) uint8
# Depth: (480, 640)    uint16

# -----
# Official Dataset Guidelines
# -----
# According to the NYU's Website the Labeled Dataset:
# images – HxWx3xN matrix of RGB images where H and W are the height and width, respectively, and N is the number of images.
# depths – HxWxN matrix of in-painted depth maps where H and W are the height and width, respectively and N is the number of images. The values of the depth elements are in meters.

# Raw Depth image to Depth (meters):
# depthParam1 = 351.3;
# depthParam2 = 1092.5;
# maxDepth = 10;

# depth_true = depthParam1./(depthParam2 - swapbytes(depth));
# depth_true(depth_true > maxDepth) = maxDepth;
# depth_true(depth_true < 0) = 0;
# ------

# -----
# Dataset Guidelines - Custom
# -----
# 1) Download the 'nyu_depth_v2_labeled.mat' and 'splits.mat' files from NYU Depth Dataset V2 website.
# 2) Uses the 'convert.py' script from https://github.com/deeplearningais/curfil/wiki/Training-and-Prediction-with-the-NYU-Depth-v2-Dataset
#   This script decompresses the information in the *.mat files to generate *.png images.
#   The above script loads the dictionary 'depth', which is given in meters, and multiplies by 1000.0 before dumping it on the PNG format.
# 3) Then, for retrieving the information from *_depth.png (uint16) to meters:
#   depth_true = ((float) depth)/1000.0
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
class NyuDepth(Dataset):
    def __init__(self, **kwargs):
        super(NyuDepth, self).__init__(**kwargs)

    def get_filenames_lists(self, mode, test_split='', test_file_path=''):
        file_path = self.get_file_path(mode, test_split, test_file_path)

        if os.path.exists(file_path):
            image_filenames, depth_filenames = self.read_text_file(file_path, self.dataset_path)
        else:
            print("[Dataloader] '%s' doesn't exist..." % file_path)
            print("[Dataloader] Searching files using glob (This may take a while)...")

            # Finds input images and labels inside the list of folders.
            image_filenames_tmp = glob.glob(self.dataset_path + mode + "ing/*/*_colors.png")  # ...ColorImage/Record*/Camera */*.jpg
            depth_filenames_tmp = glob.glob(self.dataset_path + mode + "ing/*/*_depth.png")  # ...Depth/Record*/Camera */*.png

            image_filenames_aux = [os.path.split(image)[1].replace('_colors.png', '') for image in image_filenames_tmp]
            depth_filenames_aux = [os.path.split(depth)[1].replace('_depth.png', '') for depth in depth_filenames_tmp]

            # TODO: Add Comment
            image_filenames, depth_filenames, _, _ = self.search_pairs(image_filenames_tmp, depth_filenames_tmp,
                                                                       image_filenames_aux, depth_filenames_aux)

            # Debug
            # filenames = list(zip(image_filenames[:10], depth_filenames[:10]))
            # for i in filenames:
            #     print(i)
            # input("enter")

            # TODO: Acredito que dê pra mover a chamada dessa função para fora
            self.save_list(image_filenames, depth_filenames, self.name, mode, self.dataset_path)

        return image_filenames, depth_filenames
