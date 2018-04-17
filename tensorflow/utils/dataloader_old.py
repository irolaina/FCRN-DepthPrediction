# ===========
#  Libraries
# ===========
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import sys
import glob

from scipy import misc as scp
from skimage import exposure
from skimage import dtype_limits
from skimage import transform

from utils.kitti import Kitti
from utils.nyudepth import NyuDepth


# ==================
#  Global Variables
# ==================
# DATASET_PATH_ROOT = '/media/nicolas/Documentos/workspace/datasets'
# DATASET_PATH_ROOT = '/media/olorin/Documentos/datasets'


# ===========
#  Functions
# ===========
def string_length_tf(t):
    return tf.py_func(len, [t], [tf.int64])


def checkArgumentsIntegrity(dataset):
    print("[fcrn/Dataloader] Selected Dataset:", dataset)

    try:
        if dataset != 'nyudepth' and dataset[0:5] != 'kitti':
            raise ValueError

    except ValueError as e:
        print(e)
        print("[Error] ValueError: '", dataset,
              "' is not a valid name! Please select one of the following datasets: "
              "'kitti<dataset_identification>' or 'nyudepth'", sep='')
        print("e.g: python3 ", os.path.splitext(sys.argv[0])[0], ".py -s kitti2012", sep='')
        sys.exit()


def selectedDataset(DATASET_PATH_ROOT, dataset):
    dataset_path = None

    if dataset[0:5] == 'kitti':  # If the first five letters are equal to 'kitti'
        if dataset == 'kitti2012':
            dataset_path = DATASET_PATH_ROOT + 'kitti/data_stereo_flow'

        elif dataset == 'kitti2015':
            dataset_path = DATASET_PATH_ROOT + 'kitti/data_scene_flow'

        elif dataset == 'kittiraw_city':
            dataset_path = DATASET_PATH_ROOT + 'nicolas_kitti/dataset1/city/2011_09_29_drive_0071'

        elif dataset == 'kittiraw_road':
            dataset_path = DATASET_PATH_ROOT + 'nicolas_kitti/dataset1/road/2011_10_03_drive_0042'

        elif dataset == 'kittiraw_residential':
            dataset_path = DATASET_PATH_ROOT + 'nicolas_kitti/dataset1/residential/2011_09_30_drive_0028'

        elif dataset == 'kittiraw_campus':
            dataset_path = DATASET_PATH_ROOT + 'nicolas_kitti/dataset1/campus/2011_09_28_drive_0039'

        elif dataset == 'kittiraw_residential_continuous':
            # dataset_path = DATASET_PATH_ROOT + 'nicolas_kitti/dataset1/residential_continuous'                    # Load data from Olorin's HD
            # dataset_path = '/home/olorin/Documents/nicolas/mestrado_code/monodeep/data/residential_continuous'    # Load data from Olorin's SSD
            # dataset_path = '/home/nicolas/remote/olorin_ssd_nicolas/data/residential_continuous'                  # Load data from Olorin's SSD to run on XPS
            dataset_path = '/home/nicolas/Downloads/workspace/nicolas/data/residential_continuous'                  # Load data from XPS's HD

        # print(dataset_path)
        # input()
        kitti = Kitti(dataset, dataset_path)

        return kitti, dataset_path

    elif dataset == 'nyudepth':
        dataset_path = DATASET_PATH_ROOT + '/nyu-depth-v2/images'

        nyudepth = NyuDepth(dataset, dataset_path)

        return nyudepth, dataset_path


def getListFolders(path):
    return next(os.walk(path))[1]


def removeUnusedFolders(test_folders, train_folders, datasetObj, kittiOcclusion=True):
    # According to Kitti Description, occluded == All Pixels

    print("[fcrn/Dataloader] Removing unused folders for Kitti datasets...")
    unused_test_folders_idx = []
    unused_train_folders_idx = []

    if datasetObj.name == 'kitti2012':
        unused_test_folders_idx = [0, 3, 4]

        if kittiOcclusion:
            unused_train_folders_idx = [0, 2, 3, 5, 6, 7, 8, 9, 10, 11]
        else:
            unused_train_folders_idx = [0, 2, 4, 5, 6, 7, 8, 9, 10, 11]

    if datasetObj.name == 'kitti2015':
        unused_test_folders_idx = []
        if kittiOcclusion:
            unused_train_folders_idx = [0, 1, 3, 4, 5, 7, 8, 9, 10]
        else:
            unused_train_folders_idx = [1, 2, 3, 4, 5, 7, 8, 9, 10]

    if datasetObj.name[0:8] == 'kittiraw':
        unused_test_folders_idx = []
        unused_train_folders_idx = []

    test_folders = np.delete(test_folders, unused_test_folders_idx).tolist()
    train_folders = np.delete(train_folders, unused_train_folders_idx).tolist()

    # Debug
    print(test_folders)
    print(train_folders)
    print()

    return test_folders, train_folders


def getListTestFiles(folders, datasetObj):
    colors, depth = [], []

    for i in range(len(folders)):
        if datasetObj.name == 'nyudepth':
            colors = colors + glob.glob(os.path.join(datasetObj.path, 'testing', folders[i], '*_colors.png'))
            depth = depth + glob.glob(os.path.join(datasetObj.path, 'testing', folders[i], '*_depth.png'))

        elif datasetObj.name == 'kitti2012' or datasetObj.name == 'kitti2015':
            colors = colors + glob.glob(os.path.join(datasetObj.path, 'testing', folders[i], '*.png'))
            depth = []

        elif datasetObj.name[0:8] == 'kittiraw':
            if i == 1:
                colors = colors + glob.glob(os.path.join(datasetObj.path, 'testing', folders[i], '*.png'))
            if i == 0:
                depth = colors + glob.glob(os.path.join(datasetObj.path, 'testing', folders[0], '*.png'))

    # Debug
    # print("Testing")
    # print("colors:", colors)
    # print(len(colors))
    # print("depth:", depth)
    # print(len(depth))
    # print()

    return colors, depth


def getListTrainFiles(folders, datasetObj):
    colors, depth = [], []

    for i in range(len(folders)):
        if datasetObj.name == 'nyudepth':
            colors = colors + glob.glob(os.path.join(datasetObj.path, 'training', folders[i], '*_colors.png'))
            depth = depth + glob.glob(os.path.join(datasetObj.path, 'training', folders[i], '*_depth.png'))

        elif datasetObj.name == 'kitti2012':
            if i == 0:
                colors = colors + glob.glob(os.path.join(datasetObj.path, 'training', folders[i], '*.png'))
            if i == 1:
                depth = depth + glob.glob(os.path.join(datasetObj.path, 'training', folders[i], '*.png'))

        elif datasetObj.name == 'kitti2015':
            if i == 1:
                colors = colors + glob.glob(os.path.join(datasetObj.path, 'training', folders[i], '*.png'))
            if i == 0:
                depth = depth + glob.glob(os.path.join(datasetObj.path, 'training', folders[i], '*.png'))

        elif datasetObj.name[0:8] == 'kittiraw':
            if i == 1:
                colors = colors + glob.glob(os.path.join(datasetObj.path, 'training', folders[i], '*.png'))
            if i == 0:
                depth = depth + glob.glob(os.path.join(datasetObj.path, 'training', folders[i], '*.png'))

    # Debug
    # print("Training")
    # print("colors:", colors)
    # print(len(colors))
    # print("depth:", depth)
    # print(len(depth))
    # print()

    return colors, depth


def getFilesFilename(file_path_list):
    filename_list = []

    for i in range(0, len(file_path_list)):
        filename_list.append(os.path.split(file_path_list[i])[1])

    # print(filename_list)
    # print(len(filename_list))

    return filename_list


def getValidPairFiles(colors_filename, depth_filename, datasetObj):
    valid_pairs_idx = []

    colors_filename_short = []
    depth_filename_short = []

    if datasetObj.name == 'nyudepth':
        for i in range(len(colors_filename)):
            # print(colors_filename[i])
            # print(depth_filename[i])
            # print(colors_filename[i][:-11])
            # print(depth_filename[i][:-10])

            for k in range(len(colors_filename)):
                colors_filename_short.append(colors_filename[k][:-11])

            for l in range(len(depth_filename)):
                depth_filename_short.append(depth_filename[l][:-10])

            if colors_filename_short[i] in depth_filename_short:
                j = depth_filename_short.index(colors_filename_short[i])
                # print(i,j)
                valid_pairs_idx.append([i, j])

    if datasetObj.name[0:5] == 'kitti':
        for i in range(len(colors_filename)):
            if colors_filename[i] in depth_filename:
                j = depth_filename.index(colors_filename[i])

                valid_pairs_idx.append([i, j])

    # Debug
    # print("valid_pairs_idx:", valid_pairs_idx)
    # print(len(valid_pairs_idx))

    return valid_pairs_idx


def cropImage(img, x_min=None, x_max=None, y_min=None, y_max=None, size=None):
    try:
        if size is None:
            raise ValueError
    except ValueError:
        print("[ValueError] Oops! Empty cropSize list. Please sets the desired cropSize.\n")

    if len(img.shape) == 3:
        lx, ly, _ = img.shape
    else:
        lx, ly = img.shape

    # Debug
    # print("img.shape:", img.shape)
    # print("lx:",lx,"ly:",ly)

    if (x_min is None) and (x_max is None) and (y_min is None) and (y_max is None):
        # Crop
        # (y_min,x_min)----------(y_max,x_min)
        #       |                      |
        #       |                      |
        #       |                      |
        # (y_min,x_max)----------(y_max,x_max)
        x_min = round((lx - size[0]) / 2)
        x_max = round((lx + size[0]) / 2)
        y_min = round((ly - size[1]) / 2)
        y_max = round((ly + size[1]) / 2)

        crop = img[x_min: x_max, y_min: y_max]

        # Debug
        # print("x_min:",x_min,"x_max:", x_max, "y_min:",y_min,"y_max:", y_max)
        # print("crop.shape:",crop.shape)

        # TODO: Draw cropping Rectangle

    else:
        crop = img[x_min: x_max, y_min: y_max]

    return crop


def np_resizeImage(img, size):
    try:
        if size is None:
            raise ValueError
    except ValueError:
        print("[ValueError] Oops! Empty resizeSize list. Please sets the desired resizeSize.\n")

    # TODO: Qual devo usar?
    # resized = scp.imresize(img, size, interp='bilinear')  # Attention! This method doesn't maintain the original depth range!!!
    # resized = transform.resize(image=img,output_shape=size, preserve_range=True, order=0)  # 0: Nearest - neighbor
    resized = transform.resize(image=img, output_shape=size, preserve_range=True, order=1)  # 1: Bi - linear(default)

    # resized = transform.resize(image=img, output_shape=size, preserve_range=True, order=2) # 2: Bi - quadratic
    # resized = transform.resize(image=img, output_shape=size, preserve_range=True, order=3) # 3: Bi - cubic
    # resized = transform.resize(image=img, output_shape=size, preserve_range=True, order=4) # 4: Bi - quartic
    # resized = transform.resize(image=img, output_shape=size, preserve_range=True, order=5) # 5: Bi - quintic

    # Debug
    def debug():
        print(img)
        print(resized)
        plt.figure()
        plt.imshow(img)
        plt.title("img")
        plt.figure()
        plt.imshow(resized)
        plt.title("resized")
        plt.show()

    # debug()

    return resized


def normalizeImage(img):
    pixel_depth = 255

    normed = (img - pixel_depth / 2) / pixel_depth

    # Debug
    # print("img[0,0,0]:", img[0, 0, 0], "img[0,0,1]:", img[0, 0, 1], "img[0,0,2]:", img[0, 0, 2])
    # print("normed[0,0,0]:", normed[0, 0, 0], "normed[0,0,1]:", normed[0, 0, 1], "normed[0,0,2]:", normed[0, 0, 2])

    return normed


def adjust_gamma(image, gamma, gain):
    scale = float(dtype_limits(image, True)[1] - dtype_limits(image, True)[0])  # 255.0

    return ((image / scale) ** gamma) * scale * gain  # float64


def adjust_brightness(image, brightness):
    scale = float(dtype_limits(image, True)[1] - dtype_limits(image, True)[0])  # 255.0

    return ((image / scale) * brightness) * scale  # float64


# ===================
#  Class Declaration
# ===================
class Dataloader(object):
    def __init__(self, data_path, dataset, mode, TRAIN_VALID_RATIO=0.8):
        # 80% for Training and 20% for Validation
        checkArgumentsIntegrity(dataset)
        print("\n[fcrn/Dataloader] Description: This script prepares the ", dataset,
              ".pkl file for posterior Networks training.",
              sep='')

        # Creates Dataset Handler
        self.datasetObj, dataset_path = selectedDataset(data_path, dataset)

        # Gets the list of folders inside the 'DATASET_PATH' folder
        print('\n[fcrn/Dataloader] Getting list of folders...')
        test_folders = getListFolders(os.path.join(dataset_path, 'testing'))
        train_folders = getListFolders(os.path.join(dataset_path, 'training'))

        print("Num of folders inside '", os.path.split(dataset_path)[1], "/testing': ", len(test_folders), sep='')
        print("Num of folders inside '", os.path.split(dataset_path)[1], "/training': ", len(train_folders), sep='')
        print(test_folders, '\n', train_folders, '\n', sep='')

        # Removing unused folders for Kitti datasets
        if self.datasetObj.type == 'kitti':
            test_folders, train_folders = removeUnusedFolders(test_folders, train_folders, self.datasetObj)

        # Gets the list of files inside each folder grouping *_colors.png and *_depth.png files.
        test_files_colors_path, test_files_depth_path = getListTestFiles(test_folders, self.datasetObj)
        train_files_colors_path, train_files_depth_path = getListTrainFiles(train_folders, self.datasetObj)

        print("Summary")
        print("Num of colored images found in '", os.path.split(dataset_path)[1], "/testing/*/: ",
              len(test_files_colors_path), sep='')
        print("Num of   train_depth_filepath images found in '", os.path.split(dataset_path)[1], "/testing/*/: ",
              len(test_files_depth_path), sep='')
        print("Num of colored images found in '", os.path.split(dataset_path)[1], "/training/*/: ",
              len(train_files_colors_path), sep='')
        print("Num of   train_depth_filepath images found in '", os.path.split(dataset_path)[1], "/training/*/: ",
              len(train_files_depth_path), sep='')
        print()

        # Gets only the filename from the complete file path
        print("[fcrn/Dataloader] Getting the filename files list...")
        test_files_colors_filename = getFilesFilename(test_files_colors_path)
        test_files_depth_filename = getFilesFilename(test_files_depth_path)
        train_files_colors_filename = getFilesFilename(train_files_colors_path)
        train_files_depth_filename = getFilesFilename(train_files_depth_path)

        # Checks which files have its train_depth_filepath/disparity correspondent
        print("\n[fcrn/Dataloader] Checking which files have its train_depth_filepath/disparity correspondent...")
        test_valid_pairs_idx = getValidPairFiles(test_files_colors_filename, test_files_depth_filename, self.datasetObj)
        train_valid_pairs_idx = getValidPairFiles(train_files_colors_filename, train_files_depth_filename,
                                                  self.datasetObj)

        """Testing"""
        test_colors_filepath, test_depth_filepath = [], []
        if len(test_valid_pairs_idx):  # Some test data doesn't have depth images
            for i, val in enumerate(test_valid_pairs_idx):
                test_colors_filepath.append(test_files_colors_path[val[0]])
                test_depth_filepath.append(test_files_depth_path[val[1]])

                # print('i:', i, 'idx:', val, 'colors:', test_colors_filepath[i], '\n\t\t\t\tdepth:', test_depth_filepath[i])
        else:
            test_colors_filepath = test_files_colors_path
            test_depth_filepath = []

        self.test_dataset = test_colors_filepath
        self.test_labels = test_depth_filepath

        # Divides the Processed train data into training set and validation set
        print('\n[fcrn/Dataloader] Dividing available data into training, validation and test sets...')
        trainSize = len(train_valid_pairs_idx)
        divider = int(TRAIN_VALID_RATIO * trainSize)

        """Training"""
        train_colors_filepath, train_depth_filepath = [], []
        for i, val in enumerate(train_valid_pairs_idx):
            train_colors_filepath.append(train_files_colors_path[val[0]])
            train_depth_filepath.append(train_files_depth_path[val[1]])

            # print('i:', i, 'idx:', val, 'train_colors_filepath:', train_colors_filepath[i], '\n\t\t\ttrain_depth_filepath:', train_depth_filepath[i])

        self.train_dataset = train_colors_filepath[:divider]
        self.train_labels = train_depth_filepath[:divider]

        """Validation"""
        self.valid_dataset = train_colors_filepath[divider:]
        self.valid_labels = train_depth_filepath[divider:]

        """Final"""
        print("\nSummary")
        print("train_dataset shape:", len(self.train_dataset))
        print("train_labels shape:", len(self.train_labels))
        print("valid_dataset shape:", len(self.valid_dataset))
        print("valid_labels shape:", len(self.valid_labels))
        print("test_dataset shape:", len(self.test_dataset))
        print("test_labels shape:", len(self.test_labels))

        self.data_path = data_path
        self.dataset = dataset
        self.mode = mode

        self.image_batch = None

        self.inputSize = (None, self.datasetObj.imageNetworkInputSize[0], self.datasetObj.imageNetworkInputSize[1], 3)
        self.outputSize = (None, self.datasetObj.depthNetworkOutputSize[0], self.datasetObj.depthNetworkOutputSize[1])
        self.numTrainSamples = len(self.train_dataset)
        self.numTestSamples = len(self.test_dataset)

    def readImage(self, colors_path, depth_path, mode, showImages=False):
        # The DataAugmentation Transforms should be done before the Image Normalization!!!
        # Kitti RGB Image: (375, 1242, 3) uint8     #   NyuDepth RGB Image: (480, 640, 3) uint8 #
        #     Depth Image: (375, 1242)    int32     #          Depth Image: (480, 640)    int32 #

        if mode == 'train' or mode == 'valid':
            img_colors = scp.imread(os.path.join(colors_path))
            img_depth = scp.imread(os.path.join(depth_path))

            # Data Augmentation
            img_colors_aug, img_depth_aug = self.augment_image_pair(img_colors, img_depth)

            # TODO: Implementar Random Crops
            # Crops Image
            img_colors_crop = cropImage(img_colors_aug, size=self.datasetObj.imageNetworkInputSize)
            img_depth_crop = cropImage(img_depth_aug, size=self.datasetObj.imageNetworkInputSize)

            # Normalizes RGB Image and Downsizes Depth Image
            img_colors_normed = normalizeImage(img_colors_crop)

            img_depth_downsized = np_resizeImage(img_depth_crop, size=self.datasetObj.depthNetworkOutputSize)

            # Results
            if showImages:
                def plot1():
                    scp.imshow(img_colors)
                    scp.imshow(img_depth)
                    scp.imshow(img_colors_aug)
                    scp.imshow(img_depth_aug)
                    scp.imshow(img_colors_crop)
                    scp.imshow(img_depth_crop)
                    scp.imshow(img_colors_normed)
                    scp.imshow(img_depth_downsized)

                def plot2():
                    fig, axarr = plt.subplots(4, 2)
                    axarr[0, 0].set_title("colors")
                    axarr[0, 0].imshow(img_colors)
                    axarr[0, 1].set_title("depth")
                    axarr[0, 1].imshow(img_depth)
                    axarr[1, 0].set_title("colors_aug")
                    axarr[1, 0].imshow(img_colors_aug)
                    axarr[1, 1].set_title("depth_aug")
                    axarr[1, 1].imshow(img_depth_aug)
                    axarr[2, 0].set_title("colors_crop")
                    axarr[2, 0].imshow(img_colors_crop)
                    axarr[2, 1].set_title("depth_crop")
                    axarr[2, 1].imshow(img_depth_crop)
                    axarr[3, 0].set_title("colors_normed")
                    axarr[3, 0].imshow(img_colors_normed)
                    axarr[3, 1].set_title("depth_downsized")
                    axarr[3, 1].imshow(img_depth_downsized)

                    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=2.0)

                # plot1()
                plot2()

                plt.show()  # Display it

            # Debug
            # print(img_colors.shape, img_colors.dtype)
            # print(img_depth.shape, img_depth.dtype)
            # input()

            return img_colors_normed, img_depth_downsized, img_colors_crop, img_depth_crop

        elif mode == 'test':
            img_colors = scp.imread(os.path.join(colors_path))
            img_colors_crop = cropImage(img_colors, size=self.datasetObj.imageNetworkInputSize)
            img_colors_normed = normalizeImage(img_colors_crop)

            img_depth = None
            img_depth_crop = None
            img_depth_downsized = None
            img_depth_bilinear = None

            if depth_path is not None:
                img_depth = scp.imread(
                    os.path.join(depth_path))
                img_depth_crop = cropImage(img_depth,
                                           size=self.datasetObj.imageNetworkInputSize)  # Same cropSize as the colors image

                img_depth_downsized = np_resizeImage(img_depth_crop, size=self.datasetObj.depthNetworkOutputSize)
                img_depth_bilinear = img_depth_crop  # Copy

            return img_colors_normed, img_depth_downsized, img_colors_crop, img_depth_bilinear

    @staticmethod
    def augment_image_pair(image, depth):
        """ATTENTION! Remember to also reproduce the transforms in the depth image. However, colors transformations can DEGRADE depth information!!!"""
        # Gets `image` info
        dtype = image.dtype.type  # Needs to be <class 'numpy.uint8'>

        # Copy
        image_aug = image
        depth_aug = depth

        # Randomly flip images (Horizontally)
        do_flip = np.random.uniform(0, 1)
        if do_flip > 0.5:
            image_aug = np.flip(image, axis=1)
            depth_aug = np.flip(depth, axis=1)

        # Randomly shift gamma
        random_gamma = np.random.uniform(low=0.8, high=1.2)
        # random_gamma = 5 # Test
        image_aug = adjust_gamma(image=image_aug, gamma=random_gamma, gain=1)

        # FIXME: Not working properly
        # Randomly shift brightness
        random_brightness = np.random.uniform(0.5, 2.0)
        # random_brightness = 100
        # image_aug = adjust_brightness(image=image_aug, brightness=random_brightness)

        # Randomly shift color
        random_colors = np.random.uniform(0.8, 1.2, 3)
        white = np.ones([np.shape(image)[0], np.shape(image)[1]])
        color_image = np.stack([white * random_colors[i] for i in range(3)], axis=2)
        image_aug = np.multiply(image_aug, color_image)

        # Saturate
        image_aug = dtype(exposure.rescale_intensity(image_aug, out_range="uint8"))

        # Debug
        def debug():
            print("do_flip:", do_flip)
            print("random_gamma:", random_gamma)
            print("random_brightness:", random_brightness)
            print("random_colors:", random_colors)
            print("image - (min: %d, max: %d)" % (np.min(image), np.max(image)))
            print("image_aug - (min: %d, max: %d)" % (np.min(image_aug), np.max(image_aug)))
            # print(image.dtype, image_aug.dtype)
            print()

            def showImages():
                plt.figure(1)
                plt.imshow(image)
                plt.title("image")
                plt.figure(2)
                plt.imshow(image_aug)
                plt.title("image_aug")
                plt.pause(2)

            # scp.imshow(image)
            # scp.imshow(image_aug)
            showImages()

        # debug()

        return image_aug, depth_aug
