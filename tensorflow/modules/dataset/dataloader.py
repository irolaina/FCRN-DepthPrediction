# ===========
#  Libraries
# ===========
import os
import random
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import imageio

from skimage import transform

from .kitti2012 import Kitti2012
from .kitti2015 import Kitti2015
from .kitticontinuous import KittiContinuous
from .nyudepth import NyuDepth
from .apolloscape import Apolloscape

# ==================
#  Global Variables
# ==================
LOG_INITIAL_VALUE = 1


# ===========
#  Functions
# ===========
def getFilenamesTensors(image_filenames, depth_filenames):
    tf_image_filenames = tf.convert_to_tensor(image_filenames)
    tf_depth_filenames = tf.convert_to_tensor(depth_filenames)

    return tf_image_filenames, tf_depth_filenames


# ===================
#  Class Declaration
# ===================
class Dataloader:
    def __init__(self, args):
        # Detects which dataset was selected and creates the 'datasetObj'.
        self.selectedDataset = args.dataset
        # print(selectedDataset)

        if self.selectedDataset == 'kitti2012':
            self.datasetObj = Kitti2012(args.machine)

        elif self.selectedDataset == 'kitti2015':
            self.datasetObj = Kitti2015(args.machine)

        elif self.selectedDataset == 'kitticontinuous_residential':
            self.datasetObj = KittiContinuous(args.machine)

        elif self.selectedDataset == 'nyudepth':
            self.datasetObj = NyuDepth(args.machine)

        elif self.selectedDataset == 'apolloscape':
            self.datasetObj = Apolloscape(args.machine)

        else:
            print("[Dataloader] The typed dataset '%s' is invalid. "
                  "Check the list of supported datasets." % self.selectedDataset)
            sys.exit()

        # Collects Dataset Info
        self.dataset_name = self.datasetObj.name
        self.dataset_path = self.datasetObj.dataset_path
        self.image_size = self.datasetObj.image_size
        self.depth_size = self.datasetObj.depth_size

        # Searches dataset image/depth filenames lists
        self.train_image_filenames, self.train_depth_filenames, self.numTrainSamples = None, None, -1
        self.tf_train_image_filenames, self.tf_train_depth_filenames = None, None

        self.test_image_filenames, self.test_depth_filenames, self.numTestSamples = None, None, -1
        self.tf_test_image_filenames, self.tf_test_depth_filenames = None, None

        if args.mode == 'train':
            _ = self.getTrainData()
            _ = self.getTestData()

            self.tf_train_image = None
            self.tf_train_depth = None
        elif args.mode == 'test':  # TODO: Deixar como está, ou passar aquelas flags para dentro da class.
            self.tf_test_image = None
            self.tf_test_image = None
            pass

        print("[Dataloader] dataloader object created.")

    def getTrainData(self, mode='train'):
        image_filenames, depth_filenames = self.datasetObj.getFilenamesLists(mode)
        tf_image_filenames, tf_depth_filenames = getFilenamesTensors(image_filenames, depth_filenames)

        try:
            print("Summary - TrainData")
            print("image_filenames: ", len(image_filenames))
            print("depth_filenames: ", len(depth_filenames))

            self.numTrainSamples = len(image_filenames)

        except TypeError:
            print("[TypeError] 'image_filenames' and 'depth_filenames' are None.")

        self.train_image_filenames = image_filenames
        self.train_depth_filenames = depth_filenames
        self.tf_train_image_filenames = tf_image_filenames
        self.tf_train_depth_filenames = tf_depth_filenames

        return image_filenames, depth_filenames, tf_image_filenames, tf_depth_filenames

    def getTestData(self, mode='test'):
        image_filenames, depth_filenames = self.datasetObj.getFilenamesLists(mode)
        tf_image_filenames, tf_depth_filenames = getFilenamesTensors(image_filenames, depth_filenames)

        try:
            print("Summary - TestData (Validation Set)")
            print("image_filenames: ", len(image_filenames))
            print("depth_filenames: ", len(depth_filenames))

            self.numTestSamples = len(image_filenames)

        except TypeError:
            print("[TypeError] 'image_filenames' and 'depth_filenames' are None.")

        self.test_image_filenames = image_filenames
        self.test_depth_filenames = depth_filenames
        self.tf_test_image_filenames = tf_image_filenames
        self.tf_test_depth_filenames = tf_depth_filenames

        return image_filenames, depth_filenames, tf_image_filenames, tf_depth_filenames

    def rawdepth2meters(self, tf_depth):
        '''True Depth Value Calculation. May vary from dataset to dataset.'''
        if self.dataset_name == 'kitti2012' or self.dataset_name == 'kitti2015':
            tf_depth = (tf.cast(tf_depth, tf.float32)) / 256.0
        elif self.dataset_name == 'kitticontinuous_residential':
            tf_depth = (tf.cast(tf_depth, tf.float32)) / 3.0
        elif self.dataset_name == 'nyudepth':
            depthParam1 = 351.3
            depthParam2 = 1092.5

            tf_depth = (tf.cast(tf_depth, tf.float32))
            # tf_depth = depthParam1/(depthParam2 - tf.cast(tf_depth, tf.float32)) # FIXME: Falta fazer aquele swapbyte
            # imgDepthAbs(imgDepthAbs > maxDepth) = maxDepth; # TODO: Terminar
            # imgDepthAbs(imgDepthAbs < 0) = 0; # TODO: Terminar
        elif self.dataset_name == 'apolloscape':
            tf_depth = (tf.cast(tf_depth, tf.float32)) / 200.0


        return tf_depth

    def readData(self, tf_image_filenames, tf_depth_filenames):
        # Creates Inputs Queue.
        # ATTENTION! Since these tensors operate on a FifoQueue, using .eval() may misalign the pair (image, depth)!!!
        seed = random.randint(0, 2 ** 31 - 1)
        tf_train_image_filename_queue = tf.train.string_input_producer(tf_image_filenames, shuffle=False, seed=seed)
        tf_train_depth_filename_queue = tf.train.string_input_producer(tf_depth_filenames, shuffle=False, seed=seed)

        # Reads images
        image_reader = tf.WholeFileReader()
        tf_image_key, image_file = image_reader.read(tf_train_image_filename_queue)
        tf_depth_key, depth_file = image_reader.read(tf_train_depth_filename_queue)

        tf_image = tf.image.decode_png(image_file, channels=3, dtype=tf.uint8)
        if self.dataset_name == 'kitticontinuous_residential':
            tf_depth = tf.image.decode_png(depth_file, channels=1, dtype=tf.uint8)
        else:
            tf_depth = tf.image.decode_png(depth_file, channels=1, dtype=tf.uint16)

        # print(tf_image)   # Must be uint8!
        # print(tf_depth)   # Must be uint16/uin8!

        # True Depth Value Calculation. May vary from dataset to dataset.
        tf_depth = self.rawdepth2meters(tf_depth)

        # print(tf_image) # Must be uint8!
        # print(tf_depth) # Must be float32!

        return tf_image, tf_depth

    def readTestImage(self, image_path, depth_path, input_size, output_size, showImages=False):
        # Processing the RGB Image (uint8)
        image = imageio.imread(os.path.join(image_path))
        image_downsized = transform.resize(image, output_shape=input_size.getSize())
        image_normed = self.normalizeImage(image_downsized)  # TODO: Not Used!

        # Processing the Depth Image (uint16/uint8)
        if depth_path is not None:
            depth = imageio.imread(os.path.join(depth_path))
            depth_downsized = transform.resize(depth, output_shape=(output_size.height, output_size.width))

        # Results
        if showImages:
            print("image: ", image.shape)
            print("image_downsized: ", image_downsized.shape)
            print("image_normed: ", image_normed.shape)

            print("depth: ", depth.shape)
            print("depth_downsized: ", depth_downsized.shape)

            fig, axarr = plt.subplots(2, 2)
            axarr[0, 0].imshow(image), axarr[0, 0].set_title("colors")
            axarr[0, 1].imshow(depth), axarr[0, 1].set_title("depth")
            axarr[1, 0].imshow(image_downsized), axarr[1, 0].set_title("image_downsized")
            axarr[1, 1].imshow(depth_downsized), axarr[1, 1].set_title("depth_downsized")

            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=2.0)
            plt.show()  # Display it

        return image_downsized, depth_downsized

    @staticmethod
    def np_resizeImage(img, size):
        try:
            if size is None:
                raise ValueError
        except ValueError:
            print("[ValueError] Oops! Empty resizeSize list. Please sets the desired resizeSize.\n")

        # resized = transform.resize(image=img, output_shape=size, preserve_range=True, order=0)  # 0: Nearest - neighbor
        resized = transform.resize(image=img, output_shape=size, preserve_range=True, order=1)  # 1: Bi - linear(default)
        # resized = transform.resize(image=img, output_shape=size, preserve_range=True, order=2)  # 2: Bi - quadratic
        # resized = transform.resize(image=img, output_shape=size, preserve_range=True, order=3)  # 3: Bi - cubic
        # resized = transform.resize(image=img, output_shape=size, preserve_range=True, order=4)  # 4: Bi - quartic
        # resized = transform.resize(image=img, output_shape=size, preserve_range=True, order=5)  # 5: Bi - quintic

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

    @staticmethod
    def normalizeImage(image):
        mean = np.mean(image)
        normed = image/mean

        # Debug
        # print("img[0,0,0]:", img[0, 0, 0], "img[0,0,1]:", img[0, 0, 1], "img[0,0,2]:", img[0, 0, 2])
        # print("normed[0,0,0]:", normed[0, 0, 0], "normed[0,0,1]:", normed[0, 0, 1], "normed[0,0,2]:", normed[0, 0, 2])

        return normed
