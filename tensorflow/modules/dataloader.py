# ===========
#  Libraries
# ===========
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from skimage import transform

from modules.datasets.apolloscape import Apolloscape
from modules.datasets.kitticontinuous import KittiContinuous
from modules.datasets.kittidepth import KittiDepth
from modules.datasets.kittidiscrete import KittiDiscrete
from modules.datasets.nyudepth import NyuDepth


# ==================
#  Global Variables
# ==================


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
        # Defines dataset_root path depending on which machine is used.
        dataset_root = None

        if args.machine == 'xps':
            dataset_root = "/media/nicolas/nicolas_seagate/datasets/"
        elif args.machine == 'olorin':
            dataset_root = "/media/olorin/Documentos/datasets/"

        # Detects which dataset was selected and creates the 'datasetObj'.
        self.selectedDataset = args.dataset
        # print(selectedDataset)

        if self.selectedDataset == 'apolloscape':
            self.datasetObj = Apolloscape(dataset_root, self.selectedDataset)

        elif self.selectedDataset == 'kittidepth':
            self.datasetObj = KittiDepth(dataset_root, self.selectedDataset)

        elif self.selectedDataset == 'kittidiscrete' or \
             self.selectedDataset == 'kittidiscrete_city' or \
             self.selectedDataset == 'kittidiscrete_residential' or \
             self.selectedDataset == 'kittidiscrete_road' or \
             self.selectedDataset == 'kittidiscrete_campus' or \
             self.selectedDataset == 'kittidiscrete_person':
            self.datasetObj = KittiDiscrete(dataset_root, self.selectedDataset)

        elif self.selectedDataset == 'kitticontinuous' or \
             self.selectedDataset == 'kitticontinuous_city' or \
             self.selectedDataset == 'kitticontinuous_residential' or \
             self.selectedDataset == 'kitticontinuous_road' or \
             self.selectedDataset == 'kitticontinuous_campus' or \
             self.selectedDataset == 'kitticontinuous_person':
            self.datasetObj = KittiContinuous(dataset_root, self.selectedDataset)

        elif self.selectedDataset == 'nyudepth':
            self.datasetObj = NyuDepth(dataset_root, self.selectedDataset)

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

            self.tf_train_image_key = None
            self.tf_train_image = None

            self.tf_train_depth_key = None
            self.tf_train_depth = None

        elif args.mode == 'test':
            self.tf_test_image_key = None
            self.tf_test_image = None

            self.tf_test_depth_key = None
            self.tf_test_depth = None

        print("\n[Dataloader] dataloader object created.")

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

        return image_filenames, depth_filenames, tf_image_filenames, tf_depth_filenames, self.numTrainSamples

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

        return image_filenames, depth_filenames, tf_image_filenames, tf_depth_filenames, self.numTestSamples

    @staticmethod
    def rawdepth2meters(tf_depth, dataset_name):
        """True Depth Value Calculation. May vary from dataset to dataset."""
        if dataset_name == 'apolloscape':
            # Changes the invalid pixel value (65353) to 0.
            tf_depth = tf.cast(tf_depth, tf.float32)
            tf_imask = tf.where(tf_depth < 65535, tf.ones_like(tf_depth), tf.zeros_like(tf_depth))
            tf_depth = tf_depth * tf_imask

            tf_depth = (tf.cast(tf_depth, tf.float32)) / 200.0
        elif dataset_name == 'kittidepth':
            tf_depth = (tf.cast(tf_depth, tf.float32)) / 256.0
        elif dataset_name.split('_')[0] == 'kittidiscrete' or \
             dataset_name.split('_')[0] == 'kitticontinuous':
            tf_depth = (tf.cast(tf_depth, tf.float32)) / 3.0
        elif dataset_name == 'nyudepth':
            tf_depth = (tf.cast(tf_depth, tf.float32)) / 1000.0
        return tf_depth

    @staticmethod
    def removeSky(tf_image, tf_depth, dataset_name):
        """Crops Input and Depth Images (Removes Sky)"""
        if dataset_name[0:5] == 'kitti':
            tf_image_shape = tf.shape(tf_image)
            tf_depth_shape = tf.shape(tf_depth)

            crop_height_perc = tf.constant(0.3, tf.float32)
            tf_image_new_height = crop_height_perc * tf.cast(tf_image_shape[0], tf.float32)
            tf_depth_new_height = crop_height_perc * tf.cast(tf_depth_shape[0], tf.float32)

            tf_image = tf_image[tf.cast(tf_image_new_height, tf.int32):, :]
            tf_depth = tf_depth[tf.cast(tf_depth_new_height, tf.int32):, :]

        return tf_image, tf_depth

    def readData(self, image_filenames, depth_filenames):
        # Creates Inputs Queue.
        # ATTENTION! Since these tensors operate on a FifoQueue, using .eval() may misalign the pair (image, depth)!!!

        # filenames = list(zip(image_filenames, depth_filenames))
        # for filename in filenames:
        #     print(filename)
        # print(len(filenames))
        # input("readData")

        tf_image_filenames = tf.constant(image_filenames)
        tf_depth_filenames = tf.constant(depth_filenames)

        tf_train_input_queue = tf.train.slice_input_producer([tf_image_filenames, tf_depth_filenames], shuffle=False)

        # Reads images
        tf_image_key = tf_train_input_queue[0]
        tf_depth_key = tf_train_input_queue[1]

        tf_image_file = tf.read_file(tf_train_input_queue[0])
        tf_depth_file = tf.read_file(tf_train_input_queue[1])

        if self.dataset_name == 'apolloscape':
            tf_image = tf.image.decode_jpeg(tf_image_file, channels=3)
        else:
            tf_image = tf.image.decode_png(tf_image_file, channels=3, dtype=tf.uint8)

        if self.dataset_name.split('_')[0] == 'kittidiscrete' or \
           self.dataset_name.split('_')[0] == 'kitticontinuous':
            tf_depth = tf.image.decode_png(tf_depth_file, channels=1, dtype=tf.uint8)
        else:
            tf_depth = tf.image.decode_png(tf_depth_file, channels=1, dtype=tf.uint16)

        # Retrieves shape
        # tf_image.set_shape(self.image_size.getSize())
        # tf_depth.set_shape(self.depth_size.getSize())

        tf_image_shape = tf.shape(tf_image)
        tf_depth_shape = tf.shape(tf_depth)

        # Print Tensors
        print("tf_image_key: \t", tf_image_key)
        print("tf_depth_key: \t", tf_depth_key)
        print("tf_image_file: \t", tf_image_file)
        print("tf_depth_file: \t", tf_depth_file)
        print("tf_image: \t", tf_image)
        print("tf_depth: \t", tf_depth)
        print("tf_image_shape: ", tf_image_shape)
        print("tf_depth_shape: ", tf_depth_shape)

        return tf_image_key, tf_image, tf_depth_key, tf_depth

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
        normed = image / mean

        # Debug
        # print("img[0,0,0]:", img[0, 0, 0], "img[0,0,1]:", img[0, 0, 1], "img[0,0,2]:", img[0, 0, 2])
        # print("normed[0,0,0]:", normed[0, 0, 0], "normed[0,0,1]:", normed[0, 0, 1], "normed[0,0,2]:", normed[0, 0, 2])

        return normed
