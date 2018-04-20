# ===========
#  Libraries
# ===========
import os
import random
import sys

import matplotlib.pyplot as plt
import tensorflow as tf
from scipy import misc as scp
from skimage import transform

from .kittiraw import KittiRaw
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
    tf_image_filenames = tf.constant(image_filenames)
    tf_depth_filenames = tf.constant(depth_filenames)

    return tf_image_filenames, tf_depth_filenames


# ===================
#  Class Declaration
# ===================
class Dataloader:
    def __init__(self, args):
        # Detects which dataset was selected and creates the 'datasetObj'.
        self.selectedDataset = args.dataset
        # print(selectedDataset)

        if self.selectedDataset == 'kittiraw_residential_continuous':
            self.datasetObj = KittiRaw(args.machine)

        elif self.selectedDataset == 'nyudepth':
            self.datasetObj = NyuDepth(args.machine)

        elif self.selectedDataset == 'apolloscape':
            self.datasetObj = Apolloscape(args.machine)

        else:
            print(
                "[Dataloader] The typed dataset '%s' is invalid. Check the list of supported datasets." % self.selectedDataset)
            sys.exit()

        # Collects Dataset Info
        self.dataset_name = self.datasetObj.name
        self.dataset_path = self.datasetObj.dataset_path
        self.image_size = self.datasetObj.image_size
        self.depth_size = self.datasetObj.depth_size

        # Filenames Lists
        self.train_image_filenames = None
        self.train_depth_filenames = None
        self.valid_image_filenames = None
        self.valid_depth_filenames = None

        self.numTrainSamples = -1
        self.numTestSamples = -1

        print("[Dataloader] dataloader object created.")

    def getTrainData(self, mode='train'):
        image_filenames, depth_filenames = self.datasetObj.getFilenamesLists(mode)
        tf_image_filenames, tf_depth_filenames = getFilenamesTensors(image_filenames, depth_filenames)

        try:
            print("\nSummary - TrainData")
            print("image_filenames: ", len(image_filenames))
            print("depth_filenames: ", len(depth_filenames))

            self.numTrainSamples = len(image_filenames)

        except TypeError:
            print("[TypeError] 'image_filenames' and 'depth_filenames' are None.")

        return image_filenames, depth_filenames, tf_image_filenames, tf_depth_filenames

    def getTestData(self, mode='test'):
        image_filenames, depth_filenames = self.datasetObj.getFilenamesLists(mode)
        tf_image_filenames, tf_depth_filenames = getFilenamesTensors(image_filenames, depth_filenames)

        try:
            print("\nSummary - TestData (Validation Set)")
            print("image_filenames: ", len(image_filenames))
            print("depth_filenames: ", len(depth_filenames))

            self.numTestSamples = len(image_filenames)

        except TypeError:
            print("[TypeError] 'image_filenames' and 'depth_filenames' are None.")

        return image_filenames, depth_filenames, tf_image_filenames, tf_depth_filenames

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

        # FIXME: Kitti Original as imagens de disparidade são do tipo int32, no caso do kittiraw_residential_continous são uint8
        tf_image = tf.image.decode_image(image_file, channels=3)  # uint8
        tf_depth = tf.image.decode_image(depth_file, channels=1)  # uint8

        # Restores images structure (size, type)
        tf_image.set_shape([self.image_size.height, self.image_size.width, self.image_size.nchannels])
        tf_depth.set_shape([self.depth_size.height, self.depth_size.width, self.depth_size.nchannels])

        return tf_image, tf_depth

    def splitData(self, image_filenames, depth_filenames, ratio=0.8):
        # Divides the Processed train data into training set and validation set
        print('\n[Dataloader] Dividing available data into training and validation sets...')
        divider = int(ratio * self.numSamples)

        """Training"""
        self.train_image_filenames = image_filenames[:divider]
        self.train_depth_filenames = depth_filenames[:divider]

        """Validation"""
        self.valid_image_filenames = image_filenames[divider:]
        self.valid_depth_filenames = depth_filenames[divider:]

        """Final"""
        print("\nSummary")
        print("len(train_image_filenames):", len(self.train_image_filenames))
        print("len(train_depth_filenames):", len(self.train_depth_filenames))
        print("len(valid_image_filenames):", len(self.valid_image_filenames))
        print("len(valid_depth_filenames):", len(self.valid_depth_filenames))

    def checkIntegrity(self, sess, tf_image_filenames, tf_depth_filenames, mode):
        try:
            image_filenames, depth_filenames = sess.run([tf_image_filenames, tf_depth_filenames])

            image_filenames_aux = [item.replace(self.datasetObj.image_replace[0], self.datasetObj.image_replace[1]) for
                                   item in image_filenames]
            depth_filenames_aux = [item.replace(self.datasetObj.depth_replace[0], self.datasetObj.depth_replace[1]) for
                                   item in depth_filenames]

            # flag = False
            # for i in range(len(image_filenames)):
            #     if image_filenames_aux[i] == depth_filenames_aux[i]:
            #         flag = True
            #     else:
            #         flag = False
            #
            #     print(i, image_filenames[i], depth_filenames[i], flag)

            if image_filenames_aux == depth_filenames_aux:
                print("[Dataloader/%s] Check Integrity: Pass" % mode)
            else:
                raise ValueError

        except ValueError:
            print("[Dataloader/%s] Check Integrity: Failed" % mode)
            sys.exit()

    def readImage(self, colors_path, depth_path, input_size, output_size, mode, showImages=False):
        # The DataAugmentation Transforms should be done before the Image Normalization!!!
        # Kitti RGB Image: (375, 1242, 3) uint8     #   NyuDepth RGB Image: (480, 640, 3) uint8 #
        #     Depth Image: (375, 1242)    int32     #          Depth Image: (480, 640)    int32 #

        if mode == 'train' or mode == 'valid':
            img_colors = scp.imread(os.path.join(colors_path))
            img_depth = scp.imread(os.path.join(depth_path))

            # Data Augmentation
            img_colors_aug, img_depth_aug = self.train.augment_image_pair(img_colors, img_depth)

            # TODO: Implementar Random Crops
            # Crops Image
            img_colors_crop = self.cropImage(img_colors_aug, size=input_size.getSize())
            img_depth_crop = self.cropImage(img_depth_aug, size=input_size.getSize())

            # Normalizes RGB Image and Downsizes Depth Image
            img_colors_normed = self.normalizeImage(img_colors_crop)

            img_depth_downsized = self.np_resizeImage(img_depth_crop, size=output_size.getSize())

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
            img_colors_crop = self.cropImage(img_colors, size=input_size.getSize())
            img_colors_normed = self.normalizeImage(img_colors_crop)

            img_depth = None
            img_depth_crop = None
            img_depth_downsized = None
            img_depth_bilinear = None

            if depth_path is not None:
                img_depth = scp.imread(
                    os.path.join(depth_path))
                img_depth_crop = self.cropImage(img_depth,
                                                size=input_size.getSize())  # Same cropSize as the colors image

                img_depth_downsized = self.np_resizeImage(img_depth_crop, size=output_size.getSize())
                img_depth_bilinear = img_depth_crop  # Copy

            return img_colors_normed, img_depth_downsized, img_colors_crop, img_depth_bilinear

    @staticmethod
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

    @staticmethod
    def np_resizeImage(img, size):
        try:
            if size is None:
                raise ValueError
        except ValueError:
            print("[ValueError] Oops! Empty resizeSize list. Please sets the desired resizeSize.\n")

        # TODO: Qual devo usar?
        # resized = scp.imresize(img, size, interp='bilinear')  # Attention! This method doesn't maintain the original depth range!!!
        # resized = transform.resize(image=img,output_shape=size, preserve_range=True, order=0)  # 0: Nearest - neighbor
        resized = transform.resize(image=img, output_shape=size, preserve_range=True,
                                   order=1)  # 1: Bi - linear(default)

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

    @staticmethod
    def normalizeImage(img):
        pixel_depth = 255

        normed = (img - pixel_depth / 2) / pixel_depth

        # Debug
        # print("img[0,0,0]:", img[0, 0, 0], "img[0,0,1]:", img[0, 0, 1], "img[0,0,2]:", img[0, 0, 2])
        # print("normed[0,0,0]:", normed[0, 0, 0], "normed[0,0,1]:", normed[0, 0, 1], "normed[0,0,2]:", normed[0, 0, 2])

        return normed
