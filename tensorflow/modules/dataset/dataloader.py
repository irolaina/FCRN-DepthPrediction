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

from .kitti2012 import Kitti2012
from .kitti2015 import Kitti2015
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

        elif self.selectedDataset == 'kittiraw_residential_continuous':
            self.datasetObj = KittiRaw(args.machine)

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
        self.test_image_filenames, self.test_depth_filenames, self.numTestSamples = None, None, -1

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

    def readImage(self, image_path, depth_path, input_size, output_size, mode, showImages=False):
        # The DataAugmentation Transforms should be done before the Image Normalization!!!
        # Kitti RGB Image: (375, 1242, 3) uint8     #   NyuDepth RGB Image: (480, 640, 3) uint8 #
        #     Depth Image: (375, 1242)    int32     #          Depth Image: (480, 640)    int32 #

        if mode == 'train' or mode == 'valid':
            image = scp.imread(os.path.join(image_path))
            depth = scp.imread(os.path.join(depth_path))

            # Data Augmentation
            image_aug, depth_aug = self.train.augment_image_pair(image, depth)

            # TODO: Implementar Random Crops
            # Crops Image
            image_crop = self.cropImage(image_aug, size=input_size.getSize())
            depth_crop = self.cropImage(depth_aug, size=input_size.getSize())

            # Normalizes RGB Image and Downsizes Depth Image
            image_normed = self.normalizeImage(image_crop)

            depth_downsized = self.np_resizeImage(depth_crop, size=output_size.getSize())

            # Results
            if showImages:
                def plot1():
                    scp.imshow(image)
                    scp.imshow(depth)
                    scp.imshow(image_aug)
                    scp.imshow(depth_aug)
                    scp.imshow(image_crop)
                    scp.imshow(depth_crop)
                    scp.imshow(image_normed)
                    scp.imshow(depth_downsized)

                def plot2():
                    fig, axarr = plt.subplots(4, 2)
                    axarr[0, 0].set_title("image")
                    axarr[0, 0].imshow(image)
                    axarr[0, 1].set_title("depth")
                    axarr[0, 1].imshow(depth)
                    axarr[1, 0].set_title("image_aug")
                    axarr[1, 0].imshow(image_aug)
                    axarr[1, 1].set_title("depth_aug")
                    axarr[1, 1].imshow(depth_aug)
                    axarr[2, 0].set_title("image_crop")
                    axarr[2, 0].imshow(image_crop)
                    axarr[2, 1].set_title("depth_crop")
                    axarr[2, 1].imshow(depth_crop)
                    axarr[3, 0].set_title("image_normed")
                    axarr[3, 0].imshow(image_normed)
                    axarr[3, 1].set_title("depth_downsized")
                    axarr[3, 1].imshow(depth_downsized)

                    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=2.0)

                # plot1()
                plot2()

                plt.show()  # Display it

            # Debug
            # print(image.shape, image.dtype)
            # print(depth.shape, depth.dtype)
            # input()

            input("Deprecated!!!")

            # return image_normed, depth_downsized, image_crop, depth_crop

        elif mode == 'test':
            # Local Variables
            image = None
            image_downsized = None
            image_normed = None

            depth = None
            depth_downsized = None
            img_depth_bilinear = None

            # Processing the RGB Image
            image = scp.imread(os.path.join(image_path))
            image_downsized = scp.imresize(image, size=input_size.getSize())
            image_normed = self.normalizeImage(image_downsized) # TODO: Not Used!

            # Processing the Depth Image
            if depth_path is not None:
                depth = scp.imread(os.path.join(depth_path))
                depth_downsized = self.np_resizeImage(depth, size=output_size.getSize())

            # Results
            if showImages:
                print("image: ", image.shape)
                print("image_downsized: ",image_downsized.shape)
                print("image_normed: ", image_normed.shape)

                print("depth: ", depth.shape)
                print("depth_downsized: ", depth_downsized.shape)

                fig, axarr = plt.subplots(2, 2)
                axarr[0, 0].imshow(image),                     axarr[0, 0].set_title("colors")
                axarr[0, 1].imshow(depth),                      axarr[0, 1].set_title("depth")
                axarr[1, 0].imshow(image_downsized),           axarr[1, 0].set_title("image_downsized")
                axarr[1, 1].imshow(depth_downsized[:, :, 0]),   axarr[1, 1].set_title("depth_downsized")

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

    @staticmethod
    def normalizeImage(img):
        pixel_depth = 255

        normed = (img - pixel_depth / 2) / pixel_depth

        # Debug
        # print("img[0,0,0]:", img[0, 0, 0], "img[0,0,1]:", img[0, 0, 1], "img[0,0,2]:", img[0, 0, 2])
        # print("normed[0,0,0]:", normed[0, 0, 0], "normed[0,0,1]:", normed[0, 0, 1], "normed[0,0,2]:", normed[0, 0, 2])

        return normed
