# ===========
#  Libraries
# ===========
import tensorflow as tf
import random
import glob
import os

from .size import Size
from .kitti import Kitti
from .kittiraw import KittiRaw
from .nyudepth import NyuDepth

# ==================
#  Global Variables
# ==================
LOSS_LOG_INITIAL_VALUE = 0.1


# ===========
#  Functions
# ===========


# ===================
#  Class Declaration
# ===================
# TODO: Criar rotina que subdivide os dados disponívels em train/valid
class Dataloader:
    def __init__(self, args):
        # Detects which dataset was selected and creates the 'datasetObj'.
        self.selectedDataset = args.dataset
        # print(selectedDataset)

        if self.selectedDataset == 'kittiraw_residential_continuous':
            datasetObj = KittiRaw()  # TODO: Terminar
            pass

        elif self.selectedDataset == 'nyudepth':
            datasetObj = NyuDepth()  # TODO: Terminar
            pass

        else:
            print("[Dataloader] The typed dataset '%s' is invalid. Check the list of supported datasets." % self.selectedDataset)
            raise SystemExit

        # Collects Dataset Info
        self.dataset_name = datasetObj.name
        self.dataset_path = datasetObj.dataset_path
        self.image_size = datasetObj.image_size
        self.depth_size = datasetObj.depth_size

        print("[Dataloader] dataloader object created.")

    # TODO: Ler outros Datasets
    def getTrainInputs(self, args):
        if args.machine == 'olorin':
            # KittiRaw Residential Continuous
            # Image: (375, 1242, 3) uint8
            # Depth: (375, 1242)    uint8
            if args.dataset == 'kittiraw_residential_continuous':
                # TODO: Migrar informações para os handlers de cada dataset
                search_image_files_str = "../../mestrado_code/monodeep/data/residential_continuous/training/imgs/*.png"
                search_depth_files_str = "../../mestrado_code/monodeep/data/residential_continuous/training/dispc/*.png"

                tf_image_filenames = tf.train.match_filenames_once(search_image_files_str)
                tf_depth_filenames = tf.train.match_filenames_once(search_depth_files_str)

        elif args.machine == 'xps':
            # KittiRaw Residential Continuous
            # Image: (375, 1242, 3) uint8
            # Depth: (375, 1242)    uint8
            if args.dataset == 'kittiraw_residential_continuous':
                # TODO: Migrar informações para os handlers de cada dataset
                search_image_files_str = "../../data/residential_continuous/training/imgs/*.png"
                search_depth_files_str = "../../data/residential_continuous/training/dispc/*.png"

                self.image_filenames = None
                self.depth_filenames = None

                tf_image_filenames = tf.train.match_filenames_once(search_image_files_str)
                tf_depth_filenames = tf.train.match_filenames_once(search_depth_files_str)

            # NYU Depth v2
            # Image: (480, 640, 3) ?
            # Depth: (480, 640)    ?
            elif args.dataset == 'nyudepth':
                self.image_filenames = []
                self.depth_filenames = []

                # TODO: Migrar informações para os handlers de cada dataset
                root_folder = "/media/nicolas/Nícolas/datasets/nyu-depth-v2/images/training/"

                # Finds input images and labels inside list of folders.
                for folder in glob.glob(root_folder + "*/"):
                    # print(folder)
                    os.chdir(folder)

                    for file in glob.glob('*_colors.png'):
                        # print(file)
                        self.image_filenames.append(folder + file)

                    for file in glob.glob('*_depth.png'):
                        # print(file)
                        self.depth_filenames.append(folder + file)

                    # print()

                print("Summary - Training Inputs")
                print("image_filenames: ", len(self.image_filenames))
                print("depth_filenames: ", len(self.depth_filenames))

                tf_image_filenames = tf.constant(self.image_filenames)
                tf_depth_filenames = tf.constant(self.depth_filenames)


        return self.image_filenames, self.depth_filenames, tf_image_filenames, tf_depth_filenames

    def readData(self, tf_image_filenames, tf_depth_filenames):
        # Creates Inputs Queue.
        # ATTENTION! Since these tensors operate on a FifoQueue, using .eval() may misalign the pair (image, depth)!!!
        seed = random.randint(0, 2 ** 31 - 1)
        tf_train_image_filename_queue = tf.train.string_input_producer(tf_image_filenames, shuffle=False,
                                                                       seed=seed)
        tf_train_depth_filename_queue = tf.train.string_input_producer(tf_depth_filenames, shuffle=False,
                                                                       seed=seed)

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

    # TODO: Terminar
    def splitData(self):
        print("Terminar")

    def checkIntegrity(self, sess, tf_image_filenames, tf_depth_filenames):
        try:
            # TODO: Essas informações podem ser migradas para o handler de cada dataset
            if self.selectedDataset == 'kittiraw_residential_continuous':
                image_replace = [b'/imgs/', b'']
                depth_replace = [b'/dispc/', b'']

            elif self.selectedDataset == 'nyudepth':
                image_replace = [b'_colors.png', b'']
                depth_replace = [b'_depth.png', b'']

            image_filenames, depth_filenames = sess.run([tf_image_filenames, tf_depth_filenames])

            image_filenames_aux = [item.replace(image_replace[0], image_replace[1]) for item in image_filenames]
            depth_filenames_aux = [item.replace(depth_replace[0], depth_replace[1]) for item in depth_filenames]

            # print(image_filenames)
            # input("oi1")
            # print(depth_filenames)
            # input("oi2")
            #
            # print(image_filenames_aux)
            # input("oi3")
            # print(depth_filenames_aux)
            # input("oi4")

            numSamples = len(image_filenames_aux)

            print("[Dataloader] Checking if RGB and Depth images are paired... ")
            if image_filenames_aux == depth_filenames_aux:
                print("[Dataloader] Check Integrity: Pass")
            else:
                raise ValueError

            return numSamples

        except ValueError:
            print("[Dataloader] Check Integrity: Failed")
            raise SystemExit
