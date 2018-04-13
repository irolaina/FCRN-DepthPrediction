# ===========
#  Libraries
# ===========
import tensorflow as tf
import random

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
class Dataloader_new():
    def __init__(self, args):
        # Detects which dataset was selected and creates the 'datasetObj'.
        self.selectedDataset = args.dataset
        # print(selectedDataset)

        if self.selectedDataset == 'kittiraw_residential_continuous':
            datasetObj = KittiRaw() # TODO: Terminar
            pass

        elif self.selectedDataset == 'nyudepth':
            datasetObj = NyuDepth() # TODO: Terminar
            pass

        else:
            print("[Dataset] The typed dataset '%s' is invalid. Check the list of supported datasets." % self.selectedDataset)
            raise SystemExit

        # Collects Dataset Info
        self.dataset_name = datasetObj.name
        self.dataset_path = datasetObj.dataset_path
        self.image_size = datasetObj.image_size
        self.depth_size = datasetObj.depth_size

        print("[Dataset] dataloader object created.")

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

                image_filenames = None
                depth_filenames = None

                tf_image_filenames = tf.train.match_filenames_once(search_image_files_str)
                tf_depth_filenames = tf.train.match_filenames_once(search_depth_files_str)

            # NYU Depth v2
            # Image: (480, 640, 3) ?
            # Depth: (480, 640)    ?
            elif args.dataset == 'nyudepth':
                image_filenames = []
                depth_filenames = []

                # TODO: Migrar informações para os handlers de cada dataset
                root_folder = "/media/nicolas/Nícolas/datasets/nyu-depth-v2/images/training/"

                # Finds input images and labels inside list of folders.
                for folder in glob.glob(root_folder + "*/"):
                    print(folder)
                    os.chdir(folder)

                    for file in glob.glob('*_colors.png'):
                        print(file)
                        image_filenames.append(folder + file)

                    for file in glob.glob('*_depth.png'):
                        print(file)
                        depth_filenames.append(folder + file)

                    print()

                print("Summary - Training Inputs")
                print("image_filenames: ", len(image_filenames))
                print("depth_filenames: ", len(depth_filenames))

                tf_image_filenames = tf.placeholder(tf.string)
                tf_depth_filenames = tf.placeholder(tf.string)

        return image_filenames, depth_filenames, tf_image_filenames, tf_depth_filenames

    def readData(self, tf_train_image_filenames, tf_train_depth_filenames):
        # Creates Inputs Queue.
        # ATTENTION! Since these tensors operate on a FifoQueue, using .eval() may misalign the pair (image, depth)!!!
        seed = random.randint(0, 2 ** 31 - 1)
        tf_train_image_filename_queue = tf.train.string_input_producer(tf_train_image_filenames, shuffle=False,
                                                                       seed=seed)
        tf_train_depth_filename_queue = tf.train.string_input_producer(tf_train_depth_filenames, shuffle=False,
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

    def augment_image_pair(self, image, depth):
        # randomly flip images
        do_flip = tf.random_uniform([], 0, 1)
        image_aug = tf.cond(do_flip > 0.5, lambda: tf.image.flip_left_right(image), lambda: image)
        depth_aug = tf.cond(do_flip > 0.5, lambda: tf.image.flip_left_right(depth), lambda: depth)

        # randomly shift gamma
        random_gamma = tf.random_uniform([], 0.8, 1.2)
        image_aug = image_aug ** random_gamma

        # randomly shift brightness
        random_brightness = tf.random_uniform([], 0.5, 2.0)
        image_aug = image_aug * random_brightness

        # randomly shift color
        random_colors = tf.random_uniform([3], 0.8, 1.2)
        white = tf.ones([tf.shape(image)[0], tf.shape(image)[1]])
        color_image = tf.stack([white * random_colors[i] for i in range(3)], axis=2)
        image_aug *= color_image

        # saturate
        image_aug = tf.clip_by_value(image_aug, 0, 1)

        return image_aug, depth_aug

    def prepareTrainData(self, tf_image_resized, tf_depth_resized, batch_size):
        # ------------------- #
        #  Data Augmentation  #
        # ------------------- #
        # Copy
        tf_image_proc = tf_image_resized
        tf_depth_proc = tf_depth_resized

        # randomly augment images
        do_augment = tf.random_uniform([], 0, 1)
        tf_image_proc, tf_depth_proc = tf.cond(do_augment > 0.5,
                                               lambda: self.augment_image_pair(tf_image_resized, tf_depth_resized),
                                               lambda: (tf_image_resized, tf_depth_resized))

        # Normalizes Input
        tf_image_proc = tf.image.per_image_standardization(tf_image_proc)

        tf_image_resized_uint8 = tf.cast(tf_image_resized, tf.uint8)  # Visual purpose

        # Creates Training Batch Tensors
        tf_batch_data_resized, tf_batch_data, tf_batch_labels = tf.train.shuffle_batch(
            # [tf_image_key, tf_depth_key],           # Enable for Debugging the filename strings.
            [tf_image_resized_uint8, tf_image_proc, tf_depth_proc],  # Enable for debugging images
            batch_size=batch_size,
            num_threads=1,
            capacity=16,
            min_after_dequeue=0)

        return tf_batch_data_resized, tf_batch_data, tf_batch_labels

    def checkIntegrity(self, tf_image_filenames, tf_depth_filenames, sess):
        try:
            # TODO: Essas informações podem ser migradas para o handler de cada dataset
            if self.selectedDataset == 'kittiraw_residential_continuous':
                feed_dict = None
                image_replace = [b'/imgs/', b'']
                depth_replace = [b'/dispc/', b'']

            elif self.selectedDataset == 'nyudepth':
                feed_dict = {tf_image_filenames: train_image_filenames,
                             tf_depth_filenames: train_depth_filenames}
                image_replace = ['_colors.png', '']
                depth_replace = ['_depth.png', '']

            image_filenames, depth_filenames = sess.run([tf_image_filenames, tf_depth_filenames],
                                                        feed_dict=feed_dict)

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

            print("[Dataset] Checking if RGB and Depth images are paired... ")
            if image_filenames_aux == depth_filenames_aux:
                print("[Dataset] Check Integrity: Pass")
            else:
                raise ValueError

            return numSamples, feed_dict

        except ValueError:
            print("[Dataset] Check Integrity: Failed")
            raise SystemExit