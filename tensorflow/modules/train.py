# ===========
#  Libraries
# ===========
from collections import deque

import numpy as np
import tensorflow as tf

from modules.args import args
from .dataloader import Dataloader
from .plot import Plot
from .third_party.laina.fcrn import ResNet50UpProj
from .third_party.tensorflow.inception_preprocessing import apply_with_random_selector
from .third_party.tensorflow.inception_preprocessing import distort_color

# ==================
#  Global Variables
# ==================

# Early Stop Configuration
AVG_SIZE = 20
MIN_EVALUATIONS = 1000
MAX_STEPS_AFTER_STABILIZATION = 10000


# ===================
#  Class Declaration
# ===================
class Train:
    def __init__(self, data, input_size, output_size):
        self.tf_train_image_key, self.tf_train_depth_key = None, None
        self.tf_train_image, self.tf_train_depth = None, None

        # TODO: modificar função para que não seja preciso retornar os tensors, utilizar self. variables diretamente.
        tf_image_key, tf_image, tf_depth_key, tf_depth = self.read_images(data.dataset.name, data.train_image_filenames, data.train_depth_filenames)

        with tf.name_scope('Input'):
            # Raw Input/Output
            # tf_image = tf.cast(tf_image, tf.float32)  # uint8 -> float32 [0.0, 255.0]
            tf_image = tf.image.convert_image_dtype(tf_image, tf.float32)  # uint8 -> float32 [0.0, 1.0]
            self.tf_image = tf_image
            self.tf_depth = tf_depth

            if args.data_aug:
                tf_image, tf_depth = self.augment_image_pair(tf_image, tf_depth)

            # True Depth Value Calculation. May vary from dataset to dataset.
            tf_depth = Dataloader.rawdepth2meters(tf_depth, data.dataset.name)

            # Crops Input and Depth Images (Removes Sky)
            if args.remove_sky:
                tf_image, tf_depth = Dataloader.remove_sky(tf_image, tf_depth, data.dataset.name)

            # Network Input/Output. Overwrite Tensors!
            self.tf_image = tf_image
            self.tf_depth = tf_depth

            # Downsizes Input and Depth Images
            # Recommend: align_corners=False
            # MonoDepth utilizes the AREA method.
            # self.tf_image_resized = tf.image.resize_images(self.tf_image, [input_size.height, input_size.width], tf.image.ResizeMethod.BILINEAR, False)
            # self.tf_depth_resized = tf.image.resize_images(self.tf_depth, [output_size.height, output_size.width], tf.image.ResizeMethod.BILINEAR, False)

            # self.tf_image_resized = tf.image.resize_images(self.tf_image, [input_size.height, input_size.width], tf.image.ResizeMethod.BILINEAR, True)
            # self.tf_depth_resized = tf.image.resize_images(self.tf_depth, [output_size.height, output_size.width], tf.image.ResizeMethod.BILINEAR, True)

            # self.tf_image_resized = tf.image.resize_images(self.tf_image, [input_size.height, input_size.width], tf.image.ResizeMethod.NEAREST_NEIGHBOR, False)
            # self.tf_depth_resized = tf.image.resize_images(self.tf_depth, [output_size.height, output_size.width], tf.image.ResizeMethod.NEAREST_NEIGHBOR, False)

            # self.tf_image_resized = tf.image.resize_images(self.tf_image, [input_size.height, input_size.width], tf.image.ResizeMethod.NEAREST_NEIGHBOR, True)
            # self.tf_depth_resized = tf.image.resize_images(self.tf_depth, [output_size.height, output_size.width], tf.image.ResizeMethod.NEAREST_NEIGHBOR, True)

            self.tf_image_resized = tf.image.resize_images(self.tf_image, [input_size.height, input_size.width], tf.image.ResizeMethod.AREA, False)
            self.tf_depth_resized = tf.image.resize_images(self.tf_depth, [output_size.height, output_size.width], tf.image.ResizeMethod.AREA, False)

            # self.tf_image_resized = tf.image.resize_images(self.tf_image, [input_size.height, input_size.width], tf.image.ResizeMethod.AREA, True)
            # self.tf_depth_resized = tf.image.resize_images(self.tf_depth, [output_size.height, output_size.width], tf.image.ResizeMethod.AREA, True)

            # FIXME: Por que dá erro?
            # self.tf_image_resized = tf.py_func(lambda img: imresize(img, [output_size.height, output_size.width]), [tf.cast(self.tf_image, tf.float32)], [tf.float32])
            # self.tf_depth_resized = tf.py_func(lambda img: imresize(img, [output_size.height, output_size.width]), [self.tf_depth], [tf.float32])

            # print(self.tf_image_resized)
            # print(self.tf_depth_resized)
            # input("resized")

            # self.tf_image_resized_uint8 = tf.cast(self.tf_image_resized, tf.uint8)  # Visual Purpose
            self.tf_image_resized_uint8 = tf.image.convert_image_dtype(self.tf_image_resized, tf.uint8)  # Visual Purpose

            # ==============
            #  Batch Config
            # ==============
            batch_size = args.batch_size
            num_threads = 4
            min_after_dequeue = 16
            capacity = min_after_dequeue + num_threads * batch_size

            # ===============
            #  Prepare Batch
            # ===============
            # Select:
            self.tf_batch_image_key, self.tf_batch_depth_key = tf.train.batch([tf_image_key, tf_depth_key], batch_size, num_threads, capacity)
            tf_batch_image_resized, tf_batch_image_resized_uint8, tf_batch_depth_resized = tf.train.batch([self.tf_image_resized, self.tf_image_resized_uint8, self.tf_depth_resized], batch_size, num_threads, capacity, shapes=[input_size.get_size(), input_size.get_size(), output_size.get_size()])
            # tf_batch_image, tf_batch_depth = tf.train.shuffle_batch([tf_image, tf_depth], batch_size, capacity, min_after_dequeue, num_threads, shapes=[image_size, depth_size])

            # Network Input/Output
            self.tf_batch_image = tf_batch_image_resized
            self.tf_batch_image_uint8 = tf_batch_image_resized_uint8
            self.tf_batch_depth = tf_batch_depth_resized

        self.fcrn = ResNet50UpProj({'data': self.tf_batch_image}, batch=args.batch_size, keep_prob=args.dropout, is_training=True)
        self.tf_pred = self.fcrn.get_output()

        # Clips predictions above a certain distance in meters. Inspired from Monodepth's article.
        # if max_depth is not None:
        #     self.tf_pred = tf.clip_by_value(self.tf_pred, 0, tf.constant(max_depth))

        with tf.name_scope('Train'):
            # Count the number of steps taken.
            self.tf_global_step = tf.Variable(0, trainable=False, name='global_step')
            self.tf_learning_rate = tf.constant(args.learning_rate, name='learning_rate')

            if args.ldecay:
                self.tf_learning_rate = tf.train.exponential_decay(self.tf_learning_rate,
                                                                   self.tf_global_step,
                                                                   decay_steps=1000,
                                                                   decay_rate=0.95,
                                                                   staircase=True,
                                                                   name='learning_rate')

            self.tf_loss = None
            self.loss = -1
            self.loss_hist = []

        self.train_collection()
        self.stop = EarlyStopping()

        if args.show_train_progress:
            self.plot = Plot(args.mode, title='Train Predictions')

        print("\n[Network/Train] Training Tensors created.")
        # print(tf_batch_image_resized)
        # print(tf_batch_image_resized_uint8)
        # print(tf_batch_depth_resized)
        print(self.tf_batch_image)
        print(self.tf_batch_image_uint8)
        print(self.tf_batch_depth)
        print(self.tf_global_step)
        print(self.tf_learning_rate)
        print()
        # input("train")

    def read_images(self, dataset_name, image_filenames, depth_filenames):  # Used only for train
        # Creates Inputs Queue.
        # ATTENTION! Since these tensors operate on a FifoQueue, using .eval() may get misaligned the pair (image, depth)!!!

        # filenames = list(zip(image_filenames, depth_filenames))
        # for filename in filenames:
        #     print(filename)
        # print(len(filenames))
        # input("readImage")

        tf_image_filenames = tf.constant(image_filenames)
        tf_depth_filenames = tf.constant(depth_filenames)

        tf_train_input_queue = tf.train.slice_input_producer([tf_image_filenames, tf_depth_filenames], shuffle=False)

        # Reads images
        self.tf_train_image_key = tf_train_input_queue[0]
        self.tf_train_depth_key = tf_train_input_queue[1]

        self.tf_train_image, self.tf_train_depth = Dataloader.decode_images(self.tf_train_image_key, self.tf_train_depth_key, dataset_name)

        # Retrieves shape
        # tf_image.set_shape(self.image_size.getSize())
        # tf_depth.set_shape(self.depth_size.getSize())

        tf_image_shape = tf.shape(self.tf_train_image)
        tf_depth_shape = tf.shape(self.tf_train_depth)

        # Print Tensors
        print("tf_image_key: \t", self.tf_train_image_key)
        print("tf_depth_key: \t", self.tf_train_depth_key)
        print("tf_image: \t", self.tf_train_image)
        print("tf_depth: \t", self.tf_train_depth)
        print("tf_image_shape: ", tf_image_shape)
        print("tf_depth_shape: ", tf_depth_shape)

        return self.tf_train_image_key, self.tf_train_image, self.tf_train_depth_key, self.tf_train_depth

    def train_collection(self):
        tf.add_to_collection('batch_image', self.tf_batch_image)
        tf.add_to_collection('batch_depth', self.tf_batch_depth)
        tf.add_to_collection('pred', self.tf_pred)

        tf.add_to_collection('global_step', self.tf_global_step)
        tf.add_to_collection('learning_rate', self.tf_learning_rate)

    @staticmethod
    def augment_image_pair(image, depth):
        # randomly flip images
        do_flip = tf.random_uniform([], 0, 1)
        image_aug = tf.cond(do_flip > 0.5, lambda: tf.image.flip_left_right(image), lambda: image)
        depth_aug = tf.cond(do_flip > 0.5, lambda: tf.image.flip_left_right(depth), lambda: depth)

        # randomly distort the colors.
        image_aug = apply_with_random_selector(image_aug, lambda image, ordering: distort_color(image, ordering), num_distort_cases=4)

        return image_aug, depth_aug


# TODO: Validar
class EarlyStopping(object):
    def __init__(self):
        # Local Variables
        self.mov_mean_last = 0
        self.mov_mean = deque()
        self.stab_counter = 0

    def check(self, step, valid_loss):
        self.mov_mean.append(valid_loss)

        if step > AVG_SIZE:
            self.mov_mean.popleft()

        mov_mean_avg = np.sum(self.mov_mean) / AVG_SIZE
        mov_mean_avg_last = np.sum(self.mov_mean_last) / AVG_SIZE

        if (mov_mean_avg >= mov_mean_avg_last) and step > MIN_EVALUATIONS:
            # print(step,stabCounter)

            self.stab_counter += 1
            if self.stab_counter > MAX_STEPS_AFTER_STABILIZATION:
                print("\nSTOP TRAINING! New samples may cause overfitting!!!")
                return 1
        else:
            self.stab_counter = 0

            self.mov_mean_last = deque(self.mov_mean)
