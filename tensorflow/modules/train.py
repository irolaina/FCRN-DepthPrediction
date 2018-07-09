# ===========
#  Libraries
# ===========
import numpy as np
import tensorflow as tf

from collections import deque
from .model.fcrn import ResNet50UpProj
from scipy.misc import imresize

from .plot import Plot
from .dataloader import Dataloader
from .third_party.inception_preprocessing import apply_with_random_selector
from .third_party.inception_preprocessing import distort_color

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
    def __init__(self, args, tf_image_key, tf_image, tf_depth_key, tf_depth, input_size, output_size, max_depth, dataset_name, enableDataAug):
        with tf.name_scope('Input'):
            # Raw Input/Output
            # tf_image = tf.image.convert_image_dtype(tf_image, tf.float32)  # uint8 -> float32
            self.tf_image = tf_image
            self.tf_depth = tf_depth

            # FIXME: Não está funcionando
            if enableDataAug:
                tf_image, tf_depth = self.augment_image_pair(tf_image, tf_depth)

            # True Depth Value Calculation. May vary from dataset to dataset.
            tf_depth = Dataloader.rawdepth2meters(tf_depth, dataset_name)

            # Crops Input and Depth Images (Removes Sky)
            if args.remove_sky:
                tf_image, tf_depth = Dataloader.removeSky(tf_image, tf_depth, dataset_name)

            # Network Input/Output. Overwrite Tensors!
            self.tf_image = tf_image
            self.tf_depth = tf_depth

            # Downsizes Input and Depth Images
            # FIXME: O mais correto seria utilizar NEAREST_NEIGHBOR para redimensionar as imagens, porém a rede apresenta problemas de convergência com este método.
            # FIXME: Estou na dúvida se o problema estar no method ou no align_corners
            self.tf_image_resized = tf.image.resize_images(self.tf_image, [input_size.height, input_size.width])
            self.tf_depth_resized = tf.image.resize_images(self.tf_depth, [output_size.height, output_size.width])

            # self.tf_image_resized = tf.image.resize_images(self.tf_image, [input_size.height, input_size.width], tf.image.ResizeMethod.BILINEAR, False)
            # self.tf_depth_resized = tf.image.resize_images(self.tf_depth, [output_size.height, output_size.width], tf.image.ResizeMethod.BILINEAR, False)

            # self.tf_image_resized = tf.image.resize_images(self.tf_image, [input_size.height, input_size.width], tf.image.ResizeMethod.BILINEAR, True)
            # self.tf_depth_resized = tf.image.resize_images(self.tf_depth, [output_size.height, output_size.width], tf.image.ResizeMethod.BILINEAR, True)

            # self.tf_image_resized = tf.image.resize_images(tf.cast(self.tf_image, tf.float32), [input_size.height, input_size.width], tf.image.ResizeMethod.NEAREST_NEIGHBOR, False)
            # self.tf_depth_resized = tf.image.resize_images(self.tf_depth, [output_size.height, output_size.width], tf.image.ResizeMethod.NEAREST_NEIGHBOR, False)

            # self.tf_image_resized = tf.image.resize_images(tf.cast(self.tf_image, tf.float32), [input_size.height, input_size.width], tf.image.ResizeMethod.NEAREST_NEIGHBOR, True)
            # self.tf_depth_resized = tf.image.resize_images(self.tf_depth, [output_size.height, output_size.width], tf.image.ResizeMethod.NEAREST_NEIGHBOR, True)

            # FIXME: Por que dá erro?
            # self.tf_image_resized = tf.py_func(lambda img: imresize(img, [output_size.height, output_size.width]), [tf.cast(self.tf_image, tf.float32)], [tf.float32])
            # self.tf_depth_resized = tf.py_func(lambda img: imresize(img, [output_size.height, output_size.width]), [self.tf_depth], [tf.float32])

            # print(self.tf_image_resized)
            # print(self.tf_depth_resized)
            # input("resized")

            self.tf_image_resized_uint8 = tf.cast(self.tf_image_resized, tf.uint8)  # Visual Purpose
            # self.tf_image_resized_uint8 = tf.image.convert_image_dtype(self.tf_image_resized, tf.uint8)  # Visual Purpose

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
            tf_batch_image_resized, tf_batch_image_resized_uint8, tf_batch_depth_resized = tf.train.batch([self.tf_image_resized, self.tf_image_resized_uint8, self.tf_depth_resized], batch_size, num_threads, capacity, shapes=[input_size.getSize(), input_size.getSize(), output_size.getSize()])
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

        self.trainCollection()
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

    def trainCollection(self):
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
        self.movMeanLast = 0
        self.movMean = deque()
        self.stabCounter = 0

    def check(self, step, valid_loss):
        self.movMean.append(valid_loss)

        if step > AVG_SIZE:
            self.movMean.popleft()

        movMeanAvg = np.sum(self.movMean) / AVG_SIZE
        movMeanAvgLast = np.sum(self.movMeanLast) / AVG_SIZE

        if (movMeanAvg >= movMeanAvgLast) and step > MIN_EVALUATIONS:
            # print(step,stabCounter)

            self.stabCounter += 1
            if self.stabCounter > MAX_STEPS_AFTER_STABILIZATION:
                print("\nSTOP TRAINING! New samples may cause overfitting!!!")
                return 1
        else:
            self.stabCounter = 0

            self.movMeanLast = deque(self.movMean)
