# ===========
#  Libraries
# ===========
import numpy as np
import tensorflow as tf

from collections import deque
from .model.fcrn import ResNet50UpProj
from .plot import Plot
from .dataloader import Dataloader

# ==================
#  Global Variables
# ==================
LOG_INITIAL_VALUE = 1

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
            self.tf_image = tf_image
            self.tf_depth = tf_depth

            if enableDataAug:
                tf_image, tf_depth = self.augment_image_pair(tf_image, tf_depth)

            # print(tf_image)   # Must be uint8!
            # print(tf_depth)   # Must be uint16/uin8!

            # True Depth Value Calculation. May vary from dataset to dataset.
            tf_depth = Dataloader.rawdepth2meters(tf_depth, dataset_name)

            # print(tf_image) # Must be uint8!
            # print(tf_depth) # Must be float32!

            # Crops Input and Depth Images (Removes Sky)
            self.tf_image, self.tf_depth = Dataloader.removeSky(tf_image, tf_depth, dataset_name)

            # Downsizes Input and Depth Images
            self.tf_image_resized = tf.image.resize_images(self.tf_image, [input_size.height, input_size.width])
            self.tf_depth_resized = tf.image.resize_images(self.tf_depth, [output_size.height, output_size.width])
            self.tf_image_resized_uint8 = tf.cast(self.tf_image_resized, tf.uint8)  # Visual purpose

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
            self.tf_batch_image_key, self.tf_batch_depth_key = tf.train.batch([tf_image_key, tf_depth_key], batch_size, num_threads,capacity)
            tf_batch_image_resized, tf_batch_image_resized_uint8, tf_batch_depth_resized = tf.train.batch([self.tf_image_resized, self.tf_image_resized_uint8, self.tf_depth_resized], batch_size, num_threads, capacity, shapes=[input_size.getSize(), input_size.getSize(), output_size.getSize()])
            # tf_batch_image, tf_batch_depth = tf.train.shuffle_batch([tf_image, tf_depth], batch_size, capacity, min_after_dequeue, num_threads, shapes=[image_size, depth_size])

            # Network Input/Output
            self.tf_batch_image = tf_batch_image_resized
            self.tf_batch_image_uint8 = tf_batch_image_resized_uint8
            self.tf_batch_depth = tf_batch_depth_resized
            self.tf_log_batch_depth = tf.log(self.tf_batch_depth + tf.constant(LOG_INITIAL_VALUE, dtype=tf.float32),
                                              name='log_batch_depth')

        self.fcrn = ResNet50UpProj({'data': self.tf_batch_image}, batch=args.batch_size, keep_prob=args.dropout, is_training=True)
        self.tf_pred = self.fcrn.get_output()

        # Clips predictions above a certain distance in meters. Inspired from Monodepth's article.
        # if max_depth is not None:
        #     self.tf_pred = tf.clip_by_value(self.tf_pred, 0, tf.log(tf.constant(max_depth)))

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
        print(self.tf_log_batch_depth)
        print(self.tf_global_step)
        print(self.tf_learning_rate)
        print()
        # input("train")

    def trainCollection(self):
        tf.add_to_collection('batch_image', self.tf_batch_image)
        tf.add_to_collection('batch_depth', self.tf_batch_depth)
        tf.add_to_collection('log_batch_depth', self.tf_log_batch_depth)
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
        # https://github.com/tensorflow/models/blob/master/research/inception/inception/image_processing.py
        # TODO: Atualizar dataaugmentation usando a implementação abaixo.
        # TODO: Checar se a função apply_with_random_selector() pode auxiliar a escolher os multiplos modes de color_ordering
        # https: // github.com / tensorflow / models / blob / master / research / slim / preprocessing / inception_preprocessing.py
        def color_ordering0(image_aug):
            image_aug = tf.image.random_brightness(image_aug, max_delta=32. / 255.)
            image_aug = tf.image.random_saturation(image_aug, lower=0.5, upper=1.5)
            image_aug = tf.image.random_hue(image_aug, max_delta=0.2)
            image_aug = tf.image.random_contrast(image_aug, lower=0.5, upper=1.5)

            return image_aug

        def color_ordering1(image_aug):
            image_aug = tf.image.random_brightness(image_aug, max_delta=32. / 255.)
            image_aug = tf.image.random_contrast(image_aug, lower=0.5, upper=1.5)
            image_aug = tf.image.random_saturation(image_aug, lower=0.5, upper=1.5)
            image_aug = tf.image.random_hue(image_aug, max_delta=0.2)

            return image_aug

        color_ordering = tf.random_uniform([], minval=0, maxval=2, dtype=tf.int32)
        image_aug = tf.cond(tf.equal(color_ordering, 0), lambda: color_ordering0(image_aug), lambda: color_ordering1(image_aug))

        # The random_* ops do not necessarily clamp.
        image_aug = tf.clip_by_value(tf.cast(image_aug, tf.float32), 0.0, 255.0) # TODO: Dar erro pq image_aug é uint8, posso realmente dar casting pra int32?

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
