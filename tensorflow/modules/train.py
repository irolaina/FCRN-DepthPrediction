# ===========
#  Libraries
# ===========
import numpy as np
import tensorflow as tf

from collections import deque
from .model.fcrn import ResNet50UpProj
from .plot import Plot

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
    def __init__(self, args, tf_image, tf_depth, input_size, output_size, max_depth, dataset_name, enableDataAug):
        with tf.name_scope('Input'):
            # Raw Input/Output
            self.tf_image = tf_image
            self.tf_depth = tf_depth

            if enableDataAug:
                self.tf_image, self.tf_depth = self.augment_image_pair(self.tf_image, self.tf_depth)

            # Crops Input and Depth Images (Removes Sky)
            if dataset_name[0:5] == 'kitti':
                tf_image_shape = tf.shape(tf_image)
                tf_depth_shape = tf.shape(tf_depth)

                crop_height_perc = tf.constant(0.3, tf.float32)
                tf_image_new_height = crop_height_perc * tf.cast(tf_image_shape[0], tf.float32)
                tf_depth_new_height = crop_height_perc * tf.cast(tf_depth_shape[0], tf.float32)

                self.tf_image = tf_image[tf.cast(tf_image_new_height, tf.int32):, :]
                self.tf_depth = tf_depth[tf.cast(tf_depth_new_height, tf.int32):, :]

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
            tf_batch_image_resized, tf_batch_image_resized_uint8, tf_batch_depth_resized = tf.train.batch([self.tf_image_resized, self.tf_image_resized_uint8, self.tf_depth_resized], batch_size, num_threads, capacity, shapes=[input_size.getSize(), input_size.getSize(), output_size.getSize()])
            # tf_batch_image, tf_batch_depth = tf.train.shuffle_batch([tf_image, tf_depth], batch_size, capacity, min_after_dequeue, num_threads, shapes=[image_size, depth_size])

            # Network Input/Output
            self.tf_batch_data = tf_batch_image_resized
            self.tf_batch_data_uint8 = tf_batch_image_resized_uint8
            self.tf_batch_labels = tf_batch_depth_resized
            self.tf_log_batch_labels = tf.log(self.tf_batch_labels + tf.constant(LOG_INITIAL_VALUE, dtype=tf.float32),
                                              name='log_batch_labels')

        self.fcrn = ResNet50UpProj({'data': self.tf_batch_data}, batch=args.batch_size, keep_prob=args.dropout, is_training=True)
        self.tf_pred = self.fcrn.get_output()

        # Clips predictions above a certain distance in meters. Inspired from Monodepth's article.
        if max_depth is not None:
            self.tf_pred = tf.clip_by_value(self.tf_pred, 0, tf.log(tf.constant(max_depth)))

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
        print(self.tf_batch_data)
        print(self.tf_batch_data_uint8)
        print(self.tf_batch_labels)
        print(self.tf_log_batch_labels)
        print(self.tf_global_step)
        print(self.tf_learning_rate)
        print()
        # input("train")

    def trainCollection(self):
        tf.add_to_collection('batch_data', self.tf_batch_data)
        tf.add_to_collection('batch_labels', self.tf_batch_labels)
        tf.add_to_collection('log_batch_labels', self.tf_log_batch_labels)
        tf.add_to_collection('pred', self.tf_pred)

        tf.add_to_collection('global_step', self.tf_global_step)
        tf.add_to_collection('learning_rate', self.tf_learning_rate)

    @staticmethod
    def augment_image_pair(image, depth):
        # randomly flip images
        do_flip = tf.random_uniform([], 0, 1)
        image_aug = tf.cond(do_flip > 0.5, lambda: tf.image.flip_left_right(image), lambda: image)
        depth_aug = tf.cond(do_flip > 0.5, lambda: tf.image.flip_left_right(depth), lambda: depth)

        # randomly shift gamma
        random_gamma = tf.random_uniform([], 0.8, 1.2)
        image_aug = tf.cast(image_aug, tf.float32) ** random_gamma

        # randomly shift brightness
        random_brightness = tf.random_uniform([], 0.5, 2.0)
        image_aug = image_aug * random_brightness

        # TODO: Validar transformações abaixo
        # # randomly shift color
        # random_colors = tf.random_uniform([3], 0.8, 1.2)
        # white = tf.ones([tf.shape(image)[0], tf.shape(image)[1]])
        # color_image = tf.stack([white * random_colors[i] for i in range(3)], axis=2)
        # image_aug *= color_image
        #
        # # saturate
        # image_aug = tf.clip_by_value(image_aug, 0, 1)

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
