# ===========
#  Libraries
# ===========
import numpy as np
import tensorflow as tf

from collections import deque

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
    def __init__(self, args, tf_image_resized, tf_depth_resized, input_size, output_size):
        with tf.name_scope('Inputs'):
            # Create Tensors for Batch Training
            self.tf_batch_data_resized, self.tf_batch_data, self.tf_batch_labels = self.prepareTrainData(
                tf_image_resized,
                tf_depth_resized,
                args.batch_size)

            # Raw Input/Output
            self.tf_image = self.tf_batch_data
            self.tf_labels = self.tf_batch_labels

            # Network Input/Output
            self.tf_log_labels = tf.log(self.tf_labels + tf.constant(LOG_INITIAL_VALUE, dtype=tf.float32),
                                        name='log_labels')  # Just for displaying Image

            self.tf_loss = None
            self.loss = -1

        with tf.name_scope('Train'):
            # Count the number of steps taken.
            self.tf_global_step = tf.Variable(0, trainable=False, name='global_step')
            self.tf_learningRate = args.learning_rate

            if args.ldecay:
                self.tf_learningRate = tf.train.exponential_decay(self.tf_learningRate, self.tf_global_step, 1000, 0.95,
                                                                  staircase=True,
                                                                  name='ldecay')

        self.trainCollection()

        print("\n[Network/Train] Training Tensors Created.")
        print(self.tf_batch_data)
        print(self.tf_batch_data_resized)
        print(self.tf_batch_labels)
        print(self.tf_image)
        print(self.tf_labels)
        print(self.tf_log_labels)
        print(self.tf_global_step)
        print(self.tf_learningRate)

    def trainCollection(self):
        tf.add_to_collection('image', self.tf_image)
        tf.add_to_collection('labels', self.tf_labels)
        tf.add_to_collection('global_step', self.tf_global_step)
        tf.add_to_collection('learning_rate', self.tf_learningRate)

    @staticmethod
    def augment_image_pair(image, depth):
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

    @staticmethod
    def prepareTrainData(tf_image_resized, tf_depth_resized, batch_size):
        # TODO: Neste Ponto, os dados de entrada ja deveriam estar seperados em treinamento e validação, acredito que imagens de validação não devem sofrer data augmentation

        # ------------------- #
        #  Data Augmentation  #
        # ------------------- #
        # Copy
        tf_image_proc = tf_image_resized
        tf_depth_proc = tf_depth_resized

        # TODO: Reativar
        # # randomly augment images
        # do_augment = tf.random_uniform([], 0, 1)
        # tf_image_proc, tf_depth_proc = tf.cond(do_augment > 0.5,
        #                                        lambda: self.augment_image_pair(tf_image_resized, tf_depth_resized),
        #                                        lambda: (tf_image_resized, tf_depth_resized))
        #
        # # Normalizes Input
        # tf_image_proc = tf.image.per_image_standardization(tf_image_proc)
        #

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


# TODO: Validar
class EarlyStopping:
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
