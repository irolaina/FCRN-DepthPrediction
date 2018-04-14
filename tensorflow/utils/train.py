# ===========
#  Libraries
# ===========
import numpy as np
import tensorflow as tf

from collections import deque

# ==================
#  Global Variables
# ==================
LOSS_LOG_INITIAL_VALUE = 0.1

# Early Stop Configuration
AVG_SIZE = 20
MIN_EVALUATIONS = 1000
MAX_STEPS_AFTER_STABILIZATION = 10000


# ===================
#  Class Declaration
# ===================
class Train:
    def __init__(self, args, tf_image, tf_labels, input_size, output_size):
        with tf.name_scope('Inputs'):
            # Raw Input/Output
            self.tf_image = tf_image
            self.tf_labels = tf_labels

            # Network Input/Output
            self.tf_log_labels = tf.log(
                tf.cast(self.tf_labels, tf.float32) + tf.constant(LOSS_LOG_INITIAL_VALUE, dtype=tf.float32),
                name='log_labels')  # Just for displaying Image

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
