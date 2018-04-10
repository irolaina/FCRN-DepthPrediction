# ===========
#  Libraries
# ===========
import numpy as np
import tensorflow as tf
import os
import utils.loss as loss

from utils.fcrn import ResNet50UpProj

# ==================
#  Global Variables
# ==================
LOSS_LOG_INITIAL_VALUE = 0.1


# ===========
#  Functions
# ===========
class Size():
    def __init__(self, height, width, nchannels):
        self.height = height
        self.width = width
        self.nchannels = nchannels

# ===================
#  Class Declaration
# ===================
class Model(object):
    def __init__(self, args):
        self.args = args

        self.inputSize = Size(228, 304, 3)
        self.outputSize = Size(128, 160, 1)

        model_index = 0
        self.model_collection = ['model_' + str(model_index)]

        # self.build_model(tf_image, labels)
        # self.build_losses()
        # self.build_optimizer()
        # self.build_summaries()
        # self.countParams()

    def build_model(self, tf_image, tf_labels):
        # =============================================
        #  FCRN (Fully Convolutional Residual Network)
        # =============================================
        # Construct the network graph
        self.fcrn = ResNet50UpProj({'data': tf_image}, self.args.batch_size, 1, False)

        # ======================
        #  Tensorflow Variables
        # ======================
        print("\n[Network/Model] Build Network Model...")

        with tf.name_scope('Inputs'):
            self.tf_image = tf_image
            self.tf_labels = tf_labels

            self.tf_log_labels = tf.log(
                tf.cast(self.tf_labels, tf.float32) + tf.constant(LOSS_LOG_INITIAL_VALUE, dtype=tf.float32),
                name='log_labels')  # Just for displaying Image

        with tf.name_scope('Train'):
            self.tf_global_step = tf.Variable(0, trainable=False,
                                              name='global_step')  # Count the number of steps taken.

            self.tf_learningRate = self.args.learning_rate

            if self.args.ldecay:
                self.tf_learningRate = tf.train.exponential_decay(self.tf_learningRate, self.tf_global_step, 1000, 0.95,
                                                                  staircase=True,
                                                                  name='ldecay')

        tf.add_to_collection('image', self.tf_image)
        tf.add_to_collection('labels', self.tf_labels)
        tf.add_to_collection('global_step', self.tf_global_step)
        tf.add_to_collection('learning_rate', self.tf_learningRate)
        tf.add_to_collection('pred', self.fcrn.get_output())

    def build_losses(self):
        with tf.name_scope("Losses"):
            # Select Loss Function:

            # ----- MSE ----- #
            # self.loss_name, self.tf_loss = loss.tf_MSE(self.fcrn.get_output(), self.tf_labels, valid_pixels=True)  # Default, only valid pixels
            # self.loss_name, self.tf_loss = loss.tf_MSE(self.fcrn.get_output(), self.tf_labels, valid_pixels=False)  # Default, only valid pixels

            # self.loss_name, tf_loss = loss.tf_MSE(net.get_output(), self.tf_log_labels)       # Don't Use! Regress all pixels, even the sky!

            # ----- Eigen's Log Depth ----- #
            # self.loss_name, self.tf_loss = loss.tf_L(self.fcrn.get_output(), self.tf_log_labels, self.tf_idx, gamma=0.5) # Internal Mask Out, because of calculation of gradients.

            # ----- BerHu ----- #
            self.loss_name, self.tf_loss = loss.tf_BerHu(self.fcrn.get_output(), self.tf_labels, valid_pixels=True)
            # self.loss_name, self.tf_loss = loss.tf_BerHu(self.fcrn.get_output(), self.tf_labels, valid_pixels=False)

            if self.args.l2norm:
                self.tf_loss += loss.calculateL2norm()

            print("[Network/Model] Loss Function: %s" % self.loss_name)

    def build_optimizer(self):
        with tf.name_scope("Optimizer"):
            # optimizer = tf.train.GradientDescentOptimizer(self.learningRate).minimize(self.tf_loss,
            #                                                global_step=self.global_step)
            optimizer = tf.train.AdamOptimizer(self.tf_learningRate)
            self.train = optimizer.minimize(self.tf_loss, global_step=self.tf_global_step)
            tf.add_to_collection("train_step", self.train)

    # TODO: Criar summaries das variaveis internas do modelo
    def build_summaries(self):
        # Filling Summary Obj
        with tf.name_scope("Summaries"):
            tf.summary.scalar('learning_rate', self.tf_learningRate, collections=self.model_collection)
            tf.summary.scalar('loss', self.tf_loss, collections=self.model_collection)

    @staticmethod
    def countParams():
        # Count Params
        total_num_parameters = 0
        for variable in tf.trainable_variables():
            total_num_parameters += np.array(variable.get_shape().as_list()).prod()
        print("[Network/Model] Number of trainable parameters: {}".format(total_num_parameters))

    def collectSummaries(self, save_path, graph):
        # TODO: Mover para model.py
        # TODO: Enable Summaries
        with tf.name_scope("Summaries"):
            # Summary Objects
            self.summary_writer = tf.summary.FileWriter(save_path + self.args.log_directory, graph)
            self.summary_op = tf.summary.merge_all('model_0')

    def createTrainSaver(self):
        ''' Creates Saver Object '''
        self.train_saver = tf.train.Saver()

    @staticmethod
    def saveTrainedModel(save_path, session, saver, model_name):
        # Creates saver obj which backups all the variables.
        print("[Network/Training] List of Saved Variables:")
        for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            print(i)  # i.name if you want just a name

        file_path = saver.save(session, os.path.join(save_path, "model." + model_name))
        print("\n[Results] Model saved in file: %s" % file_path)