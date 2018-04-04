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


# ===================
#  Class Declaration
# ===================
class Model(object):
    def __init__(self, args, tf_image, labels, mode):
        self.mode = mode
        self.args = args

        model_index = 0
        self.model_collection = ['model_' + str(model_index)]

        self.build_model(tf_image, labels)

        if self.mode == 'test':
            return

        self.build_losses()
        self.build_optimizer()
        self.build_summaries()
        self.countParams()

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

            # Mask Out Pixels without depth values
            tf_idx = tf.where(self.tf_labels > 0)  # Tensor 'idx' of Valid Pixel values (batchID, idx)
            tf_valid_labels = tf.gather_nd(self.tf_labels, tf_idx)
            self.tf_valid_log_labels = tf.log(tf_valid_labels, name='log_labels')

            # Mask Out Prediction
            self.tf_valid_pred = tf.gather_nd(self.fcrn.get_output(), tf_idx)

        with tf.name_scope('Train'):
            self.tf_keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.tf_global_step = tf.Variable(0, trainable=False,
                                              name='global_step')  # Count the number of steps taken.

            self.tf_learningRate = self.args.learning_rate

            if self.args.ldecay:
                self.tf_learningRate = tf.train.exponential_decay(self.tf_learningRate, self.tf_global_step, 1000, 0.95,
                                                                  staircase=True,
                                                                  name='ldecay')

        # TODO: Reativar
        # tf.add_to_collection('image', self.tf_image)
        # tf.add_to_collection('labels', self.tf_labels)
        # tf.add_to_collection('keep_prob', self.tf_keep_prob)
        # tf.add_to_collection('global_step', self.tf_global_step)
        # tf.add_to_collection('bn_train', self.tf_bn_train)
        # tf.add_to_collection('learning_rate', self.tf_learningRate)
        # tf.add_to_collection('predCoarse', self.tf_predCoarse)
        # tf.add_to_collection('predFine', self.tf_predFine)
        # tf.add_to_collection('predCoarseBilinear', self.tf_predCoarseBilinear)
        # tf.add_to_collection('predFineBilinear', self.tf_predFineBilinear)

    def build_losses(self):
        with tf.name_scope("Losses"):
            # Select Loss Function:
            self.loss_name, self.tf_loss = loss.tf_MSE(self.tf_valid_pred,
                                                       self.tf_valid_log_labels)  # Default, only valid pixels
            # self.loss_name, tf_loss = loss.tf_MSE(net.get_output(), self.tf_log_labels)       # Don't Use! Regress all pixels, even the sky!
            # self.loss_name, self.tf_loss = loss.tf_L(self.tf_predFine, self.tf_log_labels, self.tf_idx, gamma=0.5) # Internal Mask Out, because of calculation of gradients.

            # TODO: Reativar
            # if self.params['l2norm']:
            #     self.tf_loss += loss.calculateL2norm()

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
            tf.summary.scalar('keep_prob', self.tf_keep_prob, collections=self.model_collection)

    @staticmethod
    def countParams():
        # Count Params
        total_num_parameters = 0
        for variable in tf.trainable_variables():
            total_num_parameters += np.array(variable.get_shape().as_list()).prod()
        print("[Network/Model] Number of trainable parameters: {}".format(total_num_parameters))

    @staticmethod
    def saveTrainedModel(save_path, session, saver, model_name):
        """ Saves trained model """
        # Creates saver obj which backups all the variables.
        print("[Network/Training] List of Saved Variables:")
        for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            print(i)  # i.name if you want just a name

        # train_saver = tf.train.Saver()                                                  # ~4.3 Gb
        # train_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))  # ~850 mb

        file_path = saver.save(session, os.path.join(save_path, "model." + model_name))
        print("\n[Results] Model saved in file: %s" % file_path)
