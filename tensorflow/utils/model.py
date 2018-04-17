# ===========
#  Libraries
# ===========
import numpy as np
import tensorflow as tf
import os
import sys
import utils.loss as loss

from utils.size import Size
from utils.train import Train
from utils.validation import Validation
from utils.fcrn import ResNet50UpProj

# ==================
#  Global Variables
# ==================
LOG_INITIAL_VALUE = 1


# ===========
#  Functions
# ===========

# ===================
#  Class Declaration
# ===================
class Model(object):
    def __init__(self, args):
        self.args = args

        self.input_size = Size(228, 304, 3)
        self.output_size = Size(128, 160, 1)

        model_index = 0
        self.model_collection = ['model_' + str(model_index)]

        # self.build_model(tf_image, labels)
        # self.build_losses()
        # self.build_optimizer()
        # self.build_summaries()
        # self.countParams()

    def build_model(self, image_size, depth_size, tf_image_resized, tf_depth_resized):
        print("\n[Network/Model] Build Network Model...")

        # =============================================
        #  FCRN (Fully Convolutional Residual Network)
        # =============================================
        # Construct the network graphs
        with tf.variable_scope('model') as scope:
            self.train = Train(self.args, tf_image_resized, tf_depth_resized, self.input_size, self.output_size)
            self.fcrn = ResNet50UpProj({'data': self.train.tf_batch_data}, self.args.batch_size, 1, False)
            tf.add_to_collection('pred', self.fcrn.get_output())  # TODO: Move

        with tf.variable_scope("model", reuse=True):
            self.valid = Validation(image_size, depth_size, self.input_size, self.output_size)
            self.fcrn_valid = ResNet50UpProj({'data': self.valid.tf_image_resized}, self.args.batch_size, 1, False)

    def build_losses(self, selectedLoss, valid_pixels):
        with tf.name_scope("Losses"):
            # Select Loss Function:
            if selectedLoss == 0:
                self.loss_name, self.train.tf_loss = loss.tf_MSE(self.fcrn.get_output(),
                                                           self.train.tf_labels,
                                                           valid_pixels)

                _, self.valid.tf_loss = loss.tf_MSE(self.fcrn_valid.get_output(),
                                                    self.valid.tf_log_depth_resized,
                                                    valid_pixels)

            elif selectedLoss == 1:
                self.loss_name, self.train.tf_loss = loss.tf_L(self.fcrn.get_output(),
                                                         self.tf_log_labels,
                                                         valid_pixels,
                                                         gamma=0.5)
            elif selectedLoss == 2:
                self.loss_name, self.train.tf_loss = loss.tf_BerHu(self.fcrn.get_output(),
                                                             self.train.tf_labels,
                                                             valid_pixels)
            else:
                print("[Network/Loss] Invalid Loss Function Selected!")
                sys.exit()

            if self.args.l2norm:
                self.train.tf_loss += loss.calculateL2norm()

            print("\n[Network/Loss] Loss Function: %s" % self.loss_name)
            if valid_pixels:
                print("[Network/Loss] Compute: Ignore invalid pixels")
            else:
                print("[Network/Loss] Loss: All Pixels")

    def build_optimizer(self):
        with tf.name_scope("Optimizer"):
            # optimizer = tf.train.GradientDescentOptimizer(self.learningRate).minimize(self.train.tf_loss,
            #                                                global_step=self.global_step)
            optimizer = tf.train.AdamOptimizer(self.train.tf_learningRate)
            self.train_step = optimizer.minimize(self.train.tf_loss, global_step=self.train.tf_global_step)
            tf.add_to_collection("train_step", self.train_step)

    # TODO: Adicionar mais summaries das variaveis internas do modelo
    def build_summaries(self):
        # Filling Summary Obj
        with tf.name_scope("Summaries"):
            tf.summary.scalar('learning_rate', self.train.tf_learningRate, collections=self.model_collection)
            tf.summary.scalar('loss', self.train.tf_loss, collections=self.model_collection)

    @staticmethod
    def countParams():
        # Count Params
        total_num_parameters = 0
        for variable in tf.trainable_variables():
            total_num_parameters += np.array(variable.get_shape().as_list()).prod()
        print("[Network/Model] Number of trainable parameters: {}".format(total_num_parameters))

    def collectSummaries(self, save_path, graph):
        with tf.name_scope("Summaries"):
            # Summary Objects
            self.summary_writer = tf.summary.FileWriter(save_path + self.args.log_directory, graph)
            self.summary_op = tf.summary.merge_all('model_0')

    def createTrainSaver(self):
        """ Creates Saver Object """
        self.train_saver = tf.train.Saver()

    @staticmethod
    def saveTrainedModel(save_path, session, saver, model_name):
        # Creates saver obj which backups all the variables.
        print("[Network/Training] List of Saved Variables:")
        for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            print(i)  # i.name if you want just a name

        file_path = saver.save(session, os.path.join(save_path, "model." + model_name))
        print("\n[Results] Model saved in file: %s" % file_path)
