# ===========
#  Libraries
# ===========
import numpy as np
import tensorflow as tf
import os
import sys
import modules.loss as loss

from modules.size import Size
from modules.train import Train
from modules.validation import Validation

# ==================
#  Global Variables
# ==================


# ===========
#  Functions
# ===========

# ===================
#  Class Declaration
# ===================
class Model(object):
    def __init__(self, args, data, selected_loss, valid_pixels):
        self.args = args

        self.input_size = Size(228, 304, 3)
        self.output_size = Size(128, 160, 1)

        model_index = 0
        self.model_collection = ['model_' + str(model_index)]

        self.train = None
        self.valid = None

        self.loss_name = ''
        self.train_step = None
        self.summary_writer = None
        self.summary_op = None
        self.train_saver = None

        # Invoke Methods
        self.build_model(data)
        self.build_losses(selected_loss, valid_pixels)
        self.build_optimizer()
        self.build_summaries()
        self.countParams()

    def build_model(self, data):
        print("\n[Network/Model] Build Network Model...")

        # =============================================
        #  FCRN (Fully Convolutional Residual Network)
        # =============================================
        # Construct the network graphs
        with tf.variable_scope("model"):
            self.train = Train(self.args, data.tf_train_image, data.tf_train_depth, self.input_size, self.output_size, data.datasetObj.max_depth, data.dataset_name, self.args.data_aug)

        with tf.variable_scope("model", reuse=True):
            self.valid = Validation(self.args, self.input_size, self.output_size, data.datasetObj.max_depth, data.dataset_name)

    def build_losses(self, selected_loss, valid_pixels):
        with tf.name_scope("Losses"):
            # Select Loss Function:
            if selected_loss == 'mse':
                self.loss_name, self.train.tf_loss = loss.tf_MSE(self.train.tf_pred,
                                                                 self.train.tf_batch_labels,
                                                                 valid_pixels)

                _, self.valid.tf_loss = loss.tf_MSE(self.valid.tf_pred,
                                                    self.valid.tf_depth_resized,
                                                    valid_pixels)

            # TODO: Essa loss usa log, revisar implementacao
            elif selected_loss == 'eigen':
                self.loss_name, self.train.tf_loss = loss.tf_L(self.train.tf_pred,
                                                               self.train.tf_batch_labels,
                                                               valid_pixels,
                                                               gamma=0.5)

                _, self.valid.tf_loss = loss.tf_L(self.valid.tf_pred,
                                                  self.valid.tf_depth_resized,
                                                  valid_pixels)
            elif selected_loss == 'berhu':
                self.loss_name, self.train.tf_loss = loss.tf_BerHu(self.train.tf_pred,
                                                                   self.train.tf_batch_labels,
                                                                   valid_pixels)

                _, self.valid.tf_loss = loss.tf_BerHu(self.valid.tf_pred,
                                                      self.valid.tf_depth_resized,
                                                      valid_pixels)
            else:
                print("[Network/Loss] Invalid Loss Function Selected!")
                sys.exit()

            if self.args.l2norm:
                self.train.tf_loss += loss.calculateL2norm()

            if valid_pixels:
                print("[Network/Loss] Compute: Ignore invalid pixels")
            else:
                print("[Network/Loss] Loss: All Pixels")
            print("[Network/Loss] Loss Function: %s" % self.loss_name)

    def build_optimizer(self):
        with tf.name_scope("Optimizer"):
            # optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.train.tf_loss,
            #                                                global_step=self.global_step)
            optimizer = tf.train.AdamOptimizer(self.train.tf_learning_rate)
            self.train_step = optimizer.minimize(self.train.tf_loss, global_step=self.train.tf_global_step)
            tf.add_to_collection("train_step", self.train_step)

    def build_summaries(self):
        # Filling Summary Obj
        with tf.name_scope("Train"):
            tf.summary.scalar('learning_rate', self.train.tf_learning_rate, collections=self.model_collection)
            self.tf_summary_train_loss = tf.summary.scalar('loss', self.train.tf_loss, collections=self.model_collection)

            tf.summary.image('input/batch_data', self.train.tf_batch_data, max_outputs=1, collections=self.model_collection)
            # tf.summary.image('input/batch_data_uint8', self.train.tf_batch_data_uint8, max_outputs=1, collections=self.model_collection)
            tf.summary.image('input/batch_labels', self.train.tf_batch_labels, max_outputs=1, collections=self.model_collection)

            tf.summary.image('output/batch_pred', self.train.tf_pred, max_outputs=1, collections=self.model_collection)

        with tf.name_scope("Valid"):
            tf.summary.scalar('loss', self.valid.tf_loss, collections=self.model_collection)

            tf.summary.image('input/image', tf.cast(self.valid.tf_image, tf.float32), max_outputs=1, collections=self.model_collection)
            tf.summary.image('input/depth', tf.cast(self.valid.tf_depth, tf.float32), max_outputs=1, collections=self.model_collection)
            tf.summary.image('input/image_resized', self.valid.tf_image_resized, max_outputs=1, collections=self.model_collection)
            tf.summary.image('input/depth_resized', self.valid.tf_depth_resized, max_outputs=1, collections=self.model_collection)

            tf.summary.image('output/pred', self.valid.fcrn.get_output(), max_outputs=1, collections=self.model_collection)

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

    def saveResults(self, datetime, epoch, max_epochs, step, max_steps, sim_train):
        # Logs the obtained simulation results
        print("[Results] Logging simulation info to 'results.txt' file...")

        root_path = os.path.abspath(os.path.join(__file__, "../.."))
        relative_path = 'results.txt'
        save_file_path = os.path.join(root_path, relative_path)

        f = open(save_file_path, 'a')
        f.write("%s\t\t%s\t\t%s\t\t%s\t\tepoch: %d/%d\t\tstep: %d/%d\ttrain_loss: %f\tvalid_loss: %f\tt: %f s\n" % (
            datetime, self.args.model_name, self.args.dataset, self.loss_name, epoch, max_epochs, step, max_steps, self.train.loss, self.valid.loss,
            sim_train))
        f.close()
