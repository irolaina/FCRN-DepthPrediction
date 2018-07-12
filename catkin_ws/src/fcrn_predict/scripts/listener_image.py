#!/usr/bin/env python
# Software License Agreement (BSD License)
#
# Copyright (c) 2008, Willow Garage, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of Willow Garage, Inc. nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Revision $Id$

## Simple talker demo that listens to std_msgs/Strings published 
## to the 'chatter' topic

import argparse
import cv2
import rospy
import numpy as np
import tensorflow as tf
import os
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from modules.third_party.laina.fcrn import ResNet50UpProj

def argumentHandler():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, help="Select which gpu to run the code", default='0')
    parser.add_argument('-r', '--model_path', help='Converted parameters for the model', default='/home/nicolas/MEGA/workspace/FCRN-DepthPrediction/tensorflow/output/fcrn/kitticontinuous/all_px/mse/2018-06-29_13-52-58/restore/model.fcrn')
    parser.add_argument('-i', '--video_path', help='Directory of images to predict')
    return parser.parse_args()

args = argumentHandler()
bridge = CvBridge()

# # ===================
# #  Class Declaration
# # ===================
# class ImportGraph(object):
#     """  Importing and running isolated TF graph """
#     def __init__(self, restore_path):
#         # Local Variables
#         restore_filepath = None
#
#         # Get the Path of the Model to be Restored
#         restore_files = os.listdir(restore_path)
#         restore_files_ext = []
#         for i in range(len(restore_files)):
#             restore_files_ext.append(restore_files[i].split('.')[-1])
#
#         extensions = ['checkpoint', 'index', 'meta', 'data-00000-of-00001']
#         if len([i for i in extensions if i in restore_files_ext]) == len(extensions):
#             pass
#         else:
#             raise FileExistsError
#
#         for file in restore_files:
#             if not file.find("model"):
#                 model_fileName = os.path.splitext(file)[0]
#                 restore_filepath = restore_path + model_fileName  # Path to file with extension *.ckpt
#                 break
#
#         # Creates local graph and use it in the session
#         self.graph = tf.Graph()
#         self.sess = tf.Session(graph=self.graph)
#         with self.graph.as_default():
#             # Import saved model from location 'restore_filepath' into local graph
#             print('\n[Network/Restore] Restoring model from file: %s' % restore_filepath)
#             saver = tf.train.import_meta_graph(restore_filepath + '.meta', clear_devices=True)
#             saver.restore(self.sess, restore_filepath)
#             print("[Network/Restore] Model restored!")
#             print("[Network/Restore] Restored variables:\n", tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), '\n')
#
#             # Gets activation function from saved collection
#             # You may need to change this in case you name it differently
#             self.prediction = tf.get_collection('prediction')[0]
#             self.inputs = tf.get_collection('inputs')[0]
#             self.keep_prob = tf.get_collection('keep_prob')[0]
#             self.recursivity = tf.get_collection('recursivity')[0]  # Tensor Variable
#
#             # Gets recursivity value. Overwrites Tensor("Input/recursivity:0", shape=(), dtype=int32) Variable!
#             self.recursivity = self.recursivity.eval(session=self.sess)  # Int value
#
#     def networkPredict(self, netInput, numClasses):
#         """ Running the activation function previously imported """
#         # The 'inputs' corresponds to name of input placeholder
#         data = np.expand_dims(np.expand_dims(netInput, 0), 2)  # (idx, numInputs, 1) - Normalized
#         feed_dict = {self.inputs: data, self.keep_prob: 1.0}
#
#         # ----- Session Run! ----- #
#         output = self.sess.run(self.prediction, feed_dict=feed_dict)
#         output_round = (np.arange(numClasses) == np.argmax(output)).astype(np.float32)
#         # -----
#
#         # Debug
#         print(output, output_round)
#
#         return output, output_round

###############################################################################
print(args.model_path)
print(args.video_path)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

# Load from Camera or Video
# cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture(args.video_path)

# if not cap.isOpened():  # Check if it succeeded
#     print("It wasn't possible to open the camera.")
#     return -1

# ----------------
#  Building Graph
# ----------------
with tf.Graph().as_default() as graph:
    # Default input size
    height, width, channels = 228, 304, 3
    batch_size = 1

    # Create a placeholder for the input image
    input_node = tf.placeholder(tf.uint8, shape=(height, width, channels))
    # tf_image_float32 = tf.cast(input_node, tf.float32)
    tf_image_float32 = tf.image.convert_image_dtype(input_node, tf.float32)

    with tf.variable_scope('model'):  # Disable for running original models!!!
        # Construct the network
        net = ResNet50UpProj({'data': tf.expand_dims(tf_image_float32, axis=0)}, batch=batch_size, keep_prob=1,
                             is_training=False)

    tf_pred = net.get_output()

def talker(frame, pub_string, pub_pred, rate):
    # ---------------
    #  Running Graph
    # ---------------
    with tf.Session(graph=graph) as sess:
        # Load the converted parameters
        print('\nLoading the model...')

        # Use to load from ckpt file
        saver = tf.train.Saver()
        saver.restore(sess, args.model_path)

        # Use to load from npy file
        # net.load(args.model_path, sess)

        count = 0

        # Capture frame-by-frame
        # success, frame = cap.read()
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_NEAREST)
        pred = sess.run(tf_pred, feed_dict={input_node: frame})

        # Image Processing
        pred_uint8 = cv2.convertScaleAbs(pred[0])
        pred_uint8_scaled = cv2.convertScaleAbs(pred[0] * (255 / np.max(pred[0])))
        pred = pred_uint8_scaled

        image_message = bridge.cv2_to_imgmsg(pred, encoding="passthrough")
        cv2.imshow('frame', frame)
        cv2.imshow('pred', pred)

        hello_str = "hello world %s" % rospy.get_time()
        rospy.loginfo(hello_str)
        pub_string.publish(hello_str)
        pub_pred.publish(image_message)
        rate.sleep()

        if cv2.waitKey(1) & 0xFF == ord('q'):  # without waitKey() the images are not shown.
            return 0

cv_image = None

def callback(received_image_msg, args):
    # rospy.loginfo(rospy.get_caller_id() + 'I heard %s', data.data)
    global cv_image
    cv_image = bridge.imgmsg_to_cv2(received_image_msg, desired_encoding="passthrough")

    cv2.imshow("cv_image", cv_image)

    try:
        talker(cv_image, pub_string=args[0], pub_pred=args[1], rate=args[2])
    except rospy.ROSInterruptException:
        pass

    if cv2.waitKey(1) & 0xFF == ord('q'):  # without waitKey() the images are not shown.
        return 0

def listener():
    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.

    rospy.init_node('listener', anonymous=True)
    # rospy.init_node('talker', anonymous=True)
    pub_string = rospy.Publisher('chatter', String, queue_size=10)
    pub_pred = rospy.Publisher('chatter_pred', Image, queue_size=10)
    rate = rospy.Rate(10)  # 10hz

    # model = ImportGraph(args.model_path)

    rospy.Subscriber('/kitti/camera_color_left/image_raw', Image, callback, (pub_string, pub_pred, rate))

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()
