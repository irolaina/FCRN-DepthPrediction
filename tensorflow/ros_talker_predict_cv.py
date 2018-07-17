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

## Simple talker demo that published std_msgs/Strings messages
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

def talker():
    args = argumentHandler()

    pub_string = rospy.Publisher('chatter', String, queue_size=10)
    pub_pred = rospy.Publisher('chatter_pred', Image, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(10) # 10hz

    bridge = CvBridge()

    ###############################################################################
    print(args.model_path)
    print(args.video_path)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # Load from Camera or Video
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(args.video_path)

    if not cap.isOpened():  # Check if it succeeded
        print("It wasn't possible to open the camera.")
        return -1

    # ----------------
    #  Building Graph
    # ----------------
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

    # ---------------
    #  Running Graph
    # ---------------
    with tf.Session() as sess:
        # Load the converted parameters
        print('\nLoading the model...')

        # Use to load from ckpt file
        saver = tf.train.Saver()
        saver.restore(sess, args.model_path)

        # Use to load from npy file
        # net.load(args.model_path, sess)

        while not rospy.is_shutdown():
            # Capture frame-by-frame
            _, frame = cap.read()
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            pred = sess.run(tf_pred, feed_dict={input_node: frame})

            # Image Processing
            pred_uint8_scaled = cv2.convertScaleAbs(pred[0] * (255 / np.max(pred[0])))
            img = pred_uint8_scaled
            ###############################################################################

            image_message = bridge.cv2_to_imgmsg(img, encoding="passthrough")
            cv2.imshow('img', img)

            hello_str = "hello world %s" % rospy.get_time()
            rospy.loginfo(hello_str)
            pub_string.publish(hello_str)
            pub_pred.publish(image_message)
            rate.sleep()

            if cv2.waitKey(1) & 0xFF == ord('q'):  # without waitKey() the images are not shown.
                break

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
