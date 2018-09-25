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

# Simple talker demo that listens to std_msgs/Strings published to the 'chatter' topic

import argparse
import os

import cv2
import numpy as np
import rospy
import tensorflow as tf
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String
from sensor_msgs.msg import PointCloud2

from modules.third_party.laina.fcrn import ResNet50UpProj


def argumentHandler():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, help="Select which gpu to run the code", default='0')
    parser.add_argument('-r', '--model_path', help='Converted parameters for the model',
                        default='/home/nicolas/MEGA/workspace/FCRN-DepthPrediction/tensorflow/output/fcrn/kitticontinuous/all_px/mse/2018-06-29_13-52-58/restore/model.fcrn')
    parser.add_argument('-i', '--video_path', help='Directory of images to predict')
    return parser.parse_args()


args = argumentHandler()
bridge = CvBridge()

print(args.model_path)
print(args.video_path)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

def talker(image_raw, pub_string, pub_predCloud, rate):
    # # Capture frame-by-frame
    # image = cv2.resize(image_raw, (net.width, net.height), interpolation=cv2.INTER_AREA)
    # _, pred_up = net.sess.run([net.tf_pred, net.tf_pred_up],
    #                              feed_dict={net.input_node: image, net.input_shape: image_raw.shape})
    #
    # # Image Processing
    # # pred_uint8_scaled = cv2.convertScaleAbs(pred[0] * (255 / np.max(pred[0])))
    # # image_message = bridge.cv2_to_imgmsg(pred_uint8_scaled, encoding="passthrough")
    #
    # pred_up_uint8_scaled = cv2.convertScaleAbs(pred_up[0] * (255 / np.max(pred_up[0])))
    # image_message = bridge.cv2_to_imgmsg(pred_up_uint8_scaled, encoding="passthrough")

    # cv2.imshow("image_raw", image_raw)
    # cv2.imshow('image', image)
    # cv2.imshow('pred', pred_uint8_scaled)
    # cv2.imshow('pred_up', pred_up_uint8_scaled)

    hello_str = "hello world %s" % rospy.get_time()
    rospy.loginfo(hello_str)
    pub_string.publish(hello_str)

    # Compute world coordinates from the disparity image
    # cv::Mat    XYZ(disparity32F.size(), CV_32FC3);
    # reprojectImageTo3D(disparity32F, XYZ, Q, false, CV_32F);
    # print_3D_points(disparity32F, XYZ);

    rate.sleep()

    if cv2.waitKey(1) & 0xFF == ord('q'):  # without waitKey() the images are not shown.
        return 0


def callback(received_image_msg, args):
    # rospy.loginfo(rospy.get_caller_id() + 'I heard %s', data.data)
    cv_image = bridge.imgmsg_to_cv2(received_image_msg, desired_encoding="passthrough")

    try:
        talker(cv_image, pub_string=args[0], pub_predCloud=args[1], rate=args[2])
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
    pub_string = rospy.Publisher('/pred2cloud/string', String, queue_size=10)
    pub_predCloud = rospy.Publisher('/pred2cloud/cloud', PointCloud2, queue_size=10)
    rate = rospy.Rate(10)  # 10hz

    # model = ImportGraph(args.model_path)

    rospy.Subscriber('/kitti/camera_color_left/image_raw', Image, callback, (pub_string, pub_predCloud, rate))

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


if __name__ == '__main__':
    listener()
