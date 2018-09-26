#!/usr/bin/python2
# -*- coding: utf-8 -*-


# ===========
#  Libraries
# ===========
import argparse
import os

import cv2
import numpy as np
import rospy
import tensorflow as tf

from cv_bridge import CvBridge
from std_msgs.msg import String
from sensor_msgs.msg import Image

from modules.third_party.laina.fcrn import ResNet50UpProj


# ===========
#  Functions
# ===========
def argumentHandler():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, help="Select which gpu to run the code", default='0')
    parser.add_argument('-r', '--model_path', help='Converted parameters for the model',
                        default='/home/nicolas/MEGA/workspace/FCRN-DepthPrediction/tensorflow/output/fcrn/kitticontinuous/all_px/mse/2018-06-29_13-52-58/restore/model.fcrn')
    parser.add_argument('-i', '--video_path', help='Directory of images to predict')
    return parser.parse_args()


# ==================
#  Global Variables
# ==================
args = argumentHandler()
bridge = CvBridge()

print(args.model_path)
print(args.video_path)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


class Network(object):
    def __init__(self):
        # ----------------
        #  Building Graph
        # ----------------
        with tf.Graph().as_default() as self.graph:
            self.sess = tf.InteractiveSession()

            # Default input size
            self.height, self.width, channels = 228, 304, 3
            batch_size = 1

            # Create a placeholder for the input image
            self.input_node = tf.placeholder(tf.uint8, shape=(self.height, self.width, channels))
            self.input_shape = tf.placeholder(tf.int32, shape=3)
            # tf_image_float32 = tf.cast(input_node, tf.float32)
            tf_image_float32 = tf.image.convert_image_dtype(self.input_node, tf.float32)

            with tf.variable_scope('model'):  # Disable for running original models!!!
                # Construct the network
                net = ResNet50UpProj({'data': tf.expand_dims(tf_image_float32, axis=0)}, batch=batch_size, keep_prob=1,
                                     is_training=False)

            self.tf_pred = net.get_output()
            self.tf_pred_up = tf.image.resize_images(self.tf_pred, self.input_shape[:2], tf.image.ResizeMethod.BILINEAR,
                                                     align_corners=True)

            # --------------------------
            #  Restore Graph Parameters
            # --------------------------
            # Load the converted parameters
            print('\nLoading the model...')

            # Use to load from ckpt file
            saver = tf.train.Saver()
            saver.restore(self.sess, args.model_path)

            # Use to load from npy file
            # net.load(args.model_path, self.sess)


# In ROS, nodes are uniquely named. If two nodes with the same name are launched, the previous one is kicked off. The
# anonymous=True flag means that rospy will choose a unique name for our 'listener' node so that multiple listeners can
# run simultaneously.
class Listener:
    def __init__(self):
        self.rate = rospy.Rate(10)  # 10hz
        self.net = Network()

        # ------------ #
        #  Publishers  #
        # ------------ #
        self.pub_string = rospy.Publisher('pred/string', String, queue_size=10)
        self.pub_pred = rospy.Publisher('pred/image', Image, queue_size=10)

        # ------------- #
        #  Subscribers  #
        # ------------- #
        rospy.Subscriber('/kitti/camera_color_left/image_raw', Image, self.callback,
                         (self.pub_string, self.pub_pred, self.rate, self.net))

        # spin() simply keeps python from exiting until this node is stopped
        rospy.spin()

    @staticmethod
    def callback(received_image_msg, args):
        # rospy.loginfo(rospy.get_caller_id() + 'I heard %s', data.data)
        cv_image = bridge.imgmsg_to_cv2(received_image_msg, desired_encoding="passthrough")

        try:
            Talker.image2pred(cv_image, pub_string=args[0], pub_pred=args[1], rate=args[2], net=args[3])
        except rospy.ROSInterruptException:
            pass

        if cv2.waitKey(1) & 0xFF == ord('q'):  # without waitKey() the images are not shown.
            return 0


class Talker:
    def __init__(self):
        pass

    @staticmethod
    def image2pred(image_raw, pub_string, pub_pred, rate, net):
        # Capture frame-by-frame
        image = cv2.resize(image_raw, (net.width, net.height), interpolation=cv2.INTER_AREA)
        _, pred_up = net.sess.run([net.tf_pred, net.tf_pred_up],
                                  feed_dict={net.input_node: image, net.input_shape: image_raw.shape})

        # Image Processing
        # pred_uint8_scaled = cv2.convertScaleAbs(pred[0] * (255 / np.max(pred[0])))
        # image_message = bridge.cv2_to_imgmsg(pred_uint8_scaled, encoding="passthrough")

        pred_up_uint8_scaled = cv2.convertScaleAbs(pred_up[0] * (255 / np.max(pred_up[0])))
        image_message = bridge.cv2_to_imgmsg(pred_up_uint8_scaled, encoding="passthrough")

        # cv2.imshow("image_raw", image_raw)
        # cv2.imshow('image', image)
        # cv2.imshow('pred', pred_uint8_scaled)
        # cv2.imshow('pred_up', pred_up_uint8_scaled)

        hello_str = "image2pred"
        rospy.loginfo(hello_str)
        pub_string.publish(hello_str)
        pub_pred.publish(image_message)
        rate.sleep()

        if cv2.waitKey(1) & 0xFF == ord('q'):  # without waitKey() the images are not shown.
            return 0


if __name__ == '__main__':
    # Initialize the node and name it.
    rospy.init_node('listener', anonymous=True)

    try:
        listener = Listener()

    except rospy.ROSInterruptException:
        pass
