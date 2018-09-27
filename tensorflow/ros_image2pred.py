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
import image_geometry
import sensor_msgs

from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo

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

camModel = image_geometry.PinholeCameraModel()


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
            self.tf_pred_up_16UC1 = tf.image.convert_image_dtype(self.tf_pred_up, tf.uint16)

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
    def __init__(self, pub_pred_up_8UC1, pub_pred_up_32FC1, pub_pred_camera_info):
        self.rate = rospy.Rate(10)  # 10hz
        self.net = Network()

        # ------------- #
        #  Subscribers  #
        # ------------- #
        rospy.Subscriber('/kitti/camera_color_left/image_raw', Image, self.callback_image_raw,
                         (self.net, pub_pred_up_8UC1, pub_pred_up_32FC1, pub_pred_camera_info, self.rate))
        rospy.Subscriber('/kitti/camera_color_left/camera_info', CameraInfo, self.callback_camera_info)

        # spin() simply keeps python from exiting until this node is stopped
        rospy.spin()

    def callback_image_raw(self, image_raw_msg, args):
        rospy.loginfo("'image_raw' message received!")

        # rospy.loginfo(rospy.get_caller_id() + 'I heard %s', data.data)
        cv_image_raw = bridge.imgmsg_to_cv2(image_raw_msg, desired_encoding="passthrough")

        print('image_raw_msg.encoding:', image_raw_msg.encoding)
        print('cv_image_raw:', cv_image_raw.shape, cv_image_raw.dtype)

        try:
            Talker.image2pred(cv_image_raw, net=args[0], pub_pred_up_8UC1=args[1], pub_pred_up_32FC1=args[2],
                              pub_pred_camera_info=args[3], rate=args[4],
                              received_camera_info_msg=self.received_camera_info_msg)
        except rospy.ROSInterruptException:
            pass

        # FIXME:
        if cv2.waitKey(1) & 0xFF == ord('q'):  # without waitKey() the images are not shown.
            # rospy.shutdown()
            return 0

    def callback_camera_info(self, received_camera_info_msg):
        rospy.loginfo("'camera_info' message received!")

        self.received_camera_info_msg = received_camera_info_msg
        camModel.fromCameraInfo(received_camera_info_msg)
        # print(camModel)
        # print("K:\n{}".format(camModel.intrinsicMatrix()))


class Talker:
    def __init__(self):
        # ------------ #
        #  Publishers  #
        # ------------ #
        self.pub_pred_up_8UC1 = rospy.Publisher('/pred/image_8UC1', Image, queue_size=10)
        self.pub_pred_up_32FC1 = rospy.Publisher('/pred/image_32FC1', Image, queue_size=10)
        self.pub_pred_camera_info = rospy.Publisher('/pred/camera_info', CameraInfo, queue_size=10)

    @staticmethod
    def image2pred(image_raw, net, pub_pred_up_8UC1, pub_pred_up_32FC1, pub_pred_camera_info, rate,
                   received_camera_info_msg):
        # Capture/Predict frame-by-frame
        image = cv2.resize(image_raw, (net.width, net.height), interpolation=cv2.INTER_AREA)
        feed_pred = {net.input_node: image, net.input_shape: image_raw.shape}
        pred, pred_up, pred_up_16UC1 = net.sess.run([net.tf_pred, net.tf_pred_up, net.tf_pred_up_16UC1],
                                                 feed_dict=feed_pred)  # pred_up: ((1, 375, 1242, 1), dtype('float32'))

        # Image Processing
        pred_8UC1_scaled = cv2.convertScaleAbs(pred[0] * (255 / np.max(pred[0])))
        pred_up_8UC1_scaled = cv2.convertScaleAbs(pred_up[0] * (255 / np.max(pred_up[0])))

        # CV2 Image -> ROS Image Message
        # pred_8UC1_msg = bridge.cv2_to_imgmsg(pred_8UC1_scaled, encoding="passthrough")
        pred_up_8UC1_msg = bridge.cv2_to_imgmsg(pred_up_8UC1_scaled, encoding="passthrough")
        # pred_up_8UC1_msg = bridge.cv2_to_imgmsg(pred_up_8UC1_scaled, encoding="bgr8")
        # pred_up_32FC1_msg = bridge.cv2_to_imgmsg(pred_up, encoding="passthrough")
        # pred_up_32FC1_msg = bridge.cv2_to_imgmsg(pred_up_16UC1, encoding="passthrough")
        # pred_up_32FC1_msg = bridge.cv2_to_imgmsg(pred_up_16UC1, encoding="mono8")
        pred_up_32FC1_msg = bridge.cv2_to_imgmsg(pred_up_16UC1)

        print(pred_up_8UC1_msg.encoding)
        print(pred_up_32FC1_msg.encoding)

        # Display Images using OpenCV
        cv2.imshow("image_raw", image_raw)
        cv2.imshow('image', image)
        cv2.imshow('pred', pred_8UC1_scaled)
        cv2.imshow('pred_up', pred_up_8UC1_scaled)

        # Publish!
        pred_up_8UC1_msg.header = received_camera_info_msg.header
        pred_up_32FC1_msg.header = received_camera_info_msg.header

        pub_pred_up_8UC1.publish(pred_up_8UC1_msg)
        pub_pred_up_32FC1.publish(pred_up_32FC1_msg)
        pub_pred_camera_info.publish(received_camera_info_msg)

        rospy.loginfo("image2pred")
        print('---')

        # Mandatory code for the ROS node and OpenCV structures.
        rate.sleep()

        # FIXME:
        if cv2.waitKey(1) & 0xFF == ord('q'):  # without waitKey() the images are not shown.
            # return 0
            rospy.shutdown()


if __name__ == '__main__':
    # Initialize the node and name it.
    rospy.init_node('image2pred', anonymous=True)

    try:
        talker = Talker()
        listener = Listener(talker.pub_pred_up_8UC1, talker.pub_pred_up_32FC1, talker.pub_pred_camera_info)

    except rospy.ROSInterruptException:
        pass
