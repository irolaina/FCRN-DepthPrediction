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

from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo

from modules.third_party.laina.fcrn import ResNet50UpProj


# ===========
#  Functions
# ===========
def argumentHandler():
    # Parse arguments
    parser = argparse.ArgumentParser()
    # Enable if will be run through .launch file
    # parser.add_argument('__name', type=str, help="ROS Node Name", default='')
    # parser.add_argument('__log', type=str, help="ROS Node Log path", default='')
    parser.add_argument('--gpu', type=str, help="Select which gpu to run the code", default='0')
    parser.add_argument('-r', '--model_path', help='Converted parameters for the model',
                        default='/home/nicolas/MEGA/workspace/FCRN-DepthPrediction/tensorflow/output/fcrn/kitti_continuous/all_px/berhu/2019-03-18_10-02-45/restore/model.fcrn')
    parser.add_argument('-i', '--video_path', help='Directory of images to predict')
    return parser.parse_args()


def convert2uint8(src):
    return cv2.convertScaleAbs(src * (255 / np.max(src)))


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
                net = ResNet50UpProj({'data': tf.expand_dims(tf_image_float32, axis=0)}, batch=batch_size, keep_prob=1.0,
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


class Talker(object):
    def __init__(self, *args, **kwargs):
        super(Talker, self).__init__(*args, **kwargs)

        # ------------ #
        #  Publishers  #
        # ------------ #
        self.pub_pred_up_8UC1 = rospy.Publisher('/pred_depth/image_8UC1', Image, queue_size=10)
        self.pub_pred_up_32FC1 = rospy.Publisher('/pred_depth/image_32FC1', Image, queue_size=10)
        self.pub_pred_camera_info = rospy.Publisher('/pred_depth/camera_info', CameraInfo, queue_size=10)

    def image2pred(self, image_raw, net):
        # Capture/Predict frame-by-frame
        image = cv2.resize(image_raw, (net.width, net.height), interpolation=cv2.INTER_AREA)
        feed_pred = {net.input_node: image, net.input_shape: image_raw.shape}
        pred, pred_up = net.sess.run([net.tf_pred, net.tf_pred_up],
                                     feed_dict=feed_pred)  # pred_up: ((1, 375, 1242, 1), dtype('float32'))

        # Image Processing
        pred_8UC1_scaled = convert2uint8(pred[0])
        pred_up_8UC1_scaled = convert2uint8(pred_up[0])

        # CV2 Image -> ROS Image Message
        pred_up_8UC1_msg = bridge.cv2_to_imgmsg(pred_up_8UC1_scaled, encoding="passthrough")
        pred_up_32FC1_msg = bridge.cv2_to_imgmsg(pred_up[0], encoding="passthrough")

        print(pred_up_8UC1_msg.encoding)
        print(pred_up_32FC1_msg.encoding)

        # Publish!
        pred_up_8UC1_msg.header = self.rec_camera_info_msg.header
        pred_up_32FC1_msg.header = self.rec_camera_info_msg.header

        self.pub_pred_up_8UC1.publish(pred_up_8UC1_msg)
        self.pub_pred_up_32FC1.publish(pred_up_32FC1_msg)
        self.pub_pred_camera_info.publish(self.rec_camera_info_msg)

        rospy.loginfo("image2pred")
        print('---')

        # Display Images using OpenCV
        cv2.imshow("image_raw", image_raw)
        cv2.imshow('image', image)
        cv2.imshow('pred', pred_8UC1_scaled)
        cv2.imshow('pred_up', pred_up_8UC1_scaled)

        # Mandatory code for the ROS node and OpenCV structures.
        self.rate.sleep()

        # FIXME:
        if cv2.waitKey(1) & 0xFF == ord('q'):  # without waitKey() the images are not shown.
            # return 0
            rospy.shutdown()


# In ROS, nodes are uniquely named. If two nodes with the same name are launched, the previous one is kicked off. The
# anonymous=True flag means that rospy will choose a unique name for our 'listener' node so that multiple listeners can
# run simultaneously.
class Listener(Talker):
    def __init__(self, *args, **kwargs):
        super(Listener, self).__init__(*args, **kwargs)

        self.rate = rospy.Rate(10)  # 10hz
        self.net = Network()

        # ------------- #
        #  Subscribers  #
        # ------------- #
        rospy.Subscriber('/kitti/camera_color_left/image_raw', Image, self.callback_image_raw, self.net)
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
            self.image2pred(cv_image_raw, net=args)
        except rospy.ROSInterruptException:
            pass

        # FIXME:
        if cv2.waitKey(1) & 0xFF == ord('q'):  # without waitKey() the images are not shown.
            # rospy.shutdown()
            return 0

    def callback_camera_info(self, rec_camera_info_msg):
        rospy.loginfo("'camera_info' message received!")

        self.rec_camera_info_msg = rec_camera_info_msg
        camModel.fromCameraInfo(rec_camera_info_msg)
        # print(camModel)
        # print("K:\n{}".format(camModel.intrinsicMatrix()))


if __name__ == '__main__':
    # Initialize the node and name it.
    rospy.init_node('image2pred', anonymous=True)

    try:
        listener = Listener()

    except rospy.ROSInterruptException:
        pass
