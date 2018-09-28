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

# In ROS, nodes are uniquely named. If two nodes with the same name are launched, the previous one is kicked off. The
# anonymous=True flag means that rospy will choose a unique name for our 'listener' node so that multiple listeners can
# run simultaneously.
class Listener:
    def __init__(self, pub_pred_up_32FC1, pub_pred_camera_info):
        self.rate = rospy.Rate(10)  # 10hz

        # ------------- #
        #  Subscribers  #
        # ------------- #
        rospy.Subscriber('/pred/depth_32FC1', Image, self.callback_depth_32FC1, (pub_pred_up_32FC1, pub_pred_camera_info, self.rate))
        rospy.Subscriber('/kitti/camera_color_left/camera_info', CameraInfo, self.callback_camera_info)

        # spin() simply keeps python from exiting until this node is stopped
        rospy.spin()

    def callback_depth_32FC1(self, depth_32FC1_msg, args):
        rospy.loginfo("'depth_32FC1' message received!")

        # rospy.loginfo(rospy.get_caller_id() + 'I heard %s', data.data)
        cv_depth_raw = bridge.imgmsg_to_cv2(depth_32FC1_msg, desired_encoding="passthrough")

        print('depth_32FC!_msg.encoding:', depth_32FC1_msg.encoding)
        print('cv_depth_raw:', cv_depth_raw.shape, cv_depth_raw.dtype)
        input("aki")

        try:
            Talker.convertFloatType(cv_depth_raw, pub_pred_up_32FC1=args[0], pub_pred_camera_info=args[1], rate=args[2],
                                    rec_camera_info_msg=self.rec_camera_info_msg)
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


class Talker:
    def __init__(self):
        # ------------ #
        #  Publishers  #
        # ------------ #
        self.pub_pred_up_32FC1_new = rospy.Publisher('/pred/depth_32FC1_new', Image, queue_size=10)
        self.pub_pred_camera_info = rospy.Publisher('/pred/camera_info_new', CameraInfo, queue_size=10)

    @staticmethod
    def convertFloatType(depth_raw, pub_pred_up_32FC1, pub_pred_camera_info, rate, rec_camera_info_msg):

        # CV2 Image -> ROS Image Message
        depth_new_msg = bridge.cv2_to_imgmsg(depth_raw, encoding="mono16")

        # Publish!
        depth_new_msg.header = rec_camera_info_msg.header

        pub_pred_up_32FC1.publish(depth_new_msg)
        pub_pred_camera_info.publish(rec_camera_info_msg)

        rospy.loginfo("convertFloatType")
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
        listener = Listener(talker.pub_pred_up_32FC1_new, talker.pub_pred_camera_info)

    except rospy.ROSInterruptException:
        pass
