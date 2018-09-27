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
import image_geometry

from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo, PointCloud2


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

camModel = image_geometry.PinholeCameraModel()


# In ROS, nodes are uniquely named. If two nodes with the same name are launched, the previous one is kicked off. The
# anonymous=True flag means that rospy will choose a unique name for our 'listener' node so that multiple listeners can
# run simultaneously.
class Listener:
    def __init__(self, pub_predCloud):
        self.rate = rospy.Rate(10)  # 10hz

        # ------------- #
        #  Subscribers  #
        # ------------- #
        rospy.Subscriber('/pred/image', Image, self.callback_pred_image, (pub_predCloud, self.rate))
        rospy.Subscriber('/kitti/camera_color_left/camera_info', CameraInfo, self.callback_camera_info)

        # Sync Topics
        # image_raw_sub = message_filters.Subscriber('/kitti/camera_color_left/image_raw', Image)
        # pred_image_sub = message_filters.Subscriber('/pred/image', Image)
        # ts = message_filters.TimeSynchronizer([image_raw_sub, pred_image_sub], 10)
        # ts.registerCallback(callback, (pub_string, pub_predCloud, rate))

        # spin() simply keeps python from exiting until this node is stopped
        rospy.spin()

    @staticmethod
    def callback_pred_image(pred_image_msg, args):
        # rospy.loginfo(rospy.get_caller_id() + 'I heard %s', data.data)
        cv_pred_image = bridge.imgmsg_to_cv2(pred_image_msg, desired_encoding="passthrough")

        try:
            Talker.pred2cloud(cv_pred_image, pub_predCloud=args[0], rate=args[1])
        except rospy.ROSInterruptException:
            pass

        # FIXME:
        if cv2.waitKey(1) & 0xFF == ord('q'):  # without waitKey() the images are not shown.
            return 0

    @staticmethod
    def callback_camera_info(received_camera_info_msg):
        camModel.fromCameraInfo(received_camera_info_msg)
        # print(camModel)
        print("K:\n{}".format(camModel.intrinsicMatrix()))
        # input("oi")


class Talker:
    def __init__(self):
        # ------------ #
        #  Publishers  #
        # ------------ #
        self.pub_predCloud = rospy.Publisher('/pred2cloud/cloud', PointCloud2, queue_size=10)

    @staticmethod
    def pred2cloud(cv_pred_image, pub_predCloud, rate):
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

        def reconstruct():  # TODO: Move
            # Compute world coordinates from the disparity image
            depthmap_height, depthmap_width = cv_pred_image.shape[0], cv_pred_image.shape[1]

            fx, fy = 100.0, 100.0  # TODO: Change Value
            cx, cy = 1.0, 1.0  # TODO: Change Value

            print("Depth map size = {}x{}".format(depthmap_width, depthmap_height))

            camera_params = np.array([[fx, 0, cx],
                                      [0, fy, cy],
                                      [0, 0, 1]])

            print("Camera intrinsic matrix:\n{}".format(camera_params))
            print("Reconstructing...")
            # print(image_geometry.intrinsicMatrix())

            # threeDImage = cv2.rgbd.depthTo3d()(cv_pred_image, Q)
            # print(threeDImage.shape)

        # reconstruct()

        # Display Images using OpenCV
        cv2.imshow("/pred/image", cv_pred_image)

        # Publish!
        hello_str = "pred2cloud"
        rospy.loginfo(hello_str)
        # pub_cloud.publish(cloud_message) # TODO:

        # Mandatory code for the ROS node and OpenCV structures.
        rate.sleep()

        # FIXME:
        if cv2.waitKey(1) & 0xFF == ord('q'):  # without waitKey() the images are not shown.
            return 0


if __name__ == '__main__':
    # Initialize the node and name it.
    rospy.init_node('pred2cloud', anonymous=True)

    try:
        talker = Talker()
        listener = Listener(talker.pub_predCloud)

    except rospy.ROSInterruptException:
        pass
