# ROS Bag and Network Prediction Integration

1.) Generate ROS Bag file using Kitti2bag:
```shell
python2 kitti2bag.py -t 2011_09_26 -r 0001 raw_synced
```

2.) Run the following ROS nodes:

Recommended:

```shell
roslaunch fcrn image2pred.launch manager:=nodelet_manager
```

2.1) Init ROS

```shell
roscore
```

2.2) Init ROSBag node

```shell
cd ~/MEGA/workspace/kitti2bag
rosbag play -l kitti_2011_09_26_drive_0001_synced.bag
```

2.3) Init Network Prediction node (image2pred)

```shell
cd ~/MEGA/workspace/FCRN-DepthPrediction/ros_catkin_ws/src/fcrn/src/scripts
python2 ros_image2pred.py
```

2.4) Init RViz: ROS 3D Robot Visualizer

```shell
rviz
```

| [Global Options] |                                     |
|------------------|-------------------------------------|
| Fixed Frame:     | velo_link                           |

| [Grid]           |                                     |
|------------------|-------------------------------------|
| Reference Frame: | base_link                           |
| [PointCloud2]    |                                     |
| Topic:           | /kitti/velo/pointcloud              |
| [Image]          |                                     |
| Image Topic:     | /kitti/camera_color_left/image_raw  |
| [Image]          |                                     |
| Image Topic:     | /kitti/camera_color_right/image_raw |
| [Image]          |                                     |
| Image Topic:     | /kitti/camera_gray_left/image_raw   |
| [Image]          |                                     |
| Image Topic:     | /kitti/camera_gray_right/image_raw  |
| [Image]          |                                     |
| Image Topic:     | /pred_depth/image_8UC1              |
| [PointCloud2]    |                                     |
| Topic:           | /pred_depth/cloud                   |

2.5) Init '[depth_image_proc](http://wiki.ros.org/depth_image_proc#depth_image_proc.2BAC8-point_cloud_xyz)' nodelet (depth2cloud).

Command: rosrun nodelet nodelet load <pkg_name>/<nodeletclass_name> <manager_name>

```shell
rosrun nodelet nodelet manager __name:=nodelet_manager
```

```shell
rosrun nodelet nodelet load depth_image_proc/point_cloud_xyz nodelet_manager __name:=nodelet_depth camera_info:=/pred_depth/camera_info image_rect:=/pred_depth/image_32FC1 points:=/pred_depth/cloud
```

Optional:     

```shell
rostopic list
rosnode list
rostopic echo <topic>
rostopic type <topic>
rqt_graph
roslaunch <package_name> <file.launch>
```

|Terminator         |                                    |
|-------------------|------------------------------------|
|      `roscore`    |        `ros_image2pred.py`         |
|      `rosbag`     |            `empty`                 |
|       `rviz`      | `depth_image_proc/point_cloud_xyz` |
| `nodelet_manager` |      `rostopic echo <topic>`       |


Helpful Links:

[Creating a ROS Package](http://wiki.ros.org/catkin/Tutorials/CreatingPackage)

[Writing a .launch file](http://www.clearpathrobotics.com/assets/guides/ros/Launch%20Files.html#writing-a-launch-file)

[Display Image - Code Walkthrough - sdk-wiki](http://sdk.rethinkrobotics.com/wiki/Display_Image_-_Code_Walkthrough)

[nodelet/Tutorials/Running a nodelet - ROS Wiki](http://wiki.ros.org/nodelet/Tutorials/Running%20a%20nodelet)

[depth_image_proc - ROS Wiki](http://wiki.ros.org/depth_image_proc#depth_image_proc.2BAC8-point_cloud_xyzrgb)

[cv_bridgeTutorialsUsingCvBridgeToConvertBetweenROSImagesAndOpenCVImages - ROS Wiki](http://wiki.ros.org/cv_bridge/Tutorials/UsingCvBridgeToConvertBetweenROSImagesAndOpenCVImages)

[cv_bridgeTutorialsConvertingBetweenROSImagesAndOpenCVImagesPython - ROS Wiki](http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython)

[point_cloud_xyz.cpp at ros-perception/image_pipeline](https://github.com/ros-perception/image_pipeline/blob/indigo/depth_image_proc/src/nodelets/point_cloud_xyz.cpp)

[visualizing LiDAR point cloud in RViz](https://www.youtube.com/watch?v=e0r4uKK1zkk&t=0s&list=FLF_zvh-uhZH4D8PMzipB8wA&index=2)

[image_geometry](http://docs.ros.org/api/image_geometry/html/python/)
