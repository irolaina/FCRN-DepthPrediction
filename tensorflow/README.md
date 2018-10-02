# Train/Test Framework 
Run Single Prediction: 

    python predict.py ../models/NYU_FCRN-checkpoint/NYU_FCRN.ckpt ../misc/nyu_example.png --gpu 1

Train on XPS:
    
    python3 predict_nick.py -m train --machine nicolas --gpu 0 -s kitticontinuous --px all --loss berhu --max_steps 75000 --ldecay --l2norm --remove_sky -t -v

Train on Olorin:
    
    python3 predict_nick.py -m train --machine olorin --gpu 0 -s kittidiscrete --px all --loss berhu --max_steps 10 --ldecay --l2norm --remove_sky 
    
Test:

    python3 predict_nick.py -m test -s kitticontinuous_residential -r output/fcrn/2018-02-26_17-08-45/restore/model.fcrn --gpu 1 --remove_sky -u

Predict:

    python3 predict_nick.py -m pred -r ../models/NYU_FCRN-checkpoint/NYU_FCRN.ckpt -i ../misc/nyu_example.png --gpu 1

# TensorBoard

    tensorboard --logdir=MEGA/workspace/FCRN-DepthPrediction/tensorflow/output/fcrn/apolloscape
    tensorboard --logdir=MEGA/workspace/FCRN-DepthPrediction/tensorflow/output/fcrn/kittidepth
    tensorboard --logdir=MEGA/workspace/FCRN-DepthPrediction/tensorflow/output/fcrn/kittidiscrete
    tensorboard --logdir=MEGA/workspace/FCRN-DepthPrediction/tensorflow/output/fcrn/kitticontinuous
    tensorboard --logdir=MEGA/workspace/FCRN-DepthPrediction/tensorflow/output/fcrn/nyudepth

# Predictions Evaluation

Kitti Depth Prediction:

    cd /media/nicolas/nicolas_seagate/datasets/kitti/depth/depth_prediction/data/devkit/cpp
    ./evaluate_depth ~/MEGA/workspace/FCRN-DepthPrediction/tensorflow/output/tmp/gt_imgs ~/MEGA/workspace/FCRN-DepthPrediction/tensorflow/output/tmp/pred_imgs

# Real-Time Prediction using OpenCV:

Runs the specified model:

    python3 predict_cv.py -r ../models/NYU_FCRN-checkpoint/NYU_FCRN.ckpt -i ../misc/drone_indoor.mp4
    python3 predict_cv.py -r output/fcrn/kitticontinuous/all_px/berhu/2018-06-29_17-59-58/restore/model.fcrn ../misc/outdoor_dubai_city.mp4


Detects and lists the available models:

    python3 predict_cv.py -i ../misc/drone_indoor.mp4 --gpu 1

Encode Video:

    ffmpeg -r 30 -f image2 -s 304x288 -i frame%06d.png -i pred%06d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p ../test.mp4

Dependencies:

1.1) Gstreamer:

    sudo apt-get install libgstreamer1.0-0 gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-doc gstreamer1.0-tools

1.2) ffmpeg:

    sudo apt install ffmpeg

1.3) Grant access to user for using video devices:

    grep video /etc/group
    sudo usermod -a -G video olorin
    sudo chmod 777 /dev/video0
    
# DeepLab's Four Alignment Rules:
URL: https://github.com/tensorflow/tensorflow/issues/6720

1) Use of odd-sized kernels in all convolution and pooling ops.
2) Use of SAME boundary conditions in all convolution and pooling ops.
3) Use align_corners=True when upsampling feature maps with bilinear interpolation.
4) Use of inputs with height/width equal to a multiple of the output_stride, plus one (for example, when the CNN output stride is 8, use height or width equal to 8 * n + 1, for some n, e.g., image HxW set to 321x513).

# ROS Bag and Network Prediction Integration

1.) Generate ROS Bag file using Kitti2bag:

    $ python2 kitti2bag.py -t 2011_09_26 -r 0001 raw_synced

2.) Run the following ROS nodes:
    
Recommended:

    roslaunch fcrn image2pred.launch manager:=nodelet_manager
        
2.1) Init ROS
    
    $ roscore

2.2) Init ROSBag node

    $ cd ~/MEGA/workspace/kitti2bag
    $ rosbag play -l kitti_2011_09_26_drive_0001_synced.bag
    
2.3) Init Network Prediction node (image2pred)
  
    $ cd go_fcrn
    $ python2 ros_image2pred.py
    
2.4) Init RViz: ROS 3D Robot Visualizer
    
    $ rviz
        [Global Options]
            Fixed Frame: velo_link
        [Grid]
            Reference Frame: base_link
        [PointCloud2]
            Topic: /kitti/velo/pointcloud
        [Image]
            Image Topic: /kitti/camera_color_left/image_raw
        [Image]
            Image Topic: /kitti/camera_color_right/image_raw
        [Image]
            Image Topic: /kitti/camera_gray_left/image_raw
        [Image]
            Image Topic: /kitti/camera_gray_right/image_raw
        [Image]
            Image Topic: /pred_depth/image_8UC1
        [PointCloud2]
            Topic: /pred_depth/cloud

2.5) Init '[depth_image_proc](http://wiki.ros.org/depth_image_proc#depth_image_proc.2BAC8-point_cloud_xyz)' nodelet (depth2cloud).

    rosrun nodelet nodelet load <pkg_name>/<nodeletclass_name> <manager_name>

    $ rosrun nodelet nodelet manager __name:=nodelet_manager
    $ rosrun nodelet nodelet load depth_image_proc/point_cloud_xyz nodelet_manager __name:=nodelet_depth camera_info:=/pred_depth/camera_info image_rect:=/pred_depth/image_32FC1 points:=/pred_depth/cloud

Optional:     
    
    $ rostopic list
    $ rosnode list
    $ rostopic echo <topic>
    $ rostopic type <topic>
    $ rqt_graph
    $ roslaunch <package_name> <file.launch>
    

Terminal:
    
    # -------------------------------------------------- #
    |      roscore    |        ros_image2pred.py         |
    # -------------------------------------------------- #
    |      rosbag     |            empty                 |
    # -------------------------------------------------- #
    |       rviz      | depth_image_proc/point_cloud_xyz |
    # -------------------------------------------------- #
    | nodelet_manager |      rostopic echo <topic>       |
    # -------------------------------------------------- #
    
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

# Run Coverage for Codacy Support 

https://support.codacy.com/hc/en-us/articles/207279819-Coverage
https://support.codacy.com/hc/en-us/articles/207279819-Coverage
https://support.codacy.com/hc/en-us/articles/207312879-Generate-Coverage

Setup:
    pip install codacy-coverage
    export CODACY_PROJECT_TOKEN=%Project_Token%

Updating Codacy:

    coverage run predict_nick.py -m train --machine nicolas -s kittidiscrete --px all --loss mse --max_steps 150000 --ldecay --l2norm --data_aug -t
    coverage xml
    python-codacy-coverage -r coverage.xml

Use `coverage report` to report on the results:
    
    coverage report -m