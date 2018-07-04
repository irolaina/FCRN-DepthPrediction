# Train/Test Framework 
Run Single Prediction: 

    python predict.py ../models/NYU_FCRN-checkpoint/NYU_FCRN.ckpt ../misc/nyu_example.png --gpu 1

Train on XPS:
    
    python3 predict_nick.py -m train --machine xps --gpu 0 -s kitticontinuous --px all --loss berhu --max_steps 75000 --ldecay --l2norm --remove_sky -t -v

Train on Olorin:
    
    python3 predict_nick.py -m train --machine olorin --gpu 0 -s kittidiscrete --px all --loss berhu --max_steps 10 --ldecay --l2norm --remove_sky 
    
Test:

    python3 predict_nick.py -m test -s kitticontinuous_residential -r output/fcrn/2018-02-26_17-08-45/restore/model.fcrn --gpu 1 --remove_sky -u

Predict:

    python3 predict_nick.py -m pred -r ../models/NYU_FCRN-checkpoint/NYU_FCRN.ckpt -i ../misc/nyu_example.png --gpu 1

# Predictions Evaluation

Kitti Depth Prediction:

    cd /media/nicolas/nicolas_seagate/datasets/kitti/depth/depth_prediction/data/devkit/cpp
    ./evaluate_depth ~/MEGA/workspace/FCRN-DepthPrediction/tensorflow/output/tmp/gt_imgs ~/MEGA/workspace/FCRN-DepthPrediction/tensorflow/output/tmp/pred_imgs

# Real-Time Prediction using OpenCV:

    python3 predict_cv.py ../models/NYU_FCRN-checkpoint/NYU_FCRN.ckpt ../misc/drone_indoor.mp4
    python3 predict_cv.py ../models/NYU_FCRN-checkpoint/NYU_FCRN.ckpt ../misc/drone_indoor2.mp4

    python3 predict_cv.py ../models/NYU_FCRN-checkpoint/NYU_FCRN.ckpt ../misc/drone_indoor.mp4 --gpu 1 2> tmp/error.txt

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
