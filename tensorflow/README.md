# Train/Test/Pred Framework 
Train:

```shell
python3 predict_nick.py -m train --machine <'nicolas' or 'olorin'> --gpu 0 -s kitti_continuous --px all --loss berhu --max_steps 75000 --ldecay --l2norm --remove_sky --dataaug -t -v
```

Test:

```shell
python3 predict_nick.py -m test -s kitti_continuous_residential -r output/fcrn/2018-02-26_17-08-45/restore/model.fcrn --gpu 1 --remove_sky -u
```

Predict:

```shell
python3 predict_nick.py -m pred -r ../models/NYU_FCRN-checkpoint/NYU_FCRN.ckpt -i ../misc/nyu_example.png --gpu 1
```

# TensorBoard

```shell
tensorboard --logdir=MEGA/workspace/FCRN-DepthPrediction/tensorflow/output/fcrn/apolloscape
```
```shell
tensorboard --logdir=MEGA/workspace/FCRN-DepthPrediction/tensorflow/output/fcrn/kitti_depth
```
```shell
tensorboard --logdir=MEGA/workspace/FCRN-DepthPrediction/tensorflow/output/fcrn/kitti_discrete
```
```shell
tensorboard --logdir=MEGA/workspace/FCRN-DepthPrediction/tensorflow/output/fcrn/kitti_continuous
```
```shell
tensorboard --logdir=MEGA/workspace/FCRN-DepthPrediction/tensorflow/output/fcrn/nyudepth
```

# Predictions Evaluation

Using official evaluation tool from KITTI Depth Prediction Dataset:

```shell
python3 predict_nick.py -m test -s kitti_depth --gpu 0 -u --eval_tool kitti_depth --test_split eigen_kitti_depth
```

Using Monodepth's evaluation code:

```shell
python3 predict_nick.py -m test -s kitti_depth --gpu 0 -u --eval_tool monodepth --test_split eigen_kitti_depth
```
# Real-Time Prediction using OpenCV:

Runs the specified model:

```shell
python3 predict_cv.py -r ../models/NYU_FCRN-checkpoint/NYU_FCRN.ckpt -i ../misc/drone_indoor.mp4
```

```shell
python3 predict_cv.py -r output/fcrn/kitti_continuous/all_px/berhu/2018-06-29_17-59-58/restore/model.fcrn ../misc/outdoor_dubai_city.mp4
```

Detects and lists the available models:

```shell
python3 predict_cv.py -i ../misc/indoor_drone.mp4 --gpu 0
python3 predict_cv.py -i ../misc/outdoor_dubai_city.mp4 --gpu 0
```

Encode Video:

```shell
ffmpeg -r 30 -f image2 -s 304x288 -i frame%06d.png -i pred%06d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p ../test.mp4
```

Dependencies:

1.1) Gstreamer:

```shell
sudo apt-get install libgstreamer1.0-0 gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-doc gstreamer1.0-tools
```

1.2) ffmpeg:

```shell
sudo apt install ffmpeg
```

1.3) Grant access to user for using video devices:

```shell
grep video /etc/group
sudo usermod -a -G video olorin
sudo chmod 777 /dev/video0
```

# DeepLab's Four Alignment Rules:
URL: https://github.com/tensorflow/tensorflow/issues/6720

1) Use of odd-sized kernels in all convolution and pooling ops.
2) Use of SAME boundary conditions in all convolution and pooling ops.
3) Use align_corners=True when upsampling feature maps with bilinear interpolation.
4) Use of inputs with height/width equal to a multiple of the output_stride, plus one (for example, when the CNN output stride is 8, use height or width equal to 8 * n + 1, for some n, e.g., image HxW set to 321x513).

# Run Coverage for Codacy Support 

https://support.codacy.com/hc/en-us/articles/207279819-Coverage
https://support.codacy.com/hc/en-us/articles/207279819-Coverage
https://support.codacy.com/hc/en-us/articles/207312879-Generate-Coverage

Setup:

```shell
pip install codacy-coverage
export CODACY_PROJECT_TOKEN=%Project_Token%
```

Updating Codacy:

```shell
coverage run predict_nick.py -m train --machine nicolas -s kitti_discrete --px all --loss mse --max_steps 150000 --ldecay --l2norm --data_aug -t
coverage xml
python-codacy-coverage -r coverage.xml
```

Use `coverage report` to report on the results:

```shell
coverage report -m
```