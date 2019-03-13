# FCRN Framework Description

##  1. Training  #

Arguments and Flags description:

`-m` , selects the running mode of the developed framework: `train`, `test` or `pred`.

`--gpu`, specifies the GPU id to run the code.

`--machine`, identifies the current machine: `nicolas` or `olorin`.

`--model_name`, selects the network topology.

`-s`/`--dataset`, argument selects the desired dataset for training: `apolloscape`, `kitti_depth`, `kitti_discrete`, `kitti_continuous`, `nyudepth`, or `lrmjose.`

`--px`, argument selects which pixels to optimize: `all` or `valid`. Default: `valid`

`--loss`, argument selects the desired loss function: `mse`,  `berhu`, `eigen`, `eigen_grads`, etc. Default: `berhu`

`--batch-size`, argument specifies the training batch size. Default: `4`

`--max_steps`, argument specifies the max number of training steps. Default: `300000`

`-l`/`--learning_rate`, defines the initial value of the learning rate. Default: `1e-4`

`-d`/`--dropout`, enables dropout in the model during training. Default: `0.5`
`--ldecay`, enables learning decay. Default: `False`
`-n`/`--l2norm`', enables L2 Normalization. Default: `False`
`--data_aug`, enables Data Augmentation. Default: `True`

`--remove_sky`, removes sky for KITTI Datasets. Default: `False`

```python

parser.add_argument('--full_summary', action='store_true',
                    help="If set, it will keep more data for each summary. Warning: the file can become very large")

parser.add_argument('--log_directory', type=str, help="Sets the directory to save checkpoints and summaries",
                    default='log_tb/')

parser.add_argument('-t', '--show_train_progress', action='store_true', help="Shows Training Progress Images",
                    default=False)

parser.add_argument('-v', '--show_valid_progress', action='store_true', help="Shows Validation Progress Images",
                    default=False)

parser.add_argument('--test_split', type=str, help="Selects the desired test split for State-of-art evaluation: 'kitti_stereo', 'eigen', 'eigen_continuous', etc",
                    default='')

parser.add_argument('--eval_tool', type=str, help="Selects the evaluation tool for computing metrics: 'monodepth' or 'kitti_depth'",
                    default='')

parser.add_argument('--test_file_path', type=str, help="Evaluates the Model for the images speficied by test_file.txt file",
                    default='')

parser.add_argument('--min_depth', type=float, help='minimum depth for evaluation', default=1e-3)
parser.add_argument('--max_depth', type=float, help='maximum depth for evaluation', default=80)
parser.add_argument('--eigen_crop', help='If set, crops according to Eigen NIPS14', action='store_true')
parser.add_argument('--garg_crop', help='If set, crops according to Garg  ECCV16', action='store_true')

# ========= #
#  Testing  #
# ========= #
parser.add_argument('-o', '--output_directory', type=str,
                    help='Sets the output directory for test disparities, if empty outputs to checkpoint folder',
                    default='')

parser.add_argument('-u', '--show_test_results', action='store_true',
                    help="Show the first batch testing Network prediction img", default=False)

# ============ #
#  Prediction  #
# ============ #
parser.add_argument('-r', '--model_path', type=str, help="Sets the path to a specific model to be restored", default='')
parser.add_argument('-i', '--image_path', help='Sets the path to the image to be predicted', default='')

parser.add_argument('--debug', action='store_true', help="Enables the Debug Mode", default=False)
parser.add_argument('--retrain', action='store_true', help="Enables the Retrain Mode", default=False)
```

Command line:

```shell
python3 predict_nick.py -m train --machine <'nicolas' or 'olorin'> --gpu 0 -s kitti_continuous --px all --loss berhu --max_steps 75000 --ldecay --l2norm --remove_sky --dataaug -t -v
```

## 2. Testing

Command line:

```shell
python3 predict_nick.py -m test -s kitti_continuous_residential -r output/fcrn/2018-02-26_17-08-45/restore/model.fcrn --gpu 1 --remove_sky -u
```

## 3. Predict (Single Image Prediction)

Command line:

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

# KITTI Depth Prediction Evaluation

Dependencies:

```shell
sudo apt-get install libpng++-dev
```


Compilation:

```shell
cd /media/nicolas/nicolas_seagate/datasets/kitti/depth/depth_prediction/depth_devkit/devkit/cpp
sh make.sh
```

Run:

```shell
./evaluation/kitti_depth_prediction_devkit/cpp/evaluate_depth output/tmp/gt/ output/tmp/pred/
```

#### Issues

Fix:

[Ubuntu 17.04 libpng12.so.0: cannot open shared object file #95](https://github.com/tcoopman/image-webpack-loader/issues/95)

```shell
wget -q -O /tmp/libpng12.deb http://mirrors.kernel.org/ubuntu/pool/main/libp/libpng/libpng12-0_1.2.54-1ubuntu1_amd64.deb   && sudo dpkg -i /tmp/libpng12.deb   && rm /tmp/libpng12.deb
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

# TODO: 

4.2 KITTI dataset.

To be able to compare with the state-of-the-art monocular depth learning approaches, we trained and evaluated our networks using two different train/test splits: Godard and Eigen. 

**Godard Split**. We use the same train/test sets that Godard et al [5] proposed in their work. 200 high quality disparity images in 28 scenes provided by the official KITTI training set are served as the ground truth for benchmarking. For the rest of 33 scenes with a total of 30,159 images, 29,000 images are picked for training and the remaining 1,159 images for testing. 

**Eigen Split**. For fair comparison with more previous works, we also use the
test split proposed by Eigen et al [12] that has been widely evaluated by the works of Garg et al [4], Liu et al [21], Zhou et al [6] and Godard et al [5]. This test split contains 697 images of 29 scenes. The rest of 32 scenes contain 23,488 images, in which 22,600 are used for training and the remaining for testing, similar to [4] and [5].

Trecho retirado de "Self-Supervised Monocular Image Depth Learning and Confidence Estimation"

## Monodepth's Evaluation on KITTI Description
Monodepth Evaluation Code:

    https://github.com/mrharicot/monodepth/blob/master/utils/evaluate_kitti.py

To evaluate run:  
```shell
python utils/evaluate_kitti.py --split kitti --predicted_disp_path ~/tmp/my_model/disparities.npy \
--gt_path ~/data/KITTI/
```
The `--split` flag allows you to choose which dataset you want to test on.  
* `kitti` corresponds to the 200 official training set pairs from [KITTI stereo 2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo).  
* `eigen` corresponds to the 697 test images used by [Eigen NIPS14](http://www.cs.nyu.edu/~deigen/depth/) and uses the raw LIDAR points.

**Warning**: The results on the Eigen split are usually cropped, which you can do by passing the `--garg_crop` flag.

## Semi-supervised monocular depth map prediction (CVPR2017)

https://github.com/a-jahani/semodepth/blob/master/eval/eval_kitti.py