# FCRN Framework Description

`--debug`, enables the Debug Mode. Default= `False`

`--machine`, identifies the current machine: `nicolas` or `olorin`. Default= `nicolas`

`-m` , selects the running mode of the developed framework: `train`, `test` or `pred`. Default= `train`
`--model_name`, selects the network topology. Default= `fcrn`

`--gpu`, specifies the GPU id to run the code. Default= `0`

##  1. Training  #

**Arguments and flags descriptions:**

`--retrain`, enables the Retrain Mode. Default= `False`

`-s`/`--dataset`, argument selects the desired dataset for training: `apolloscape`, `kitti_depth`, `kitti_discrete`, `kitti_continuous`, `nyudepth`, or `lrmjose.`

`--px`, argument selects which pixels to optimize: `all` or `valid`. Default= `valid`

`--loss`, argument selects the desired loss function: `mse`,  `berhu`, `eigen`, `eigen_grads`, etc. Default= `berhu`

`--batch_size`, argument specifies the training batch size. Default= `4`

`--max_steps`, argument specifies the max number of training steps. Default= `300000`

`-l`/`--learning_rate`, defines the initial value of the learning rate. Default= `1e-4`

`-d`/`--dropout`, enables dropout in the model during training. Default= `0.5`
`--ldecay`, enables learning decay. Default= `False`
`-n`/`--l2norm`', enables L2 Normalization. Default= `False`
`--data_aug`, enables Data Augmentation. Default= `True`

`--remove_sky`, removes sky for KITTI Datasets. Default= `False`

`--full_summary`, If set, it will keep more data for each summary. Warning: the file can become very large.

`--log_directory`, sets the directory to save checkpoints and summaries. Default= `log_tb/`

`-t/--show_train_progress`, shows the training images progress. Default= `False`

`-v/--show_valid_progress`, shows the validation images progress. Default= `False`

**Command line:**

```shell
python3 predict_nick.py --machine nicolas -m train --gpu 0 -s kitti_continuous --px valid --loss berhu --max_steps 300000 -l 1e-4 -d 0.5 --ldecay --l2norm --data_aug --remove_sky -t -v
```

## 2. Testing/Evaluation

**Arguments and flags descriptions:**

`--eval_tool`, selects the evaluation tool for computing metrics: `monodepth` or `kitti_depth`. Default= `''`

`--test_split`, selects the desired test split for evaluation: `kitti_stereo`, `eigen`, or `eigen_kitti_depth`. Default= `''`

The `--test_split` flag allows you to choose which dataset you want to test on.  
* `kitti_stereo` corresponds to the 200 official training set pairs from [KITTI stereo 2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo).  
* `eigen` corresponds to the 697 test images used by [Eigen NIPS14](http://www.cs.nyu.edu/~deigen/depth/) and uses the raw LIDAR points.
* `eigen_kitti_depth` corresponds to the 652 test images used by [Aleotti arXiv 2018](http://vision.deis.unibo.it/~ftosi/papers/monoGan.pdf) and uses ground truth semi-dense annotated depth images.

`--test_file_path`, evaluates the model for the speficied images from a test_file`--debug', action='store_true', help="Enables the Debug Mode", default=False)
.txt file. Default= `''`

`--min_depth`, specifies the minimum depth for evaluation. Default= `1e-3`
`--max_depth`, specifies the maximum depth for evaluation'. Default= `80`
`--eigen_crop`, If set, crops according to Eigen NIPS14.
`--garg_crop`, If set, crops according to Garg  ECCV16. **Warning**: The results on the Eigen split are usually cropped, which you can do by passing the `--garg_crop` flag.

`-o/--output_directory`, sets the output directory for test disparities, if empty outputs to checkpoint folder. Default= `''`

`-u/--show_test_results`, shows the network predictions for the specified test split images. Default= `False`

**Command line, when selecting a desired trained model:**

```shell
python3 predict_nick.py --machine nicolas -m test --gpu 0 -s kitti_continuous -r output/fcrn/2018-02-26_17-08-45/restore/model.fcrn --eval_tool monodepth --test_split eigen_kitti_depth -u
```

**Using official evaluation tool from KITTI Depth Prediction Dataset:**

```shell
python3 predict_nick.py -m test --gpu 0 -s kitti_continuous --eval_tool kitti_depth --test_split eigen_kitti_depth -u
```

**Using Monodepth's evaluation code:**

```shell
python3 predict_nick.py -m test --gpu 0 -s kitti_continuous --eval_tool monodepth --test_split eigen_kitti_depth -u
```

## 3. Predict (Single Image Prediction)

**Arguments and flags descriptions:**

`-r`/ `--model_path`, sets the path to a specific model to be restored. Default= `''`
`-i`, `--image_path`, sets the path to the image to be predicted. Default=`''`

**Command line:**

```shell
python3 predict_nick.py -m pred --gpu 0 -r ../models/NYU_FCRN-checkpoint/NYU_FCRN.ckpt -i ../misc/nyu_example.png 
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

# Real-Time Prediction using OpenCV:

**Run a specific model:**

```shell
python3 predict_cv.py -r ../models/NYU_FCRN-checkpoint/NYU_FCRN.ckpt -i ../misc/drone_indoor.mp4
```

```shell
python3 predict_cv.py -r output/fcrn/kitti_continuous/all_px/berhu/2018-06-29_17-59-58/restore/model.fcrn ../misc/outdoor_dubai_city.mp4
```

**Detects and lists the available models:**

```shell
python3 predict_cv.py -i ../misc/indoor_drone.mp4 --gpu 0
python3 predict_cv.py -i ../misc/outdoor_dubai_city.mp4 --gpu 0
```

**Encode Video:**

```shell
ffmpeg -r 30 -f image2 -s 304x288 -i frame%06d.png -i pred%06d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p ../test.mp4
```

**Dependencies:**

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

**Setup:**

```shell
pip install codacy-coverage
export CODACY_PROJECT_TOKEN=%Project_Token%
```

**Updating Codacy:**

```shell
coverage run predict_nick.py -m train --machine nicolas -s kitti_discrete --px all --loss mse --max_steps 150000 --ldecay --l2norm --data_aug -t
coverage xml
python-codacy-coverage -r coverage.xml
```

**Use `coverage report` to report on the results:**

```shell
coverage report -m
```

# Third-Party Evaluation Code
## 1. KITTI Depth Prediction Dataset's Evaluation tool

**Dependencies:**

```shell
sudo apt-get install libpng++-dev
```

**Compilation:**

```shell
cd /media/nicolas/nicolas_seagate/datasets/kitti/depth/depth_prediction/depth_devkit/devkit/cpp
sh make.sh
```

**Run:**

```shell
./evaluation/kitti_depth_prediction_devkit/cpp/evaluate_depth output/tmp/gt/ output/tmp/pred/
```

**Fixes**

1. [Ubuntu 17.04 libpng12.so.0: cannot open shared object file #95](https://github.com/tcoopman/image-webpack-loader/issues/95)

```shell
wget -q -O /tmp/libpng12.deb http://mirrors.kernel.org/ubuntu/pool/main/libp/libpng/libpng12-0_1.2.54-1ubuntu1_amd64.deb   && sudo dpkg -i /tmp/libpng12.deb   && rm /tmp/libpng12.deb
```

## 2. Monodepth's Evaluation Code

Monodepth Evaluation Code:

    https://github.com/mrharicot/monodepth/blob/master/utils/evaluate_kitti.py

To evaluate run:  
```shell
python utils/evaluate_kitti.py --split kitti --predicted_disp_path ~/tmp/my_model/disparities.npy \
--gt_path ~/data/KITTI/
```
## 3. A-jahani's Evaluation Code

https://github.com/a-jahani/semodepth/blob/master/eval/eval_kitti.py