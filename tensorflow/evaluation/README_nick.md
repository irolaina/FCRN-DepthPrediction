# Kitti Depth Prediction Evaluation

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

# Issues

Fix:

[Ubuntu 17.04 libpng12.so.0: cannot open shared object file #95](https://github.com/tcoopman/image-webpack-loader/issues/95)

```shell
wget -q -O /tmp/libpng12.deb http://mirrors.kernel.org/ubuntu/pool/main/libp/libpng/libpng12-0_1.2.54-1ubuntu1_amd64.deb   && sudo dpkg -i /tmp/libpng12.deb   && rm /tmp/libpng12.deb
```
