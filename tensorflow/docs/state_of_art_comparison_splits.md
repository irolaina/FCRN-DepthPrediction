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