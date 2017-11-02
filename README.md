# Deeper Depth Prediction with Fully Convolutional Residual Networks

By [Iro Laina](http://campar.in.tum.de/Main/IroLaina), [Christian Rupprecht](http://campar.in.tum.de/Main/ChristianRupprecht), [Vasileios Belagiannis](http://www.robots.ox.ac.uk/~vb/), [Federico Tombari](http://campar.in.tum.de/Main/FedericoTombari), [Nassir Navab](http://campar.in.tum.de/Main/NassirNavab).

## Contents
0. [Introduction](#introduction)
0. [Quick Guide](#quick-guide)
0. [Models](#models)
0. [Results](#results)
0. [Citation](#citation)
0. [License](#license)


## Introduction

This repository contains the CNN models trained for depth prediction from a single RGB image, as described in the paper "[Deeper Depth Prediction with Fully Convolutional Residual Networks](https://arxiv.org/abs/1606.00373)". The provided models are those that were used to obtain the results reported in the paper on the benchmark datasets NYU Depth v2 and Make3D for indoor and outdoor scenes respectively. Moreover, the provided code can be used for inference on arbitrary images. 


## Quick Guide

The trained models are currently provided in two frameworks, MatConvNet and TensorFlow. Please read below for more information on how to get started.

### TensorFlow
The code provided in the *tensorflow* folder requires accordingly a successful installation of the [TensorFlow](https://www.tensorflow.org/) library (any platform). 
The model's graph is constructed in ```fcrn.py``` and the corresponding weights can be downloaded using the link below. The implementation is based on [ethereon's](https://github.com/ethereon/caffe-tensorflow) Caffe-to-TensorFlow conversion tool. 
```predict.py``` provides sample code for using the network to predict the depth map of an input image. Use ```python predict.py NYU_FCRN.ckpt yourimage.jpg``` to try the code.

### MatConvNet

**Prerequisites**

The code provided in the *matlab* folder requires the [MatConvNet toolbox](http://www.vlfeat.org/matconvnet/) for CNNs. It is required that a version of the library equal or newer than the 1.0-beta20 is successfully compiled either with or without GPU support. 
Furthermore, the user should modify  ``` matconvnet_path = '../matconvnet-1.0-beta20' ``` within `evaluateNYU.m` and `evaluateMake3D.m` so that it points to the correct path, where the library is stored. 

**How-to** 

For acquiring the predicted depth maps and evaluation on NYU or Make3D *test sets*, the user can simply run  `evaluateNYU.m` or `evaluateMake3D.m` respectively. Please note that all required data and models will be then automatically downloaded (if they do not already exist) and no further user intervention is needed, except for setting the options `opts` and `netOpts` as preferred. Make sure that you have enough free disk space (up to 5 GB). The predictions will be eventually saved in a .mat file in the specified directory.  

Alternatively, one could run `DepthMapPrediction.m` in order to manually use a trained model in test mode to predict the depth maps of arbitrary images. 

## Models

The models are fully convolutional and use the residual learning idea also for upsampling CNN layers. Here we provide the fastest variant in which interleaving of feature maps is used for upsampling. For this reason, a custom layer `+dagnn/Combine.m` is provided.

The trained models - namely **ResNet-UpProj** in the paper - can also be downloaded here:

- NYU Depth v2: [MatConvNet model](http://campar.in.tum.de/files/rupprecht/depthpred/NYU_ResNet-UpProj.zip), [TensorFlow model (.npy)](http://campar.in.tum.de/files/rupprecht/depthpred/NYU_ResNet-UpProj.npy), [TensorFlow model (.ckpt)](http://campar.in.tum.de/files/rupprecht/depthpred/NYU_FCRN-checkpoint.zip)
- Make3D: [MatConvNet model](http://campar.in.tum.de/files/rupprecht/depthpred/Make3D_ResNet-UpProj.zip), TensorFlow model (soon)


## Results

**NEW!** The predictions for the validation set of NYU-Depth-v2 dataset can also be downloaded [here](http://campar.in.tum.de/files/rupprecht/depthpred/predictions_NYUval.mat) (.mat). 

In the following tables, we report the results that should be obtained after evaluation and also compare to other (most recent) methods on depth prediction from a single image. 
- Error metrics on NYU Depth v2:

| State of the art on NYU     |  rel  |  rms  | log10 |
|-----------------------------|:-----:|:-----:|:-----:|
| [Roy & Todorovic](http://web.engr.oregonstate.edu/~sinisa/research/publications/cvpr16_NRF.pdf) (_CVPR 2016_) | 0.187 | 0.744 | 0.078 |
| [Eigen & Fergus](http://cs.nyu.edu/~deigen/dnl/) (_ICCV 2015_)  | 0.158 | 0.641 |   -   |
| **Ours**                        | **0.127** | **0.573** | **0.055** |
	
- Error metrics on Make3D:

| State of the art on Make3D  |  rel  |  rms  | log10 |
|-----------------------------|:-----:|:-----:|:-----:|
| [Liu et al.](https://bitbucket.org/fayao/dcnf-fcsp) (_CVPR 2015_)      | 0.314 |  8.60 | 0.119 |
| [Li et al.](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Li_Depth_and_Surface_2015_CVPR_paper.pdf) (_CVPR 2015_)      | 0.278 | 7.19 | 0.092 |
| **Ours**                        | **0.175** |  **4.45** | **0.072** |

- Qualitative results:
![Results](http://campar.in.tum.de/files/rupprecht/depthpred/images.jpg)

## Citation

If you use this method in your research, please cite:

    @inproceedings{laina2016deeper,
            title={Deeper depth prediction with fully convolutional residual networks},
            author={Laina, Iro and Rupprecht, Christian and Belagiannis, Vasileios and Tombari, Federico and Navab, Nassir},
            booktitle={3D Vision (3DV), 2016 Fourth International Conference on},
            pages={239--248},
            year={2016},
            organization={IEEE}
    }

## License

Simplified BSD License

Copyright (c) 2016, Iro Laina  
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
