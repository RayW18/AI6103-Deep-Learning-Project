# IKC implementation
Here is an implementation of ['Blind Super-Resolution With Iterative Kernel Correction'](https://arxiv.org/abs/1904.03377)
## Install
!git clone
import os
os.chdir('pytorch-CycleGAN-and-pix2pix/')
## prerequirements
GPU A100 40G
## Dataset Prepraration
For training: You can choose from [Flickr2K](http://cv.snu.ac.kr/research/EDSR/Flickr2K.tar), [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)

For testing: You can choose from [Set5](https://uofi.box.com/shared/static/kfahv87nfe8ax910l85dksyl2q212voc.zip), [Set14](https://uofi.box.com/shared/static/igsnfieh4lz68l926l8xbklwsnnk8we9.zip)

## Pretrained Model
You can download the models from ./checkpoints directory

## Train
1. To train the SFTMD network, change the DIR in config.py into your own dataset diretory
   
   !python train_SFTMD.py
2. To train the Predictor and Corrector network, change the DIR in config.py into your own dataset diretory
   
   !python train_IKC.py

## Test
1. If you 