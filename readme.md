# IKC implementation
Here is an implementation of ['Blind Super-Resolution With Iterative Kernel Correction'](https://arxiv.org/abs/1904.03377)
## Install
```
!git clone https://github.com/RayW18/AI6103-Deep-Learning-Project.git
import os
os.chdir('/')
```
## prerequirements
GPU A100 40G
## Dataset Prepraration
For training: You can choose from [Flickr2K](http://cv.snu.ac.kr/research/EDSR/Flickr2K.tar), [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)

For testing: You can choose from [Set5](https://uofi.box.com/shared/static/kfahv87nfe8ax910l85dksyl2q212voc.zip), [Set14](https://uofi.box.com/shared/static/igsnfieh4lz68l926l8xbklwsnnk8we9.zip)

## Pretrained Model
You can download the models from ./checkpoints directory

## Train
1. To train the SFTMD network, change the DIR in config.py to your own dataset diretory
   ```
   !python train_SFTMD.py
   ```
2. To train the Predictor and Corrector network, change the DIR in config.py in your own dataset directory
   ```
   !python train_IKC.py
   ```
## Test
If you want to test the data in our providing images, just run the code below. If you want to test the data in other datasets, download and move them into the ./dataset directory.
1. if you want to test on SFTMD, change the DIR in config.py into your own dataset diretory and run
    ```
    !python test_SFTMD.py
    ```
2. if you want to test on IKC, change the DIR in config.py into your own dataset diretory and run
   ```
   !python test_IKC.py
   ```
## Visualize
```
```
## Results
### Synthetic Images
![](images/SyntheticImages/bird.png)
![](images/SyntheticImages/girl.png)
### Real World Images
![](images/RealWorldImages/Real.png)
### Corrctor process
![](images/Corrector/butter.png)
![](images/Corrector/womensit.png)