# IKC implementation
Here is an implementation of ['Blind Super-Resolution With Iterative Kernel Correction'](https://arxiv.org/abs/1904.03377)
## Install
To clone the project, you can run the code below.
```
git clone https://github.com/RayW18/AI6103-Deep-Learning-Project.git
import os
os.chdir('/')
```
## prerequirements
40G RAM memory GPU
## Dataset Prepraration
For training: You can choose from [Flickr2K](http://cv.snu.ac.kr/research/EDSR/Flickr2K.tar), [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)

For testing: You can choose from [Set5](https://uofi.box.com/shared/static/kfahv87nfe8ax910l85dksyl2q212voc.zip), [Set14](https://uofi.box.com/shared/static/igsnfieh4lz68l926l8xbklwsnnk8we9.zip)

## Pretrained Model
You can download the models from CodeInpy/model directory

## 

## Train
1. To train the SFTMD network, change the DIR_PATH in main.yml to your own dataset diretory
   ```
   python TrainSFTMD.py
   ```
2. To train the Predictor and Corrector network, change the DIR_PATH in main.yml in your own dataset directory
   ```
   python Train_IKC.py
   ```
## Test
If you want to test the data in our providing images, just run the code below. If you want to test the data in other datasets, download and move them into the IKC-master/dataset2/SuperResDT/Train/img256X256/ directory.
1. if you want to test on SFTMD, change the DIR_PATH in main.yml into your own dataset diretory and run
    ```
    python Test_SFTMD.py
    ```
2. if you want to test on IKC, change the DIR_PATH in main.yml into your own dataset diretory and run
   ```
   python Test_IKC.py
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
### PSNR & SSIM on test sets
![](images/result.jpg)