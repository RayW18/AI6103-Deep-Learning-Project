import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from torch.autograd import Variable
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import math
from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
import yaml


def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def psnr_fn(image, reconstructed_image):
    criterion = nn.MSELoss().to('cuda')
    loss = criterion(reconstructed_image, image)
    rmse = math.sqrt(loss.cpu().numpy())
    max_value = image.max()
    return 20 * math.log10(max_value / rmse)

def ssim_fn(image, reconstructed_image):
    # print(image.shape)
    i_C,i_H,i_W = image.size()
    r_C,r_H,r_W = i_C,i_H,i_W
    image = image.view([i_C,i_H,i_W])
    reconstructed_image = reconstructed_image.view([r_C,r_H,r_W])
    image = image.cpu().numpy()
    reconstructed_image = reconstructed_image.cpu().numpy()
    image = image.transpose(1,2,0)
    reconstructed_image = reconstructed_image.transpose(1,2,0)
    return ssim(image, reconstructed_image, multichannel=True)


def ssim_fn_SFTMD(image, reconstructed_image):
    i_B, i_C,i_H,i_W = image.size()
    r_B_, r_C,r_H,r_W = i_B, i_C,i_H,i_W
    image = image.view([i_C,i_H,i_W])
    reconstructed_image = reconstructed_image.view([r_C,r_H,r_W])
    image = image.cpu().numpy()
    reconstructed_image = reconstructed_image.cpu().numpy()
    image = image.transpose(1,2,0)
    reconstructed_image = reconstructed_image.transpose(1,2,0)
    return ssim(image, reconstructed_image, multichannel=True)

def new_isotropic_gaussian_kernel(l, sigma, tensor=False):
    # assert (l % 2) == 1
    x, y = np.meshgrid(np.linspace(-(l - 1) / 2, (l - 1) / 2, l), np.linspace(-(l - 1) / 2, (l - 1) / 2, l))
    dst = np.sqrt(x ** 2 + y ** 2)
    gauss_kernel = np.exp(-(dst ** 2 / (2.0 * sigma ** 2)))
    gauss_kernel = gauss_kernel / np.sum(gauss_kernel)
    return gauss_kernel


def new_batch_kernel_generation(batch, l=21, sig_min=1.8, sig_max=3.2, rate_iso=1.0, scaling=3, tensor=True,
                                random_choice=True):
    assert (l % 2) == 1
    batch_kernel = np.zeros((batch, l, l))
    if random_choice == True:
        for i in range(batch):
            if np.random.random() < 1:
                # batch_kernel[i]=random_anisotropic_gaussian_kernel(l=l, sig_min=sig_min, sig_max=sig_max, scaling=scaling, tensor=tensor)
                new_sig = np.random.random() * (sig_max - sig_min) + sig_min
                batch_kernel[i] = new_isotropic_gaussian_kernel(l, sigma=new_sig, tensor=tensor)
        #  else:
        #   batch_kernel[i]=random_anisotropic_gaussian_kernel(l=l, sig_min=sig_min, sig_max=sig_max, scaling=scaling, tensor=tensor)
    if random_choice == False:
        for i in range(batch):
            batch_kernel[i] = new_isotropic_gaussian_kernel(sigma=2.6, l=l, tensor=tensor)
    return torch.FloatTensor(batch_kernel) if tensor else batch_kernel


def Bicubic_interpolation1(inputs, scale):
    tensor = inputs.cpu().data
    B, C, H, W = tensor.size()
    H_new = int(H / scale)
    W_new = int(W / scale)
    tensor_view = tensor.view((B * C, 1, H, W))
    re_tensor = torch.zeros((B * C, 1, H_new, W_new))
    for i in range(B * C):
        to_pil_image = T.ToPILImage()
        img = to_pil_image(tensor_view[i])
        BICUBIC_interpolation = T.Resize(size=(H_new, W_new), interpolation=Image.BICUBIC)
        img1 = BICUBIC_interpolation(img)
        img2 = torch.ByteTensor(torch.ByteStorage.from_buffer(img1.tobytes()))
        img2 = img2.view(img1.size[1], img1.size[0], 1)
        img2 = img2.transpose(0, 1).transpose(0, 2).contiguous()
        img2 = img2.float().div(255)
        re_tensor[i] = img2

    outputs = re_tensor.view((B, C, H_new, W_new))
    return outputs


# def PCA(data, k=2):
#     X = torch.from_numpy(data)
#     X_mean = torch.mean(X, 0)
#     X = X - X_mean.expand_as(X)
#     U, S, V = torch.svd(torch.t(X))
#     return U[:, :k]  # PCA matrix
#
#
# class PCAEncoder(object):
#     def __init__(self, weight, cuda=False):
#         self.weight = weight  # [l^2, k]
#         self.size = self.weight.size()
#         if cuda:
#             self.weight = Variable(self.weight).cuda()
#         else:
#             self.weight = Variable(self.weight)
#
#     def __call__(self, batch_kernel):
#         B, H, W = batch_kernel.size()  # [B, l, l]
#
#         return torch.bmm(batch_kernel.view((B, 1, H * W)), self.weight.expand((B,) + self.size)).view((B, -1))

def PCA(data, k=10):
    A = torch.from_numpy(data)
    #X_mean = torch.mean(A, 0)
    A = A - torch.mean(A, 0).expand_as(A)
    U, S, V = torch.svd(torch.t(A))
    return U[:, :k].float()
    
class PCA_Strech(object):
    def __init__(self, weight, device='cpu'):
        self.weight = weight  # [l^2, k]
        self.size = self.weight.size()
        self.weight = Variable(self.weight).to(device)

    def __call__(self, batch_kernel):
        B, H, W = batch_kernel.size()  # [B, l, l]
        streching = self.weight.expand((B,) + self.size)
        batch_kernel = batch_kernel.view((B, 1, H * W))
        return torch.bmm(batch_kernel, streching).view((B, -1))

def random_batch_noise(batch, high, rate_cln=0.5):
    noise_level = np.random.uniform(size=(batch, 1)) * high
    noise_mask = np.random.uniform(size=(batch, 1))
    noise_mask[noise_mask < rate_cln] = 0
    noise_mask[noise_mask >= rate_cln] = 1
    return noise_level * noise_mask


def b_GaussianNoising(tensor, sigma, mean=0.0, noise_size=None, min=0.0, max=1.0):
    if noise_size is None:
        size = tensor.size()
    else:
        size = noise_size
    noise = torch.mul(torch.FloatTensor(np.random.normal(loc=mean, scale=1.0, size=size)), sigma.view(sigma.size() + (1, 1)))
    return torch.clamp(noise + tensor, min=min, max=max)

# Add noise to the LR image
def LR_random_batch_noise(batch, high, LR_H ,LR_W):
    sig_max=high
    sig_min=0
    real_sig=sig_max*np.random.random()/255
    noise = real_sig * np.random.randn(batch,3,LR_H,LR_W)
    return torch.FloatTensor(noise)
def HR_random_batch_noise(batch, high, HR_H, HR_W):
    sig_max=high
    sig_min=0
    real_sig=sig_max*np.random.random()/255
    noise = real_sig * np.random.randn(batch,3,HR_H,HR_W)
    return torch.FloatTensor(noise)

# def HR_random_batch_noise(batch, high, HR_H, HR_W):
#     noise1 = np.zero()
#     for i in range(0,256):
#       sig_max=np.random.random((1000, 15))
#       sig_min=0
#       real_sig=sig_max*np.random.random()/255
#       noise = real_sig * np.random.randn(batch,3,4,4)
#       noise1 = np.concatenate(noise,noise1)
#     print(noise1.shape)
#     return torch.FloatTensor(noise)

def HR_random_batch_noise(batch, high, HR_H, HR_W,rand_size):
    import random
    #rand_size = 16
    max = HR_H*HR_W/(rand_size**2)
    for i in range(0,int(max)):
      sig_max = random.randint(0,high)
      #print(sig_max)
      sig_min=0
      real_sig=sig_max*np.random.random()/255
      noise = real_sig * np.random.randn(batch,3,rand_size,rand_size)
      # print(noise.shape)
      #noise = np.reshape(noise,(1,3,32*32))
      noise = torch.FloatTensor(noise).to('cuda')
      if i ==0:
        noise1 = noise
      else:
        #noise1 = np.concatenate([noise1,noise],axis=2)
        noise1 = torch.cat((noise1,noise),1)
    noise1 = noise1.view(batch,3,HR_H,HR_W).to("cuda")
    noise2 =torch.rot90(noise1,1,dims=[2,3]).to("cuda")
    noise1 = noise1.add(noise2).div(2)
    return noise1




class NewBatchBlur(nn.Module):
    def __init__(self, l=15):
        super(NewBatchBlur, self).__init__()
        self.l = l
        # print("l=",l)
        self.pad = nn.ReflectionPad2d(l // 2)

    def forward(self, input, kernel):
        B, C, H, W = input.size()
        pad = self.pad(input)
        H_p, W_p = pad.size()[-2:]
        # print(len(kernel.size()))
        input_CBHW = pad.view((1, C * B, H_p, W_p))
        kernel_var = kernel.contiguous().view((B, 1, self.l, self.l)).repeat(1, C, 1, 1).view(
            (B * C, 1, self.l, self.l))
        return F.conv2d(input_CBHW, kernel_var, groups=B * C).view((B, C, H, W))

class BatchSRKernel(object):
    def __init__(self, l=21, sig=2.6, sig_min=0.2, sig_max=4.0, rate_iso=1.0, scaling=3, random=True):
        self.l = l
        self.sig = sig
        self.sig_min = sig_min
        self.sig_max = sig_max
        self.rate = rate_iso
        self.scaling = scaling
        self.random = random

    def __call__(self, random, batch, tensor=False):
        return  new_batch_kernel_generation(batch, l=self.l, sig_min=self.sig_min, sig_max=self.sig_max, rate_iso=self.rate, scaling=self.scaling, tensor=tensor, random_choice=self.random)

def prepro(inputs):
  batch_ker = new_batch_kernel_generation(batch=10000, l=21, sig_min=0.2, sig_max=4, rate_iso=1.0, scaling=4, tensor=False)
  print('batch kernel shape: {}'.format(batch_ker.shape))
  b = np.size(batch_ker, 0)
  batch_ker = batch_ker.reshape((b, -1))
  pca_matrix = PCA(batch_ker, k=10).float()
  print('PCA matrix shape: {}'.format(pca_matrix.shape))
  encoder = PCA_Strech(pca_matrix,'cuda')
  inputs=inputs.float()
  O_B, O_C, O_H, O_W = inputs.size()
  inputs=inputs.view(O_B,3,O_H,O_W)
  inputs=inputs.float()
  inputs = inputs.to("cuda")


  kernel_gen = BatchSRKernel(l=21, sig=2.6, sig_min=0.2, sig_max=4, rate_iso=1, scaling=4,random=True)
  blur=NewBatchBlur(l=21)
  B, C, H, W = inputs.size()
  b_kernels = Variable(kernel_gen(True, B, tensor=True)).cuda() 
  kernel_code = encoder(b_kernels) # B x self.para_input
  hr_blured_var =blur(Variable(inputs).cuda(), b_kernels)
  lr_blured_t = Bicubic_interpolation1(hr_blured_var,4)
  B,C,LRI_H,LRI_W=lr_blured_t.shape
  ker_map =   kernel_code
  LR_img=lr_blured_t

  return LR_img, ker_map
