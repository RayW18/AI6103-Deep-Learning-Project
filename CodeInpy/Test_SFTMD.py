from model import *
from Functions import *
from dataset import *
from collections import OrderedDict

dic = read_yaml('main.yaml')
DIR_PATH = dic['DIR_PATH']
HR_file_path = dic['HR_file_path']
LR_file_path = dic['LR_file_path']
load_path_c = dic['C_CKPT_FIR_NAME']
load_path_p = dic['P_CKPT_FIR_NAME']
load_path_f = dic['F_CKPT_FIR_NAME']
test_data_HR = test_dataset(HR_file_path)
test_data_LR = test_dataset(LR_file_path)
test_data_HR_IKC = test_dataset_IKC(HR_file_path)
# self.test_data_LR_IKC = test_dataset_IKC(LR_file_path)



def load_net_func(net, load_path):
    load_net = torch.load(DIR_PATH+load_path)
    load_net_clean = OrderedDict()  # remove unnecessary 'module.'
    for k, v in load_net.items():
        if k.startswith('module.'):
            [k[7:]] = v
        else:
            load_net_clean[k] = v
    net.load_state_dict(load_net_clean, strict=True)
    return net





def main():
    
    net_p = load_net_func(Predictor().to('cuda'), load_path_p)
    net_c = load_net_func(Corrector().to('cuda'), load_path_c)
    net_f = load_net_func(SFTMD().to('cuda'), load_path_f)
    accimage = None
    noise = False
    train_loss = 0
    correct = 0
    total = 0
    pic = [];
    iteration = 0

    print("****** Test on SFTMD Model ******")

    batch_ker = new_batch_kernel_generation(batch=30000, l=21, sig_min=1.8, sig_max=3.2, rate_iso=1.0, scaling=4,
                                            tensor=False)
    b = np.size(batch_ker, 0)
    batch_ker = batch_ker.reshape((b, -1))
    pca_matrix = PCA(batch_ker, k=10).float()
    encoder = PCA_Strech(pca_matrix, 'cuda')

    testloader = torch.utils.data.DataLoader(test_data_HR, batch_size=1, shuffle=False, num_workers=2)
    count_batch = 0

    psnr_f_sum = []
    ssim_f_sum = []

    for batch_idx, inputs in enumerate(testloader):
        O_B, O_C, O_H, O_W = inputs.size()
        inputs1 = inputs
        print(inputs.shape)

        inputs = inputs.view(O_B, 3, O_H, O_W)
        inputs = inputs.float()
        inputs = inputs.to("cuda")

        kernel_gen = BatchSRKernel(l=21, sig=2.6, sig_min=0.2, sig_max=4, rate_iso=1, scaling=4)
        blur = NewBatchBlur(l=21)
        B, C, H, W = inputs.size()

        b_kernels = Variable(kernel_gen(True, B, tensor=True)).cuda()
        kernel_code = encoder(b_kernels)  # B x self.para_input
        hr_blured_var = blur(Variable(inputs).cuda(), b_kernels)

        lr_blured_t = Bicubic_interpolation1(hr_blured_var, 4)

        B, C, LRI_H, LRI_W = lr_blured_t.shape

        if noise == True:
            Noise_level = LR_random_batch_noise(batch=B, high=75, LR_H=LRI_H, LR_W=LRI_W)
            lr_noised_t = lr_blured_t + Noise_level
        else:
            Noise_level = torch.zeros((B, 1))
            lr_noised_t = lr_blured_t

        Noise_level = Variable(Noise_level).cuda()
        re_code = kernel_code
        lr_re = Variable(lr_noised_t).cuda()

        LR_img = lr_re
        ker_map = re_code

        SR_img_F = net_f(LR_img, ker_map.to('cuda')).detach().float().cpu().to('cuda')

        psnr_f = psnr_fn(inputs, SR_img_F)
        ssim_f = ssim_fn_SFTMD(inputs, SR_img_F)
        psnr_f_sum.append(psnr_f)
        ssim_f_sum.append(ssim_f)

        print(f'Image {batch_idx + 1} PSNR {psnr_f} SSIM {ssim_f}')

        psnr_f_avg = np.mean(psnr_f_sum)
        ssim_f_avg = np.mean(ssim_f_sum)
        print("Average Results: ")
        print(f"PSNR {psnr_f_avg} SSIM {ssim_f_avg}")

if __name__ == '__main__':
    main()