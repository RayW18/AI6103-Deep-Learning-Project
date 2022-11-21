from model import *
from Functions import *
from dataset import *
from collections import OrderedDict
import torch

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
    load_net = torch.load(DIR_PATH + load_path)
    load_net_clean = OrderedDict()  # remove unnecessary 'module.'
    for k, v in load_net.items():
        if k.startswith('module.'):
            [k[7:]] = v
        else:
            load_net_clean[k] = v
    net.load_state_dict(load_net_clean, strict=True)
    return net


def main():
    net_f = load_net_func(SFTMD().to('cuda'), load_path_f)
    net_p = load_net_func(Predictor().to('cuda'), load_path_p)
    net_c = load_net_func(Corrector().to('cuda'), load_path_c)
    HR_testloader = torch.utils.data.DataLoader(test_data_HR_IKC, batch_size=1, shuffle=False, num_workers=2)
    LR_testloader = torch.utils.data.DataLoader(test_data_LR, batch_size=1, shuffle=False, num_workers=2)
    count_batch = 0
    psnr_p_sum = []
    ssim_p_sum = []

    psnr_c_sum = []
    ssim_c_sum = []

    for batch_idx, inputs in enumerate(LR_testloader):
        LR_img = inputs
        print(LR_img.shape)
        GT_img = test_data_HR_IKC[batch_idx]
        print(GT_img.shape)
        fake_ker = net_p(LR_img.to('cuda'))
        est_ker_map = fake_ker.detach().float().cpu().to('cuda')

        SR_img = net_f(LR_img.to('cuda'), est_ker_map.to('cuda')).detach().float().cpu().to('cuda')
        print(SR_img.shape, GT_img.shape)
        psnr_p = psnr_fn(GT_img.to('cuda'), SR_img.to('cuda'))
        ssim_p = ssim_fn(GT_img.to('cuda'), SR_img.to('cuda'))
        psnr_p_sum.append(psnr_p)
        ssim_p_sum.append(ssim_p)

        print(f'Image {batch_idx + 1} P+SFTMD PSNR {psnr_p} SSIM {ssim_p}')

        psnr_c_list = []
        ssim_c_list = []

        for steps in range(10):
            delta_h = net_c(SR_img, est_ker_map.to('cuda'))
        est_ker_map = delta_h.detach().float().cpu()

        SR_img = net_f(LR_img.to('cuda'), est_ker_map.to('cuda')).detach().float().cpu().to('cuda')

        psnr_c = psnr_fn(GT_img.to('cuda'), SR_img.to('cuda'))
        ssim_c = ssim_fn(GT_img.to('cuda'), SR_img.to('cuda'))
        # ssim_c = 0
        psnr_c_list.append(psnr_c)
        ssim_c_list.append(ssim_c)

        print(f'Image {batch_idx + 1} Corrector Iteration {steps + 1} PSNR {psnr_c} SSIM {ssim_c}')
        psnr_c_max = max(psnr_c_list)
        ssim_c_max = max(ssim_c_list)
        psnr_c_sum.append(psnr_c_max)
        ssim_c_sum.append(ssim_c_max)
        print((f'Image {batch_idx + 1} Corrector Iteration MAX PSNR {psnr_c_max} MAX SSIM {ssim_c_max}'))

    psnr_p_avg = np.mean(psnr_p_sum)
    ssim_p_avg = np.mean(ssim_p_sum)

    psnr_c_avg = np.mean(psnr_c_sum)
    ssim_c_avg = np.mean(ssim_c_sum)
    print("Predictor + SFTMD Average Results:")
    print(f"PSNR {psnr_p_avg} SSIM {ssim_p_avg}")
    print("IKC Average Results:")
    print(f"PSNR {psnr_c_avg} SSIM {ssim_c_avg}")


if __name__ == '__main__':
    main()