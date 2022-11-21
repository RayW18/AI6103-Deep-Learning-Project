
from Functions import *
from model import *
from dataset import *


def main():
    accimage = None
    noise = False
    train_loss = 0
    correct = 0
    total = 0
    pic = [];
    iteration = 0
    dic = read_yaml('main.yaml')
    batch_ker = new_batch_kernel_generation(batch=30000, l=21, sig_min=1.8, sig_max=3.2, rate_iso=1.0, scaling=4,
                                            tensor=False)
    print('batch kernel shape: {}'.format(batch_ker.shape))
    b = np.size(batch_ker, 0)
    batch_ker = batch_ker.reshape((b, -1))
    print(batch_ker.shape)
    pca_matrix = PCA(batch_ker, k=10)
    print('PCA matrix shape: {}'.format(pca_matrix.shape))
    encoder = PCA_Strech(pca_matrix, 'cuda')

    train_data=train_data=train_dataset(DIR_PATH=dic['DIR_PATH'], DATASET_DIR=dic['DATASET_DIR'])
    net= SFTMD().to('cuda')
    optimizer=torch.optim.Adam(net.parameters(), lr=0.0002, betas=(0.9, 0.999), eps=1e-07)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=105)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 21000, T_mult=1, eta_min=0, last_epoch=- 1, verbose=False)

    for epoch in range(200):
        trainloader = torch.utils.data.DataLoader(train_data, batch_size=dic['SFTMD_Batch_Size'], shuffle=True, num_workers=2)
        for batch_idx, inputs in enumerate(trainloader):
            O_B, O_C, O_H, O_W = inputs.size()
            inputs1 = inputs

            inputs = inputs.view(O_B, 3, O_H, O_W)
            inputs = inputs.float()
            inputs = inputs.to("cuda")
            Noise = HR_random_batch_noise(batch=O_B, high=75, HR_H=O_H, HR_W=O_W, rand_size=8).to("cuda")
            inputs = inputs + Noise
            # Noise=HR_random_batch_noise(batch=B, high=15, HR_H=256, HR_W=256)
            kernel_gen = BatchSRKernel(l=21, sig=2.6, sig_min=0.2, sig_max=4, rate_iso=1, scaling=4)
            blur = NewBatchBlur(l=21)
            B, C, H, W = inputs.size()

            b_kernels = Variable(kernel_gen(True, B, tensor=True)).cuda()
            kernel_code = encoder(b_kernels)  # B x self.para_input
            hr_blured_var = blur(Variable(inputs).cuda(), b_kernels)

            lr_blured_t = Bicubic_interpolation1(hr_blured_var, 4)

            B, C, LRI_H, LRI_W = lr_blured_t.shape

            if noise == True:
                Noise_level = torch.FloatTensor(random_batch_noise(batch=B, high=0.08, rate_cln=0.5))
                lr_noised_t = b_GaussianNoising(lr_blured_t, Noise_level)
            #  Noise_level=LR_random_batch_noise(batch=B, high=15, LR_H=LRI_H ,LR_W=LRI_W)
            #  lr_noised_t = lr_blured_t+Noise_level
            else:
                Noise_level = torch.zeros((B, 1))
                lr_noised_t = lr_blured_t

            # Noise=LR_random_batch_noise(batch=B, high=15, LR_H=64, LR_W=64)
            # lr_blured_t=(lr_blured_t+Noise)

            Noise_level = Variable(Noise_level).cuda()
            # re_code =   kernel_code
            re_code = torch.cat([kernel_code, Noise_level * 10], dim=1) if noise else kernel_code
            lr_re = Variable(lr_noised_t).cuda()

            LR_img = lr_re
            ker_map = re_code

            Inputs = LR_img
            Codes = ker_map
            Targets = inputs

            criterion = nn.L1Loss().to('cuda')
            optimizer.zero_grad()
            outputs = net(Inputs, Codes)
            # print(outputs.shape)
            loss = criterion(outputs, Targets)
            loss.backward()
            optimizer.step()

            print("Epoch:%d, Total Intereration:%d, Loss:%5f" % (epoch, iteration, loss))
            iteration = iteration + 1
            scheduler.step()
        torch.save(net.state_dict(),dic['NEW_F_CKPT_FIR_NAME'])
                   
        
if __name__ == '__main__':
    main()
