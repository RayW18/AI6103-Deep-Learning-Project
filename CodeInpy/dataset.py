from torch.utils import data
import numpy as np
from PIL import Image
import os
import cv2
import torchvision.transforms as T

class train_dataset():
    def __init__(self, DIR_PATH, DATASET_DIR):
        self.file_path = DIR_PATH+DATASET_DIR
        # self.file_path =r'/content/drive/MyDrive/Colab Notebooks/DeepLearningFinalProject/IKC-master/dataset2/SuperResDT/Train/img256X256'
        # label=np.load('/content/drive/MyDrive/LMZ/NEWtrain/label.npy', allow_pickle=True)
        # label=label.tolist()
        self.label_dict = os.listdir(self.file_path)

    def __getitem__(self, index):
        up_scale = 4
        mod_scale = 4
        img_id = self.label_dict[index - 1]
        img_path = self.file_path + '/' + (img_id)
        img = cv2.imread(img_path)

        image = img
        width = int(np.floor(image.shape[1] / mod_scale))
        height = int(np.floor(image.shape[0] / mod_scale))
        # modcrop
        if len(image.shape) == 3:
            image_HR = image[0:mod_scale * height, 0:mod_scale * width, :]
        else:
            image_HR = image[0:mod_scale * height, 0:mod_scale * width]

        # LR_blur, by random gaussian kernel

        transform1 = T.Compose([T.ToTensor(), ])  # range [0, 255] -> [0.0,1.0]
        image_HR = transform1(image_HR)
        image_HR = image_HR[[2, 1, 0], :, :].float()

        # image_HR=img2tensor(image_HR)
        # print(img_HR.shape)
        return image_HR

    def __len__(self):
        return len(os.listdir(self.file_path))
        
class test_dataset_IKC():
    def __init__(self, path):
        self.file_path = path
        self.label_dict=sorted(os.listdir(self.file_path))

    def __getitem__(self,index):
        up_scale = 16
        mod_scale = 16
        img_id = self.label_dict[index-1]
        img_path = self.file_path+'/'+(img_id)
        img=cv2.imread(img_path)
        image=img
        width = int(np.floor(image.shape[1] / mod_scale))
        height = int(np.floor(image.shape[0] / mod_scale))
        # modcrop
        h, w, c = image.shape
        if c == 1:
          image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        if len(image.shape) == 3:
            image_HR = image[0:mod_scale * height, 0:mod_scale * width, :]
        else:
            image_HR = image[0:mod_scale * height, 0:mod_scale * width]

        
        # LR_blur, by random gaussian kernel
        transform1 = T.Compose([T.ToTensor(), ])# range [0, 255] -> [0.0,1.0]
        image_HR = transform1(image_HR)
        image_HR = image_HR[[2,1,0],:,:].float()

        return image_HR

    def __len__(self):
        return len(os.listdir(self.file_path))
        
class test_dataset():
    def __init__(self, path):
        self.file_path = path
        self.label_dict=sorted(os.listdir(self.file_path))

    def __getitem__(self,index):
        up_scale = 4
        mod_scale = 4
        img_id = self.label_dict[index-1]
        img_path = self.file_path+'/'+(img_id)
        img=cv2.imread(img_path)


        image=img
        width = int(np.floor(image.shape[1] / mod_scale))
        height = int(np.floor(image.shape[0] / mod_scale))
        # modcrop
        h, w, c = image.shape
        if c == 1:
          image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        if len(image.shape) == 3:
            image_HR = image[0:mod_scale * height, 0:mod_scale * width, :]
        else:
            image_HR = image[0:mod_scale * height, 0:mod_scale * width]
        
        # LR_blur, by random gaussian kernel

        transform1 = T.Compose([T.ToTensor(), ])# range [0, 255] -> [0.0,1.0]
        image_HR = transform1(image_HR)
        image_HR = image_HR[[2,1,0],:,:].float()

        return image_HR

    def __len__(self):
        return len(os.listdir(self.file_path))