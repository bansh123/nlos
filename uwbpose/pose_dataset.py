import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
import matplotlib.pyplot as pyplot
import torch.nn as nn
import os 
import glob
import numpy as np
import time
import cv2
import random
from scipy import signal

class PoseDataset(Dataset):
    def __init__(self, args, is_correlation=False, mode='train'):
        
        '''
        dataset 처리
        rf와 이미지의 경우에는 init 할 때부터 읽어와서 메모리에 올리지만 gt는 데이터를 활용할 때마다 load함.
        mode - train : 학습을 위함.  rf, gt, img 다 있는 경우
                test : test를 위함. rf, gt, img 다 있는 경우 
                valid: valid를 위함(demo). rf, img만 있는 경우
        '''
        self.is_correlation = is_correlation
        self.load_img = args.vis
        self.mode = mode
        
        self.is_gaussian = args.gaussian
        self.std = 0.1
        self.mean = 0
        
        self.is_normalize = args.normalize
        self.cutoff = args.cutoff
        
        self.augmentation = args.augment
        self.augmentation_prob = 1
        self.intensity = Intensity(scale=0.05)

        self.flatten = args.flatten
        self.arch = args.arch
        if self.arch =='hrnet':
            self.input_size= 128
        else:
            self.input_size = 120

        data_path = '/data/nlos/save_data_ver2'
        data_path2 = '/home/bansh123/deep-high-resolution-net.pytorch/gt/'
        #data_path_list = os.listdir(data_path)
        data_path_list = glob.glob(data_path + '/*')
        data_path_list2 = glob.glob(data_path2 + '/*')
        #print("data list", data_path_list)
        data_path_list = sorted(data_path_list)
        data_path_list2 = sorted(data_path_list2)
        #print(data_path_list)
        rf_data = []  # rf data list
        gt_list = []  # ground truth
        img_list = []
        print("start - data read")
        #test_dir = [8, 9] # past version - 1
        test_dir = [2,5,10,14,16,19] # cur version - 2 : 2, 5, 10, 14, 16, 19
        #test_dir = [2, 5, 10] # los
        #test_dir = [14, 16, 19]  #nlos
        #test_dir = [10, 19] # demo - with mask  ,  los , nlos
        remove_dir = [3, 4] 
        #valid_dir = [25, 26, 27]
        #valid_dir = [21]
        #valid_dir = [28, 29] # nlos wall
        valid_dir = [x for x in range(21, 40)]
        #valid_dir = [x for x in range(1, 40)]  # Model test
        dir_count = 0
        rf_index = 0
        if mode == 'train':
            outlier_list = range(49500, 50000)
        else:
            outlier_list = range(18000, 19000)
        rf_index = -1
        gt_index = -1
        img_index = -1

        for file, file2 in zip(data_path_list,data_path_list2):
            if dir_count in remove_dir:
                dir_count += 1
                continue

            if mode == 'train' and (dir_count in test_dir or dir_count in valid_dir):
                dir_count += 1
                continue
            elif mode == 'test' and dir_count not in test_dir:
                dir_count += 1
                continue
            elif mode == 'valid' and dir_count not in valid_dir:
                dir_count += 1
                continue

            if os.path.isdir(file) is True:
                # 각 폴더 안의 npy 데이터
                rf_file_list = glob.glob(file + '/raw/*.npy')
                rf_file_list = sorted(rf_file_list)
                print('dir(raw):', file, '\t# of data :', len(rf_file_list))
                #print(rf_file_list)
                for rf in rf_file_list:
                    rf_index += 1
                    if rf_index in outlier_list:
                        continue
                    temp_raw_rf = np.load(rf)[:, :, 284:]
                    
                    if self.augmentation=='fft':
                        rf_data.append(temp_raw_rf)
                        continue
                    #print("raw shape", temp_raw_rf.shape)
                    #----- normalization ------
                    if self.is_normalize is True:
                        for i in range(temp_raw_rf.shape[0]):
                            for j in range(temp_raw_rf.shape[1]):
                                stdev = np.std(temp_raw_rf[i, j])
                                temp_raw_rf[i, j] = temp_raw_rf[i, j]/stdev
                    
                    #temp_raw_rf = np.transpose(temp_raw_rf, (2, 1, 0)).transpose(0, 2, 1)
                    #temp_raw_rf = torch.tensor(temp_raw_rf).float()    
                    #---------- 2차원으로 만들기 -----------
                    if self.flatten:
                        '''
                        ix=1
                        s = torch.zeros((9,1764))
                        t = torch.zeros((9,1764))
                        for p in range(3):
                            for q in range(3):
                                fft= np.fft.fft(temp_raw_rf[p,q,:]/1764)
                                x=torch.tensor(abs(fft))
                                y=torch.tensor(np.angle(fft))
                                s[ix-1]=x
                                t[ix-1]=y
                                ix+=1
                        s = torch.cat([s[:,:126],s[:,126:126*2],s[:,126*2:126*3],s[:,126*3:126*4],s[:,126*4:126*5],s[:,126*5:126*6],s[:,126*6:126*7],s[:,126*7:126*8],s[:,126*8:126*9],s[:,126*9:126*10],s[:,126*10:126*11],s[:,126*11:126*12],s[:,126*12:126*13],s[:,126*13:126*14]])
                        t = torch.cat([t[:,:126],t[:,126:126*2],t[:,126*2:126*3],t[:,126*3:126*4],t[:,126*4:126*5],t[:,126*5:126*6],t[:,126*6:126*7],t[:,126*7:126*8],t[:,126*8:126*9],t[:,126*9:126*10],t[:,126*10:126*11],t[:,126*11:126*12],t[:,126*12:126*13],t[:,126*13:126*14]])
                        s = resize_transform(s)
                        t = resize_transform(t)
                        s = s.squeeze(0)
                        t = t.squeeze(0)
                        d = torch.stack([s,t])
                        '''

                        '''
                        resize_transform = transforms.Compose(
                            [transforms.ToPILImage(),
                             transforms.Resize((120, 120)),
                             transforms.ToTensor()]
                        )
                        ix=1
                        s = torch.zeros((9,6,241,241))
                        t = torch.zeros((6,120,120))
                        arr = [0,21,42,84,126,189,882]
                        for p in range(3):
                            for q in range(3):
                                fft= np.fft.fft(temp_raw_rf[p,q,:])
                                X=abs(fft)
                                X /= np.mean(X)
                                X= torch.tensor(X).float()
                                a=np.array(np.clip(np.around(fft.real+120),0,240),dtype=np.int)
                                b=np.array(np.clip(np.around(fft.imag+120),0,240),dtype=np.int)
                                for r in range(6):
                                    s[ix-1,r,a[arr[r]:arr[r+1]],b[arr[r]:arr[r+1]]] += X[arr[r]:arr[r+1]]*0.01
                                ix+=1
                        s2 = s.sum(0)
                        for i in range(6):
                            t[i]=resize_transform(s2[i])
                        '''
                        #t=torch.zeros(9,120,120)
                        temp_raw_rf= torch.tensor(temp_raw_rf).float()
                        temp_raw_rf= temp_raw_rf.view(-1,1764)
                        #np.random.seed()
                        #arr=np.random.permutation(14)*126
                        temp_raw_rf = torch.cat([temp_raw_rf[:,:126],temp_raw_rf[:,126:126*2],temp_raw_rf[:,126*2:126*3],temp_raw_rf[:,126*3:126*4],temp_raw_rf[:,126*4:126*5],temp_raw_rf[:,126*5:126*6],temp_raw_rf[:,126*6:126*7],temp_raw_rf[:,126*7:126*8],temp_raw_rf[:,126*8:126*9],temp_raw_rf[:,126*9:126*10],temp_raw_rf[:,126*10:126*11],temp_raw_rf[:,126*11:126*12],temp_raw_rf[:,126*12:126*13],temp_raw_rf[:,126*13:126*14]])
                        #temp_raw_rf = torch.cat([temp_raw_rf[:,arr[0]:arr[0]+126],temp_raw_rf[:,arr[1]:arr[1]+126],temp_raw_rf[:,arr[2]:arr[2]+126],temp_raw_rf[:,arr[3]:arr[3]+126],temp_raw_rf[:,arr[4]:arr[4]+126],temp_raw_rf[:,arr[5]:arr[5]+126],temp_raw_rf[:,arr[6]:arr[6]+126],temp_raw_rf[:,arr[7]:arr[7]+126],temp_raw_rf[:,arr[8]:arr[8]+126],temp_raw_rf[:,arr[9]:arr[9]+126],temp_raw_rf[:,arr[10]:arr[10]+126],temp_raw_rf[:,arr[11]:arr[11]+126],temp_raw_rf[:,arr[12]:arr[12]+126],temp_raw_rf[:,arr[13]:arr[13]+126]])
                        temp_raw_rf= temp_raw_rf.unsqueeze(0)
                        '''
                        resize_transform = transforms.Compose(
                            [transforms.ToPILImage(),
                             transforms.Resize((120, 120)),
                             transforms.ToTensor()]
                        )
                        temp_raw_rf=resize_transform(temp_raw_rf)
                        '''
                        #for i in range(9):
                        #    t[i]=resize_transform(temp_raw_rf[i].view(-1,42))
                        #temp_raw_rf = resize_transform(temp_raw_rf)
                    #print("now shape",temp_raw_rf.shape)
                    rf_data.append(temp_raw_rf)

                #break
                '''
                ground truth data 읽어오기.
                heatmap 형태. 총 데이터의 개수* keypoint * width * height
                '''
                gt_file_list = glob.glob(file2 + '/gt/*')
                gt_file_list = sorted(gt_file_list)
                print('dir(gt):', file2, '\t# of data :', len(gt_file_list))
                #np_load_old = np.load
                #np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
                #----- gt 메모리에 올려놓기 -----
                """for gt in gt_file_list:
                    temp_gt = np.load(gt)
                    temp_gt = torch.tensor(temp_gt).float()
                    temp_gt = temp_gt.reshape(13, 120, 120)
                    #print(temp_gt.shape, temp_gt.dtype)
                    gt_list.append(temp_gt)
                """
                #----- gt 파일 이름명만 리스트에 넣어놓기 -----
                for gt in gt_file_list:
                    gt_index += 1
                    if gt_index in outlier_list:
                        continue
                    gt_list.append(gt)
                #np.load = np_load_old
                if self.load_img is True:
                    img_file_list = glob.glob(file + '/img/*.jpg')
                    img_file_list = sorted(img_file_list)
                    print('dir(img):', file, '\t# of data :', len(img_file_list))
                    for img in img_file_list:
                        img_index += 1
                        if img_index in outlier_list:
                            img_index += 1
                            continue
                        temp_img = cv2.imread(img)
                        img_list.append(temp_img)
            dir_count += 1
        self.rf_data = rf_data
        self.gt_list = gt_list
        print(len(gt_list))
        if self.mode == 'valid' and len(self.gt_list) == 0:
            for i in range(len(self.rf_data)):
                self.gt_list.append(np.zeros((13, 120, 120)))
        self.img_list = img_list
        print("end - data read")
        print("size of dataset", len(self.rf_data))

    def __len__(self):    
        return len(self.rf_data)

    def __getitem__(self, idx):
        if self.mode == 'valid':
            gt = np.zeros((13, 120, 120))
        else:
            gt = np.load(self.gt_list[idx])
        gt = torch.tensor(gt).float()
        gt = gt.reshape(13, 120, 120)
        
        rf = self.rf_data[idx] 

        #---- augmentation  ----#
        random_prob = torch.rand(1)

        if self.mode == 'train' and self.augmentation != 'None' and random_prob < self.augmentation_prob:
            random_target = torch.randint(low=0, high=len(self.rf_data), size=(1,)).item() 
            
            #while random_target == idx: # random target이 동일하다면 다시 뽑음.
            #    random_target = torch.randint(low=0, high=len(self.rf_data), size=(1,)).item()
            target_gt = np.load(self.gt_list[random_target])
            target_gt = torch.tensor(target_gt).reshape(gt.shape)
            target_rf = self.rf_data[random_target]
            #print("augmetatied rf = ", rf.shape)
            #print("augmented gt = ", gt.shape)
            if self.augmentation == 'cutmix':
                rf, gt = cutmix(rf, target_rf, gt, target_gt)
            elif self.augmentation == 'mixup':
                rf, gt = mixup(rf, target_rf, gt, target_gt)
            elif self.augmentation =='intensity':
                rf = self.intensity(rf,target_rf,gt,target_gt)
            elif self.augmentation =='fft':
                rf = fft_bandstop(rf)
            elif self.augmentation =='all':
                r = np.random.rand(1)
                if r < 0.4:
                    rf, gt = cutmix(rf, target_rf, gt, target_gt)
                #elif r < 0.7:
                    #rf = self.intensity(rf)
                elif r < 0.8:
                    rf, gt = mixup(rf, target_rf, gt, target_gt)
            else:
                print('wrong augmentation')
        elif self.mode == 'test' and self.augmentation == 'fft':
            rf = fft_bandstop(rf)

        if self.load_img is False:
            #gaussian noise
            if self.mode == 'train' and self.is_gaussian is True:
                gt = gt + torch.randn(gt.size()) * self.std + self.mean
                    
            return rf, gt
            # return self.rf_data[idx], self.gt_list[idx]
        else:
            return rf, gt, self.img_list[idx]

def cutmix(rf, target_rf, gt, target_gt):
    beta = 1.0
    lam = np.random.beta(beta, beta)
    # print("rf.size ", rf.size())
    # print(rf.size()[-2])
    bbx1, bby1, bbx2, bby2 = rand_bbox(rf.size(), lam)
    # print(bbx1, bbx2, bby1, bby2)
    rf[:, bbx1:bbx2, bby1:bby2] = target_rf[:, bbx1:bbx2, bby1:bby2]
    # print((bbx2-bbx1)*(bby2-bby1))
    # print(rf.size()[-1] * rf.size()[-2])
    lam = 1 - (((bbx2 - bbx1) * (bby2 - bby1)) / (rf.size()[-1] * rf.size()[-2]))
    new_rf = rf
    new_gt = lam * gt + (1 - lam) * target_gt
    return new_rf, new_gt

def fft_bandstop(rf):
    ''' add random noise
    r = torch.rand(1)*3
    for p in range(3):
        for q in range(3):
            fft= np.fft.fft(rf[p,q,:])
            a=fft.real+np.random.normal(0,r,1764)
            b=fft.imag+np.random.normal(0,r,1764)
            fft=a+b*1j
            rf[p,q]=(np.fft.ifft(fft)).real
    '''
    ''' cancel noise
    for p in range(3):
        for q in range(3):
            fft= np.fft.fft(rf[p,q,:])
            fft[abs(fft)<=0.005*1764]=0
            rf[p,q]=np.fft.ifft(fft).real
    '''
    r = torch.rand(1)
    rf = torch.tensor(rf).float()
    rf= rf.view(-1,1764)
    np.random.seed()
    #ar=np.random.permutation(9)
    #rf = torch.stack([rf[ar[0],:],rf[ar[1],:],rf[ar[2],:],rf[ar[3],:],rf[ar[4],:],rf[ar[5],:],rf[ar[6],:],rf[ar[7],:],rf[ar[8],:]])  
    arr=np.random.permutation(14)*126
    temp_raw_rf = rf
    if r<0.5:
        temp_raw_rf = torch.cat([temp_raw_rf[:,:126],temp_raw_rf[:,126:126*2],temp_raw_rf[:,126*2:126*3],temp_raw_rf[:,126*3:126*4],temp_raw_rf[:,126*4:126*5],temp_raw_rf[:,126*5:126*6],temp_raw_rf[:,126*6:126*7],temp_raw_rf[:,126*7:126*8],temp_raw_rf[:,126*8:126*9],temp_raw_rf[:,126*9:126*10],temp_raw_rf[:,126*10:126*11],temp_raw_rf[:,126*11:126*12],temp_raw_rf[:,126*12:126*13],temp_raw_rf[:,126*13:126*14]])
    else:
        temp_raw_rf = torch.cat([temp_raw_rf[:,arr[0]:arr[0]+126],temp_raw_rf[:,arr[1]:arr[1]+126],temp_raw_rf[:,arr[2]:arr[2]+126],temp_raw_rf[:,arr[3]:arr[3]+126],temp_raw_rf[:,arr[4]:arr[4]+126],temp_raw_rf[:,arr[5]:arr[5]+126],temp_raw_rf[:,arr[6]:arr[6]+126],temp_raw_rf[:,arr[7]:arr[7]+126],temp_raw_rf[:,arr[8]:arr[8]+126],temp_raw_rf[:,arr[9]:arr[9]+126],temp_raw_rf[:,arr[10]:arr[10]+126],temp_raw_rf[:,arr[11]:arr[11]+126],temp_raw_rf[:,arr[12]:arr[12]+126],temp_raw_rf[:,arr[13]:arr[13]+126]])
    resize_transform = transforms.Compose(
        [transforms.ToPILImage(),
        transforms.Resize((120, 120)),
        transforms.ToTensor()]
    )
    temp_raw_rf = resize_transform(temp_raw_rf)
    return temp_raw_rf

def mixup(rf, target_rf, gt, target_gt):
    '''
    논문에서는 배치 내에서 섞지만, 전체 데이터에서 mixup.
    '''
    alpha = 1.0
    lam = np.random.beta(alpha, alpha)

    new_gt = lam * gt + (1 - lam) * target_gt
    new_rf = lam * rf + (1 - lam) * target_rf
    return new_rf, new_gt

class Intensity(nn.Module):
    def __init__(self, scale=0.05):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        r = torch.randn((x.size(0), 1, 1, ))
        noise = 1.0 + (self.scale * r.clamp(-2.0, 2.0))
        return x * noise


def rand_bbox(size, lam):
    W = size[1]
    H = size[2]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
