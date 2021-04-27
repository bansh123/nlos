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
import copy
from generator import *
from scipy import signal

#g = torch.load("final_2d_2500_5class.pt",map_location=torch.device('cpu'))
#g.eval()
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

        #g = torch.load("final_pose4_g.pt",map_location=torch.device('cpu'))
        #g.eval()
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
        train_dir = [0,1,6,7,8,9,11,12,13,15,17,18,20]  #0,2,3,5,6,8,12,13,20
        test_dir = [2, 5, 10, 14, 16, 19] # cur version - 2 : 2, 5, 10, 14, 16, 19    // 0,2,6,8,13,15,17,18,16   1,7,20,9,10,11,12,5,14,19  1,7,20,9,10,12,5,14,16,19,21,22,23,24,25,26,27,28,29
        #test_dir = [2, 5, 10] # los
        #test_dir = [14, 16, 19]  #nlos
        #test_dir = [10, 19] # demo - with mask  ,  los , nlos
        remove_dir = [3,4]
        #valid_dir = [25, 26, 27]
        #valid_dir = [21]
        #valid_dir = [28, 29] # nlos wall
        valid_dir = [x for x in range(21, 40)]
        #valid_dir = [x for x in range(1, 40)]  # Model test
        dir_count = 0
        rf_index = 0
        if mode == 'train':
            outlier_list = []
            #outlier_list = range(0,2500)
            #outlier_list2 = range(0,2500)
        else:
            outlier_list = []
            #l = []
            #l.extend(range(4000,10000))
            #l.extend(range(0,4000,200))
            #outlier_list = l
            #random.seed(7)
            #outlier_list = random.sample(range(3000,5000),1000)
            #a = range(3000,5000)
            #outlier_list2 = [x for x in a if x not in outlier_list]
        
        rf_index = -1
        gt_index = -1
        img_index = -1

        for file,file2 in zip(data_path_list,data_path_list2):
            rf_index= -1
            gt_index= -1
            if dir_count in remove_dir:
                dir_count += 1
                continue
            if mode == 'train' and dir_count not in train_dir:
                dir_count += 1
                continue
            elif mode == 'test' and dir_count not in test_dir:
                dir_count += 1
                continue
            elif mode == 'valid' and dir_count not in valid_dir:
                dir_count += 1
                continue
            
            #D_gt = torch.zeros(1,9)
            '''
            if dir_count ==0:
                D_gt=0
            elif dir_count ==2:
                D_gt=1
            elif dir_count ==3:
                D_gt=2
            elif dir_count ==5:
                D_gt=3
            elif dir_count ==6:
                D_gt=4
            elif dir_count ==8:
                D_gt=5
            elif dir_count ==12:
                D_gt=6
            elif dir_count ==13:
                D_gt=7
            elif dir_count==20:
                D_gt=8
            '''
            '''
            if dir_count ==0:
                D_gt[0][0]=1
            elif dir_count ==2:
                D_gt[0][1]=1
            elif dir_count ==3:
                D_gt[0][2]=1
            elif dir_count ==5:
                D_gt[0][3]=1
            elif dir_count ==6:
                D_gt[0][4]=1
            elif dir_count ==8:
                D_gt[0][5]=1
            elif dir_count ==12:
                D_gt[0][6]=1
            elif dir_count ==13:
                D_gt[0][7]=1
            elif dir_count==20:
                D_gt[0][8]=1
            '''
            '''
            D_gt = torch.zeros(1,4)
            if dir_count ==8:
                D_gt[0][0]=1
            elif dir_count ==12:
                D_gt[0][1]=1
            elif dir_count ==13:
                D_gt[0][2]=1
            elif dir_count==20:
                D_gt[0][3]=1
            '''
            if os.path.isdir(file) is True:
                # 각 폴더 안의 npy 데이터
                rf_file_list = glob.glob(file + '/raw/*.npy')
                rf_file_list = sorted(rf_file_list)
                print('dir(raw):', file, '\t# of data :', len(rf_file_list))
                gt_file_list = glob.glob(file2 + '/gt/*')
                gt_file_list = sorted(gt_file_list)
                print('dir(gt):', file2, '\t# of data :', len(gt_file_list))
                #print(rf_file_list)
                for rf,gt in zip(rf_file_list,gt_file_list):
                    rf_index += 1
                    if mode=='train' and rf_index in outlier_list:
                        continue
                    if mode == 'test' and rf_index in outlier_list:
                        continue
                    #if dir_count ==16:
                    #    if rf_index not in range(2450,2550):
                    #        continue
                    temp_raw_rf = np.load(rf)[:,:, 256:1056]
                    #print("raw shape", temp_raw_rf.shape)
                    #----- normalization ------
                    if self.is_normalize is True:
                        for i in range(temp_raw_rf.shape[0]):
                            for j in range(temp_raw_rf.shape[1]):
                                stdev = np.std(temp_raw_rf[i, j])
                                temp_raw_rf[i, j] = temp_raw_rf[i, j]/stdev
                    #---------- 2차원으로 만들기 -----------
                    if self.flatten:
                        temp_raw_rf= torch.tensor(temp_raw_rf).float()
                        temp_raw_rf= temp_raw_rf.view(-1,800)
                        
                    rf_data.append(temp_raw_rf)
                    #gt_list.append(gt)
                    temp_gt = np.load(gt)
                    temp_gt = torch.tensor(temp_gt).float()
                    gt_list.append(temp_gt)
                    '''
                    for i in range(40):
                        z = torch.randn((1,400))
                        z = z.float()
                        temp=temp_raw_rf.view(40,40)
                        with torch.no_grad():
                            res = g(temp.unsqueeze(0).unsqueeze(0),z)
                            res = res[0]
                        res=res.squeeze()
                        res=res.view(-1,1600)
                        rf_data.append(res)
                        gt_list.append(D_gt)
                    '''
                #break
                '''
                ground truth data 읽어오기.
                heatmap 형태. 총 데이터의 개수* keypoint * width * height
                '''
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
                self.load_img = False
                '''
                for gt in gt_file_list:
                    gt_index += 1
                    if mode == 'train' and gt_index not in outlier_list:
                        continue
                    if mode == 'test' and gt_index in outlier_list:
                        continue
                    temp_gt = np.load(gt)
                    temp_gt = torch.tensor(temp_gt).float()
                    gt_list.append(temp_gt)
                    #gt_list.append(D_gt)
                    #for i in range(39):
                    #    gt_list.append(D_gt)
                ''' 
                    #gt_list.append([D_gt,gt])
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
            #idxs = random.sample(range(0,79500),2)
            gt = self.gt_list[idx] #np.load()
        rf = self.rf_data[idx] 
        
        #---- augmentation  ----#
        random_prob = torch.rand(1)

        if self.mode == 'train' and self.augmentation != 'None' and random_prob < self.augmentation_prob:
            random_target = torch.randint(low=0, high=len(self.rf_data), size=(1,)).item() 
            #while random_target == idx: # random target이 동일하다면 다시 뽑음.
            #    random_target = torch.randint(low=0, high=len(self.rf_data), size=(1,)).item()
            target_gt = self.gt_list[random_target]
            #target_gt = torch.tensor(target_gt).reshape(gt.shape)
            target_rf = self.rf_data[random_target]
            #print("augmetatied rf = ", rf.shape)
            #print("augmented gt = ", gt.shape)
            if self.augmentation == 'cutmix':
                rf, gt = cutmix(rf, target_rf, gt, target_gt)
            elif self.augmentation == 'cutmix_1d':
                rf, gt= cutmix_1d(rf, target_rf, gt, target_gt)
            elif self.augmentation == 'mixup':
                rf, gt= mixup(rf, target_rf, gt, target_gt)
            elif self.augmentation =='intensity':
                rf = self.intensity(rf,target_rf,gt,target_gt)
            elif self.augmentation =='noise':
                rf = add_noise(rf)
            elif self.augmentation =='cutbox_1d':
                rf = cutbox_1d(rf)
            elif self.augmentation =='dagan':
                rf = dagan(rf)
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
                    
            return rf, gt#rf1,rf2,gt1,gt2,gt1_map,gt2_map
            # return self.rf_data[idx], self.gt_list[idx]
        else:
            return rf, gt, self.img_list[idx]

def cutmix(rf, target_rf, gt, target_gt):
    r = np.random.rand(1)
    if r>0.5:
        return rf,gt
    beta = 1.0
    lam = np.random.beta(beta, beta)
    bbx1, bby1, bbx2, bby2 = rand_bbox(rf.size(), lam)
    rf[:, bbx1:bbx2, bby1:bby2] = target_rf[:, bbx1:bbx2, bby1:bby2]
    lam = 1 - (((bbx2 - bbx1) * (bby2 - bby1)) / (rf.size()[-1] * rf.size()[-2]))
    new_rf = rf
    new_gt = lam * gt + (1 - lam) * target_gt
    return new_rf, new_gt

def cutmix_1d(rf, target_rf, gt, target_gt):
    r = np.random.rand(1)
    if r>0.5:
        return rf,gt
    beta = 1.0
    lam = np.random.beta(beta, beta)
    bbx1, bbx2 = rand_range(lam)
    rf[0, bbx1:bbx2] = target_rf[0, bbx1:bbx2]
    lam = 1 - ((bbx2 - bbx1) / 1792)
    new_rf = rf
    new_gt = lam * gt + (1 - lam) * target_gt
    return new_rf, new_gt

def dagan(rf):
    r = np.random.rand(1)
    if r>0.8:
        return rf
    z = torch.randn((1,400))
    z = z.float()
    rf=rf.view(40,40)
    with torch.no_grad():
        res = g(rf.unsqueeze(0).unsqueeze(0),z)
        res = res[0]
    res=res.squeeze()
    res=res.view(-1,1600)
    return res

def cutbox_1d(rf):
    r = np.random.rand(1)
    if r>0.5:
        return rf
    
    for attempt in range(100):
        area = 1792
        target_area = random.uniform(0.02,0.4)*area
        l = np.int(target_area)
        if l < 1792:
            x1 = random.randint(0,1792-l)
            rf[0,x1:x1+l] = 0
        return rf

    return rf

def add_noise(rf):
    r = np.random.rand(1)
    if r>0.9:
        return rf
    a = torch.normal(0.0,0.05,(1,1792))
    rf = rf+a
    return rf
    
def mixup(rf,target_rf,gt,target_gt):
    r = np.random.rand(1)
    if r>0.5:
        return rf,gt
    alpha = 1.0
    lam = np.random.beta(alpha, alpha)
    new_gt = lam * gt + (1 - lam) * target_gt
    new_rf = lam * rf + (1 - lam) * target_rf
    return new_rf,new_gt

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

def rand_range(lam):
    W = 1792
    cut_rat = 1. - lam
    cut_w = np.int(W * cut_rat)
    # uniform
    cx = np.random.randint(W)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)

    return bbx1, bbx2
