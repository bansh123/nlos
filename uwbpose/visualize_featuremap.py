import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as pyplot
import numpy
import scipy
import cmath
from scipy import signal
from pose_hrnet import *
import copy
import resnet

model = resnet.nf_resnet18()
x = torch.zeros(32,1,120,120)
model(x)
#model = get_pose_hrnet()
#model.load_state_dict(torch.load('./save_model/210209_ban_test_nlayer18_adam_lr0.001_batch32_momentum0.9_schedule[10, 20]_nepoch20_hrnet_epoch19.pt'))

test_input = np.load('/data/nlos/save_data_ver2/2020-11-30_15-44-12.376844/raw/raw_01500.npy')[:, :,284:]
#plot raw inputs
'''
ix=1
for i in range(3):
    for j in range(3):
        ax = pyplot.subplot(3,3,ix)
        pyplot.plot(test_input[i,j,:])
        ix += 1
pyplot.show()
pyplot.savefig('test_rawinputs.png')
'''

#freq,phase domain input

'''
resize_transform = transforms.Compose(
     [transforms.ToPILImage(),
     transforms.Resize((120, 120)),
     transforms.ToTensor()]
     )
ix=1
s = torch.zeros((9,1764))
t = torch.zeros((9,1764))
for p in range(3):
     for q in range(3):
          fft= np.fft.fft(test_input[p,q,:])
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
print(d.shape)
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
          fft= np.fft.fft(test_input[p,q,:])
          X=abs(fft)
          X/=np.mean(X)
          X= torch.tensor(X).float()
          a=np.around(fft.real+120)
          a=np.array(np.clip(a,0,240),dtype=np.int)
          b=np.around(fft.imag+120)
          b=np.array(np.clip(b,0,240),dtype=np.int)
          for r in range(6):
               s[ix-1,r,a[arr[r]:arr[r+1]],b[arr[r]:arr[r+1]]] += X[arr[r]:arr[r+1]]*0.01
          ix+=1
s2 = s.sum(0)
ix=1
for i in range(6):
     pyplot.subplot(3,3,ix)
     t[i]=resize_transform(s2[i])      
     pyplot.imshow(t[i],cmap='gray')
     ix+=1

pyplot.savefig('test.png')
'''

#rotation
'''
t=np.pi
for p in range(3):
     for q in range(3):
          fft= np.fft.fft(test_input[p,q,:])
          #fft_mag=abs(fft)
          #fft_ang=np.zeros((1764))
          #for i in range(1764):
          #    fft_ang[i]=cmath.phase(fft[i])
          r1= np.array(( (np.cos(t), -np.sin(t)),
               (np.sin(t),  np.cos(t)) ))
          r2= np.array(( (np.cos(-t), -np.sin(-t)),
               (np.sin(-t),  np.cos(-t)) ))     
          a=fft.real
          b=fft.imag
          temp1=np.array([a[0:882],b[0:882]])
          temp2=np.array([a[882:1764],b[882:1764]])
          temp1=r1.dot(temp1)
          temp2=r2.dot(temp2)
          x=np.concatenate((temp1,temp2),1)
          #print(np.fft.ifft(x[0]+x[1]*1j))
          test_input[p,q]=np.fft.ifft(x[0]+x[1]*1j).real
#pyplot.show()
#pyplot.savefig('mag_ang.png')
ix=1
for i in range(3):
    for j in range(3):
        ax = pyplot.subplot(3,3,ix)
        ax.set_xticks([])
        ax.set_yticks([])
        pyplot.plot(test_input[i,j,:])
        ix += 1
pyplot.show()
pyplot.savefig('rotate.png')
'''

'''
#noise canceling
ix=1
for p in range(3):
     for q in range(3):
          fft= np.fft.fft(test_input[p,q,:])
          #fft[abs(fft)<=0.005*1764]=0
          a=fft.real
          b=fft.imag
          #a+=np.random.normal(0, 0.001, 1764)
          #b+=np.random.normal(0, 0.001, 1764)
          ax = pyplot.subplot(3,3,ix)
          pyplot.plot(a,b,',')
          ix += 1
          #fft=a+b*1j
          #test_input[p,q]=np.fft.ifft(fft).real
'''

'''
for i in range(3):
    for j in range(3):
        ax = pyplot.subplot(3,3,ix)
        ax.set_xticks([])
        ax.set_yticks([])
        pyplot.plot(test_input[i,j,:])
        ix += 1
pyplot.show()
pyplot.savefig('fftinputs.png')

'''        


#plot bandstopped inputs
'''
sr=1764
b= signal.firwin(101,cutoff=[700,881],fs=sr,pass_zero='bandpass')
for i in range(1):
    for j in range(1):
         x1 = signal.lfilter(b,[1.0],test_input[i,j,:])
         test_input[i][j]=x1
ix=1
for i in range(1):
    for j in range(1):
        #ax = pyplot.subplot(1,1,ix)
        #ax.set_xticks([])
        #ax.set_yticks([])
        pyplot.plot(test_input[i,j,:])
        ix += 1
pyplot.show()
pyplot.savefig('test_rawinputs_lpf.png')
'''

'''
print(test_input.shape)
test_input = torch.tensor(test_input).float()
test_input= test_input.view(-1,1764)
test_input= test_input.view(-1,126)
print(test_input.shape)
test_input = test_input.unsqueeze(0)
resize_transform = transforms.Compose(
                            [transforms.ToPILImage(),
                             transforms.Resize((120, 120)),
                             transforms.ToTensor()]
                        )
test_input = resize_transform(test_input)
test_input = test_input.unsqueeze(0)
pyplot.imshow(test_input[0,0,:,:],cmap='gray')
pyplot.show()
pyplot.savefig('test_input.png')
'''

'''
test_input = torch.tensor(test_input).float()
test_input = test_input.view(126, -1)
test_input = test_input.unsqueeze(0)
resize_transform = transforms.Compose(
                            [transforms.ToPILImage(),
                             transforms.Resize((120, 120)),
                             transforms.ToTensor()]
                        )
test_input = resize_transform(test_input)
test_input = test_input.unsqueeze(0)
pyplot.imshow(test_input[0,0,:,:],cmap='gray')
pyplot.show()
pyplot.savefig('test_input.png')
'''


#feature_map = model.conv1(test_input)
#feature_map = feature_map.detach().numpy()
#square = 8
#ix = 1
#for _ in range(square):
#    for _ in range(square):
#        ax = pyplot.subplot(square,square,ix)
#        ax.set_xticks([])
#        ax.set_yticks([])
#        pyplot.imshow(feature_map[0,ix-1,:,:],cmap='gray')
#        ix += 1
#pyplot.show()
#pyplot.savefig('firstlayer_featuremap.png')