import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as pyplot
import numpy
from pose_hrnet import *

model = get_pose_hrnet()
model.load_state_dict(torch.load('./save_model/210209_ban_test_nlayer18_adam_lr0.001_batch32_momentum0.9_schedule[10, 20]_nepoch20_hrnet_epoch19.pt'))

weight = model.conv1.weight.data.numpy()
#plt.imshow(weight[0])
w_min, w_max = weight.min(),weight.max()
weight = (weight-w_min) / (w_max-w_min)
n,ix=8,1
for i in range(64):
    f=weight[i,:,:,:]
    ax = pyplot.subplot(8,8,ix)
    ax.set_xticks([])
    ax.set_yticks([])
    pyplot.imshow(f[0,:,:],cmap='gray')
    ix += 1


pyplot.show()
pyplot.savefig('firstlayer_filters.png')

