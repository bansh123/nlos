############################################
#
#   Visualize results of trained model
#
############################################
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime

from tqdm import tqdm

from pose_dataset import *
from pose_target_dataset import *
from pose_resnet import *
from pose_resnet_1d import *
from pose_hrnet import *
import arguments
from make_log import *
from evaluate import *
from loss import *


args = arguments.get_arguments()

# model name
model_name = '{}_nlayer{}_{}_lr{}_batch{}_momentum{}_schedule{}_nepoch{}_{}'.format(
        args.name,
        args.nlayer,
        args.optimizer,
        args.lr,
        args.batch_size,
        args.momentum,
        args.schedule,
        args.nepochs,
        args.arch
    )
logger = make_logger(log_file=model_name)
logger.info("saved model name "+model_name)        

arguments.print_arguments(args, logger)

multi_gpu = args.multi_gpu
set_gpu_num = args.gpu_num

if torch.cuda.is_available():
    print("gpu", torch.cuda.current_device(), torch.cuda.get_device_name())
else:
    print("cpu")
    
#----- model -----
if args.arch =='hrnet':
    model = get_pose_hrnet()
else:
    if args.flatten:
        src_encoder = get_encoder(num_layer=args.nlayer, input_depth=1)
        src_encoder.load_state_dict(torch.load('./save_model/1*1_encoder.pt'))
        for param in src_encoder.parameters():
            param.requires_grad = False
        tgt_encoder = get_encoder(num_layer=args.nlayer, input_depth=1)
        tgt_encoder.load_state_dict(torch.load('./save_model/1*1_encoder.pt'))
        critic = get_Discriminator()
    else:
        model = get_pose_net(num_layer=args.nlayer, input_depth=1764)

if multi_gpu is True:
    model = torch.nn.DataParallel(model).cuda()
    logger.info("Let's use multi gpu\t# of gpu : {}".format(torch.cuda.device_count()))
else:
    torch.cuda.set_device(set_gpu_num)
    logger.info("Let's use single gpu\t now gpu : {}".format(set_gpu_num))
    #model.cuda()
    src_encoder = torch.nn.DataParallel(src_encoder, device_ids = [set_gpu_num]).cuda()
    tgt_encoder = torch.nn.DataParallel(tgt_encoder, device_ids = [set_gpu_num]).cuda()
    critic = torch.nn.DataParallel(critic, device_ids = [set_gpu_num]).cuda()
#model.cuda() # torch.cuda_set_device(device) 로 send
#model.to(device) # 직접 device 명시

#----- loss function -----
criterion = nn.CrossEntropyLoss().cuda()
cr = JointsMSELoss().cuda()
#----- optimizer and scheduler -----
if args.optimizer == 'adam':
    optimizer_tgt = torch.optim.RMSprop(tgt_encoder.parameters(), lr=args.lr)
    optimizer_critic = torch.optim.RMSprop(critic.parameters(), lr=args.lr)
    logger.info('use adam optimizer')
else:
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4, nesterov=False)
    logger.info('use sgd optimizer')

lr_scheduler_tgt = torch.optim.lr_scheduler.MultiStepLR(optimizer_tgt, milestones=args.schedule, gamma=args.gammas)
lr_scheduler_critic = torch.optim.lr_scheduler.MultiStepLR(optimizer_critic, milestones=args.schedule, gamma=args.gammas)
#----- dataset -----
train_data = PoseDataset(mode='train', args=args)
target_data = PoseDataset2(mode='train', args=args)
train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True)
target_dataloader = DataLoader(target_data, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True)

#----- training -----
max_acc = 0
max_acc_epoch = 0

src_encoder.eval()
tgt_encoder.train()
critic.train()
#name = './save_model/210309_1_nlayer18_adam_lr0.001_batch32_momentum0.9_schedule[4, 10]_nepoch1_resnet_epoch0.pt'
#state_dict = torch.load(name)
#model.module.load_state_dict(state_dict)
begin_time = datetime.now()
print(begin_time)
for epoch in range(args.nepochs):
    logger.info("Epoch {}\tcurrent lr : {} {}".format(epoch, optimizer_tgt.param_groups[0]['lr'], lr_scheduler_tgt.get_last_lr()))
    epoch_loss_tgt = []
    epoch_loss_cri = []
    iterate = 0
    data_zip = enumerate(zip(train_dataloader,target_dataloader))
    for step, ((rf_src, _), rf_tgt) in data_zip:
        
        rf_src, rf_tgt = rf_src.cuda(), rf_tgt.cuda()

        optimizer_critic.zero_grad()
        
        feat_src = src_encoder(rf_src)

        feat_tgt = tgt_encoder(rf_tgt)
        #feat_concat = torch.cat((feat_src,feat_tgt),0)
        #pred_concat = critic(feat_concat.detach())


        #label_src = torch.ones(feat_src.size(0)).long().cuda()
        #label_tgt = torch.zeros(feat_tgt.size(0)).long().cuda()
        #label_concat = torch.cat((label_src,label_tgt),0)
        #pred_src = critic(feat_src.detach())
        #pred_tgt = critic(feat_tgt.detach())
        loss_critic = -torch.mean(critic(feat_src.detach())) +torch.mean(critic(feat_tgt.detach()))
        #loss_critic = criterion(pred_concat,label_concat)
        loss_critic.backward()
        
        optimizer_critic.step()
        for p in critic.parameters():
            p.data.clamp_(-0.01,0.01)

        epoch_loss_cri.append(loss_critic)

        #pred_cls = torch.squeeze(pred_concat.max(1)[1])
        #acc = (pred_cls==label_concat).float().mean()
        
        if step % 5==0:
        #optimizer_critic.zero_grad()
            optimizer_tgt.zero_grad()

            #feat_tgt = tgt_encoder(rf_tgt)
            #pred_tgt = critic(feat_tgt)
            #label_tgt = torch.ones(feat_tgt.size(0)).long().cuda()

            #loss_tgt = criterion(pred_tgt,label_tgt)
            loss_tgt = -torch.mean(critic(feat_tgt))
            loss_tgt.backward()
            optimizer_tgt.step()
            epoch_loss_tgt.append(loss_tgt)
        
        if step==450: 
            logger.info("iteration[%d] loss_cri %.6f\tloss_tgt %.6f\t"%(iterate, loss_critic.item(),loss_tgt.item()))
        iterate += 1

    #logger.info("epoch loss_cri : %.6f\tloss_tgt : %.6f"%(torch.tensor(epoch_loss_cri).mean().item(),torch.tensor(epoch_loss_tgt).mean().item()))
    
    #if args.multi_gpu == 1:
    if epoch%100==0 or epoch==2999:
        torch.save(tgt_encoder.state_dict(), "save_model/tgt_encoder_epoch{}.pt".format(epoch))
    #else:
    #    torch.save(model.state_dict(), "save_model/" + model_name + "_epoch{}.pt".format(epoch))

logger.info("training end | elapsed time = " + str(datetime.now() - begin_time))


