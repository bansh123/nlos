import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import sys

import arguments
from pose_dataset import *
from pose_resnet_1d import *
from pose_resnet import *
from pose_hrnet import *
from loss import *
from visualize import *
from inference import *
from make_log import *
from evaluate import *
import resnet
from cutmix.utils import CutMixCrossEntropyLoss

def prediction(model, rf, target_heatmap, criterion):
    out = model(rf)
    #loss = criterion(out, target_heatmap)

    #_, temp_avg_acc, cnt, pred = accuracy(out.detach().cpu().numpy(),
    #        target_heatmap.detach().cpu().numpy())

    preds, maxvals = get_final_preds(out.clone().cpu().numpy())

    target_label, target_maxvals = get_final_preds(target_heatmap.clone().cpu().numpy())

    temp_true_det, temp_whole_cnt = pck(preds*4, target_label*4)

    return out, preds, target_label, temp_true_det, temp_whole_cnt #loss, temp_avg_acc, cnt, 

def validate(dataloader, model, logger, criterion, debug_img=False):
    model.eval()
    with torch.no_grad():
        sum_acc = 0
        total_cnt = 0
        iterate = 0
        for rf,gt in tqdm(dataloader): 
            rf,gt=rf.cuda(),gt.cuda()
            out = model(rf)
            c=torch.argmax(out, dim=1)
            d=torch.argmax(gt.squeeze(), dim=1)
            #print(out,gt)
            sum_acc+=torch.sum(d==c)
            iterate += 1
            total_cnt+=len(gt)
        logger.info("epoch acc on test data : %.4f"%(sum_acc/total_cnt))
        return sum_acc/total_cnt


if __name__ == '__main__':

    args = arguments.get_arguments()
   
    model_name = args.model_name
    #model_name = "210109_newdata_normalize_nlayer18_adam_lr0.001_batch32_momentum0.9_schedule[10, 20]_nepoch30"
    #model_name = "210112_mixup_nlayer18_adam_lr0.001_batch32_momentum0.9_schedule[10, 20]_nepoch30"
    #model_name = '210113_intensity_nlayer18_adam_lr0.001_batch32_momentum0.9_schedule[10, 20]_nepoch30'
    model_name = '210427_classifier_1x1_1d_2500_mixup_nlayer18_adam_lr0.0001_batch100_momentum0.9_schedule[]_nepoch100_resnet'
    if len(model_name) == 0:
        print("You must enter the model name for testing")
        sys.exit()
    
    #if model_name[-3:] != '.pt':
    #    print("You must enter the full name of model")
    #    sys.exit()

    #model_name = model_name.split('_epoch')[0]
    print("vaildate mode = ", model_name)
    log_name = model_name.split('/')[-1]
    print("log_name = ", log_name)
    logger = make_logger(log_file='valid_'+log_name)
    logger.info("saved valid log file "+'valid_'+log_name)        

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
            #model = get_2d_pose_net(num_layer=args.nlayer, input_depth=1)
            model = get_Classifier(num_layer=args.nlayer)
        else:
            model = get_pose_net(num_layer=args.nlayer, input_depth=1)

    if multi_gpu is True:
        model = torch.nn.DataParallel(model).cuda()
        logger.info("Let's use multi gpu\t# of gpu : {}".format(torch.cuda.device_count()))
    else:
        torch.cuda.set_device(set_gpu_num)
        logger.info("Let's use single gpu\t now gpu : {}".format(set_gpu_num))
        #model.cuda()
        model = torch.nn.DataParallel(model, device_ids = [set_gpu_num]).cuda()

    #model.cuda() # torch.cuda_set_device(device) 로 send
    #model.to(device) # 직접 device 명시
    

    #----- loss function -----
    #criterion = nn.MSELoss().cuda()
    criterion = JointsMSELoss().cuda()
    #----- dataset -----
    test_data = PoseDataset(mode='test', args=args)
    dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True)
    celos = nn.CrossEntropyLoss()
    clos = CutMixCrossEntropyLoss(True)
    model_name = model_name + '_epoch{}.pt'
    # 원하는 모델 구간 지정해야함.
    #for i in range(20, 30):
    bestidx=0
    bestacc=0
    tempacc=0
    for i in range(0,100):
        logger.info("epoch %d"%i)
        logger.info('./save_model/' + model_name.format(i))
        model.module.load_state_dict(torch.load('./save_model/'+model_name.format(i)))
        #model.module.load_state_dict(torch.load(model_name.format(i)))
        #model.module.load_state_dict(torch.load(model_name))
        tempacc=validate(dataloader, model, logger, clos, debug_img=False)
        if bestacc<tempacc:
            bestacc=tempacc
            bestidx=i
    print("bestacc : %f"%bestacc)
    print("bestidx : %d"%bestidx)

