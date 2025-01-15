# -*- coding: utf-8 -*-
import argparse
from glob import glob
from collections import OrderedDict
from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split

from skimage.io import imread
from baseline_model.unet2plus import UNet_2Plus
from baseline_model.deeplabv3 import DeepLabV3
from baseline_model.attention_unet import AttU_Net
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from VGUNet import *
from dataset import Dataset
from metrics import dice_coef, batch_iou, mean_iou, iou_score, AverageMeter
import losses
from utils.utils import str2bool, count_params
import pandas as pd
from baseline_model.vision_transformer import SwinUnet as ViT_seg
from skimage.io import imread, imsave
import os
from torch.nn import SyncBatchNorm
from config import get_config
import torch.distributed as dist

COMPILE = True
USE_AMP = True
DISTRIBUTED = False

loss_names = []
loss_names.append('BCEWithLogitsLoss')

def get_param_num(net):
    total = sum(p.numel() for p in net.parameters())
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('total params: %d,  trainable params: %d' % (total, trainable))
    
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", type=str, default=test,help="train or test")
    parser.add_argument('--name', default="vgunet",
                        help='model name: (default: arch+timestamp)')
    parser.add_argument("-a",'--appdix', default="normal",)
    parser.add_argument('--deepsupervision', default=False, type=str2bool)
    parser.add_argument('--saveout', default=False, type=str2bool)
    parser.add_argument('--dataset', default="jiu0Monkey",
                        help='dataset name')
    parser.add_argument('--input-channels', default=4, type=int,
                        help='input channels')
    parser.add_argument('--image-ext', default='png',
                        help='image file extension')
    parser.add_argument('--mask-ext', default='png',
                        help='mask file extension')
    parser.add_argument('--aug', default=False, type=str2bool)
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=loss_names,
                        help='loss: ' +
                            ' | '.join(loss_names) +
                            ' (default: BCEDiceLoss)')
    parser.add_argument('--epochs', default=400, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--early-stop', default=50, type=int,
                        metavar='N', help='early stopping (default: 20)')
    parser.add_argument('-b', '--batch-size', default=18, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                            ' | '.join(['Adam', 'SGD']) +
                            ' (default: Adam)')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')
    parser.add_argument('--pretrain', default=False, type=str2bool,
                        help='nesterov')
    parser.add_argument('--cfg', type=str, required=False,default="./swinunet.pth", metavar="FILE", help='path to config file', )
    parser.add_argument('--local_rank', default=1, type=int,
                        help='node rank for distributed training')

    args = parser.parse_args()

    return args

def train(args, train_loader, model, criterion, optimizer, epoch, scaler=None, scheduler=None):
    losses = AverageMeter()
    ious = AverageMeter()
    device = torch.device("cuda", args.local_rank)
    model.train()

    for i, (input, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        input = input.to(device)
        target = target.to(device)
        
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            output = model(input)
            loss = criterion(output, target)

        iou = iou_score(output, target)

        losses.update(loss.item(), input.size(0))
        ious.update(iou, input.size(0))

        optimizer.zero_grad()
        
        if USE_AMP:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)  # Unscale gradients before clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            scaler.step(optimizer)
            scaler.update()
            
        else:
            loss.backward()
            optimizer.step()

    log = OrderedDict([
        ('loss', losses.avg),
        ('iou', ious.avg),
    ])

    return log

def rgb_out(output):
    rgbPic = np.zeros([160, 160, 3], dtype=np.uint8)
    for idx in range(output.shape[1]):
        for idy in range(output.shape[2]):
            if output[0, idx, idy] > 0.5:
                rgbPic[idx, idy, 0] = 0
                rgbPic[idx, idy, 1] = 128
                rgbPic[idx, idy, 2] = 0
            if output[1, idx, idy] > 0.5:
                rgbPic[idx, idy, 0] = 255
                rgbPic[idx, idy, 1] = 0
                rgbPic[idx, idy, 2] = 0
            if output[2, idx, idy] > 0.5:
                rgbPic[idx, idy, 0] = 255
                rgbPic[idx, idy, 1] = 255
                rgbPic[idx, idy, 2] = 0
    return rgbPic

def validate(args, val_loader, model, criterion, save_output=False):
    device = torch.device("cuda", args.local_rank)
    losses = AverageMeter()
    ious = AverageMeter()
    if save_output==True:
        save_path = os.path.join("./datasets/BraTs2019/rgb_results/", args.name)
        os.makedirs(save_path,exist_ok=True)
        img_path = os.path.join("./datasets/BraTs2019/rgb_results/", "img")
        os.makedirs(img_path, exist_ok=True)

    model.eval()
    with torch.no_grad():
        for i, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            input = input.to(device)
            target = target.to(device)
            
            if USE_AMP:
                with torch.cuda.amp.autocast():
                    output = model(input)
                    if save_output==True:
                        gt_path = os.path.join(save_path, str(i) + "gt.png")
                        gt = rgb_out(target.squeeze())
                        imsave(gt_path, gt)
                    loss = criterion(output, target)
            iou = iou_score(output, target)

            losses.update(loss.item(), input.size(0))
            ious.update(iou, input.size(0))

    log = OrderedDict([
        ('loss', losses.avg),
        ('iou', ious.avg),
    ])

    return log

def main():
    args = parse_args()

    device = torch.device("cuda", args.local_rank)
    if args.name is None:
        args.name = '%s_%s_woDS' %(args.dataset, args.name)
    if not os.path.exists('models/%s' %args.name):
        os.makedirs('models/%s' %args.name,exist_ok=True)

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')

    with open('models/%s/args.txt' %args.name, 'w') as f:
        for arg in vars(args):
            print('%s: %s' %(arg, getattr(args, arg)), file=f)

    # define loss function (criterion)
    if args.loss == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().to(device)
    else:
        criterion = losses.__dict__[args.loss]().to(device)

    cudnn.benchmark = True

    # Data loading code
    img_paths = glob(r'./datasets/2-MICCAI_BraTS_2018/BraTS2018_trainImage/*')
    mask_paths = glob(r'./datasets/2-MICCAI_BraTS_2018/BraTS2018_trainMask/*')

    train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = \
        train_test_split(img_paths, mask_paths, test_size=0.2, random_state=41)
    print("train_num:%s"%str(len(train_img_paths)))
    print("val_num:%s"%str(len(val_img_paths)))

    print("=> creating model %s" %args.name)
    if args.name=="vgunet":
        model = VGUNet(in_ch=4,out_ch=3)
    else:
        print("Removed all other models")
        exit(1)

    get_param_num(model)
    model = model.to(device)

    if args.pretrain==True:
        print("usig pretrained model!!!")
        pretrain_pth = './models/unet/'+"unet_plain91.08.pth"#str(args.name)+'/'+str(args.name)+'_max_pool.pth'
        pretrain_pth = './models/'+str(args.name)+'/'+str(args.name)+'_parallel_pretrained.pth'
        pretrained_model_dict = torch.load(pretrain_pth)
        model_dict = model.state_dict()
        
        pretrained_dict = {k: v for k, v in pretrained_model_dict.items() if k in model_dict}  # filter out unnecessary keys
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    print(count_params(model))

    if args.optimizer == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, eps=1e-4 if USE_AMP else 1e-8)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
            momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.95, patience=3, verbose=True, min_lr=1e-7)

    train_dataset = Dataset(args, train_img_paths, train_mask_paths, args.aug)
    val_dataset = Dataset(args, val_img_paths, val_mask_paths)
    
    if DISTRIBUTED:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler,shuffle=False,pin_memory=True,drop_last=False)
        model = SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
        print("start parallel!!")
    
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset,num_workers=4,batch_size=args.batch_size,shuffle=True,pin_memory=True,drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,num_workers=4,batch_size=args.batch_size,shuffle=False,pin_memory=True,drop_last=False)
    
    if COMPILE:
        model = torch.compile(model)
    
    log = pd.DataFrame(index=[], columns=['epoch', 'lr', 'loss', 'iou', 'val_loss', 'val_iou'])

    best_iou = 0
    trigger = 0  ###triger for early stop
    for epoch in range(args.epochs):
        print('Epoch [%d/%d]' %(epoch, args.epochs))

        # train for one epoch
        scaler = None if not USE_AMP else torch.cuda.amp.GradScaler()
        train_log = train(args, train_loader, model, criterion, optimizer, epoch, scaler=scaler)
        scheduler.step(train_log['iou'])
        # evaluate on validation set
        val_log = validate(args, val_loader, model, criterion)

        print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f'
            %(train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']))

        tmp = pd.Series([
            epoch,
            args.lr,
            train_log['loss'],
            train_log['iou'],
            val_log['loss'],
            val_log['iou'],
        ], index=['epoch', 'lr', 'loss', 'iou', 'val_loss', 'val_iou'])

        log = log.append(tmp, ignore_index=True)
        log.to_csv('models/%s/log.csv' %args.name, index=False)

        trigger += 1

        if val_log['iou'] > best_iou and torch.distributed.get_rank() == 0: # specify the first node to save the model
            save_pth = 'models/'+str(args.name)+'/'+str(args.name)+"_"+str(args.appdix)+'.pth'
            os.makedirs('models/'+str(args.name)+'/',exist_ok=True)
            torch.save(model.state_dict(), save_pth)
            best_iou = val_log['iou']
            print("=> saved best model")
            trigger = 0

        # early stopping
        if not args.early_stop is None:
            if trigger >= args.early_stop:
                print("=> early stopping")
                break

        torch.cuda.empty_cache()
    return 0

def test():
    args = parse_args()
    device = torch.device("cuda", args.local_rank)

    if args.name is None:
        args.name = '%s_%s_woDS' %(args.dataset, args.name)
    if not os.path.exists('models/%s' %args.name):
        os.makedirs('models/%s' %args.name)

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')

    with open('models/%s/args.txt' %args.name, 'w') as f:
        for arg in vars(args):
            print('%s: %s' %(arg, getattr(args, arg)), file=f)

    #joblib.dump(args, 'models/%s/args.pkl' %args.name)

    # define loss function (criterion)
    if args.loss == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().to(device)
    else:
        criterion = losses.__dict__[args.loss]().to(device)

    cudnn.benchmark = True

    # Data loading code
    img_paths = glob(r'./datasets/BraTs2019/testImage/*')
    mask_paths = glob(r'./datasets/BraTs2019/testMask/*')

    # create model
    if args.name == "vgunet":
        model = VGUNet(in_ch=4, out_ch=3)
    if args.name == "unet++":
        model = UNet_2Plus(in_channels=4, n_classes=3)
    if args.name == "deeplabv3":
        model = DeepLabV3(class_num=3)
    if args.name == "attunet":
        model = AttU_Net(img_ch=4, output_ch=3)
    if args.name == "swinunet":
        config = get_config(args)
        model = ViT_seg(config,  num_classes=3)
    model = model.to(device)
    print(count_params(model))

    test_dataset = Dataset(args, img_paths, mask_paths)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        drop_last=False)
    test_log = validate(args, test_loader, model, criterion,save_output=args.saveout)

    print('loss %.4f - iou %.4f'
        %(test_log['loss'], test_log['iou']))

    torch.cuda.empty_cache()
    
    
if __name__ == '__main__':
    args = parse_args()
    if args.action=="train":
        main()
    if args.action == "test":
        test()
