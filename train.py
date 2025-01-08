# -*- coding: utf-8 -*-
import argparse
from glob import glob
from collections import OrderedDict
from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split
# from sklearn.externals import joblib
from skimage.io import imread
from baseline_model.unet2plus import UNet_2Plus
# from baseline_model.modeling.deeplab import *
# from baseline_model.UNet3Plus import *
from baseline_model.deeplabv3 import DeepLabV3
from baseline_model.attention_unet import AttU_Net
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from VGUNet import *
from dataset import Dataset
from metrics import dice_coef, batch_iou, mean_iou, iou_score, compute_metrics
import losses
from utils.utils import str2bool, count_params, AverageMeter
import pandas as pd
from baseline_model.vision_transformer import SwinUnet as ViT_seg
from skimage.io import imread, imsave
import os
from torch.nn import SyncBatchNorm
from config import get_config
import torch.distributed as dist
loss_names = []
loss_names.append('BCEWithLogitsLoss')

# manual seed 0 gives pretty good results
torch.manual_seed(0)
torch.set_float32_matmul_precision('high')
torch.cuda.empty_cache()

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
    parser.add_argument('--use-amp', default=True, type=str2bool)
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

import gc

def train(args, train_loader, model, criterion, optimizer, epoch, scaler, scheduler=None):
    device = torch.device("cuda", args.local_rank)
    
    losses = AverageMeter(device)
    ious = AverageMeter(device)

    model.train()
    
    for i, (input, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        torch.cuda.synchronize()
        input = input.to(device)
        target = target.to(device)

        if args.deepsupervision:
            outputs = model(input)
            if args.name == "dual":
                b, c, h, w = input.shape
                output = F.interpolate(output, size=(h, w), mode='bilinear', align_corners=True)
            loss = 0
            for output in outputs:
                loss += criterion(output, target)
            loss /= len(outputs)
            iou = iou_score(outputs[-1], target)
        else:
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    output = model(input)

                    if args.name == "dual":
                        b,c,h,w = input.shape
                        output = F.interpolate(output, size=(h, w), mode='bilinear', align_corners=True)

                    loss = criterion(output, target)
            else:
                output = model(input)

                if args.name == "dual":
                    b,c,h,w = input.shape
                    output = F.interpolate(output, size=(h, w), mode='bilinear', align_corners=True)

                loss = criterion(output, target)
                
            iou = iou_score(output, target)

        skip = ious.update(iou, input.size(0))
        if skip:
            print(input.isfinite().all())
            print(output.isfinite().all())
            print(target.isfinite().all())
            print(loss.isfinite().all())
        losses.update(loss, input.size(0), skip=skip)
        
        if args.use_amp:
            scaler.scale(loss).backward()

            # Gradient clipping
            scaler.unscale_(optimizer)  # Unscale gradients before clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()

        optimizer.zero_grad()
        
        if i % 100 == 99:
            gc.collect()
            torch.cuda.empty_cache()

    log = OrderedDict([
        ('loss', losses.get_avg),
        ('iou', ious.get_avg),
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

def validate(args, val_loader, model, criterion, save_output=False, is_test=False):
    device = torch.device("cuda", args.local_rank)
    losses = AverageMeter(device)
    ious = AverageMeter(device)
    if is_test:
        dice_scores = AverageMeter(device)  # For overall Dice score
        tc_dice_scores = AverageMeter(device)  # Tumor Core Dice
        wt_dice_scores = AverageMeter(device)  # Whole Tumor Dice
        et_dice_scores = AverageMeter(device)  # Enhanced Tumor Dice
        ppvs = AverageMeter(device)  # For overall Dice score
        tc_ppvs = AverageMeter(device)  # Tumor Core Dice
        wt_ppvs = AverageMeter(device)  # Whole Tumor Dice
        et_ppvs = AverageMeter(device)  # Enhanced Tumor Dice
        Hausdorff = AverageMeter(device)  # For overall Dice score
        tc_Hausdorff = AverageMeter(device)  # Tumor Core Dice
        wt_Hausdorff = AverageMeter(device)  # Whole Tumor Dice
        et_Hausdorff = AverageMeter(device)  # Enhanced Tumor Dice

    if save_output:
        save_path = os.path.join("./datasets/BraTs2019/rgb_results/", args.name)
        os.makedirs(save_path, exist_ok=True)
        img_path = os.path.join("./datasets/BraTs2019/rgb_results/", "img")
        os.makedirs(img_path, exist_ok=True)

    # Switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for i, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            input = input.to(device)
            target = target.to(device)

            # Compute output
            if args.deepsupervision:
                outputs = model(input)
                loss = sum(criterion(output, target) for output in outputs) / len(outputs)
                iou = iou_score(outputs[-1], target)
                pred = outputs[-1]
            else:
                if args.use_amp:
                    with torch.cuda.amp.autocast():
                        output = model(input)

                        if args.name == "dual":
                            b,c,h,w = input.shape
                            output = F.interpolate(output, size=(h, w), mode='bilinear', align_corners=True)

                        loss = criterion(output, target)
                else:
                    output = model(input)
                    loss = criterion(output, target)
                iou = iou_score(output, target)
                pred = output

            # Save outputs if required
            if save_output:
                gt_path = os.path.join(save_path, str(i) + "gt.png")
                gt = rgb_out(target.squeeze())
                imsave(gt_path, gt)

            # Update metrics
            losses.update(loss, input.size(0))
            ious.update(iou, input.size(0))
            
            if is_test:
                metrics = compute_metrics(pred, target)
                
                wt_dice_scores.update(torch.tensor(metrics['dice'][0], dtype=float).to(device), input.size(0))
                tc_dice_scores.update(torch.tensor(metrics['dice'][1], dtype=float).to(device), input.size(0))
                et_dice_scores.update(torch.tensor(metrics['dice'][2], dtype=float).to(device), input.size(0))
                dice_scores.update(torch.tensor(metrics['dice'], dtype=float).to(device).mean(), input.size(0))
                try:
                    wt_Hausdorff.update(torch.tensor(metrics['hd95'][0], dtype=float).to(device), input.size(0))
                    tc_Hausdorff.update(torch.tensor(metrics['hd95'][1], dtype=float).to(device), input.size(0))
                    et_Hausdorff.update(torch.tensor(metrics['hd95'][2], dtype=float).to(device), input.size(0))
                    Hausdorff.update(torch.tensor(metrics['hd95'], dtype=float).to(device).mean(), input.size(0))
                except ValueError:
                    pass
                
                wt_ppvs.update(torch.tensor(metrics['ppv'][0], dtype=float).to(device), input.size(0))
                tc_ppvs.update(torch.tensor(metrics['ppv'][1], dtype=float).to(device), input.size(0))
                et_ppvs.update(torch.tensor(metrics['ppv'][2], dtype=float).to(device), input.size(0))
                ppvs.update(torch.tensor(metrics['ppv'], dtype=float).to(device).mean(), input.size(0))

    log = OrderedDict([
        ('loss', losses.get_avg),
        ('iou', ious.get_avg),
    ])
    
    if is_test:
        log |= OrderedDict([
            ('dice', dice_scores.get_avg),
            ('wt_dice', wt_dice_scores.get_avg),
            ('tc_dice', tc_dice_scores.get_avg),
            ('et_dice', et_dice_scores.get_avg),
            
            ('ppv', ppvs.get_avg),
            ('wt_ppv', wt_ppvs.get_avg),
            ('tc_ppv', tc_ppvs.get_avg),
            ('et_ppv', et_ppvs.get_avg),
            
            ('Hausdorff', Hausdorff.get_avg),
            ('wt_Hausdorff', wt_Hausdorff.get_avg),
            ('tc_Hausdorff', tc_Hausdorff.get_avg),
            ('et_Hausdorff', et_Hausdorff.get_avg),
        ])

    return log


def main():
    args = parse_args()
    #args.dataset = "datasets"

    device = torch.device("cuda", args.local_rank) ########!!!!!!!!!!!
    
    # Data loading code
    img_paths = glob(r'./datasets/2-MICCAI_BraTS_2018/BraTS2018_trainImage/*')
    mask_paths = glob(r'./datasets/2-MICCAI_BraTS_2018/BraTS2018_trainMask/*')

    train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = \
        train_test_split(img_paths, mask_paths, test_size=0.2, random_state=41)
    print("train_num:%s"%str(len(train_img_paths)))
    print("val_num:%s"%str(len(val_img_paths)))
    train_dataset = Dataset(args, train_img_paths, train_mask_paths, args.aug)
    val_dataset = Dataset(args, val_img_paths, val_mask_paths)
    print(f"Batch size: {args.batch_size}")
    train_loader = torch.utils.data.DataLoader(train_dataset,num_workers=4,batch_size=args.batch_size,shuffle=True,pin_memory=True,drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,num_workers=4,batch_size=args.batch_size,shuffle=False,pin_memory=True,drop_last=False)
    
    if args.name is None:
        if args.deepsupervision:
            args.name = '%s_%s_wDS' %(args.dataset, args.name)
        else:
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

    # create model
    print("=> creating model %s" %args.name)
    if args.name=="vgunet":
        model = VGUNet(in_ch=4,out_ch=3)
    if args.name == "unet++":
        model = UNet_2Plus(in_channels=4,n_classes=3)
    if args.name == "deeplabv3":
        model = DeepLabV3(class_num=3)
    if args.name == "attunet":
        model = AttU_Net(img_ch=4, output_ch=3)
    if args.name == "swinunet":
        config = get_config(args)
        model = ViT_seg(config,  num_classes=3)
    if args.name=="dual":
        from baseline_model.DualGCN import DualSeg_res50
        import torch.nn.functional as F
        model = DualSeg_res50(num_classes=3)
    if args.name == "swin2":
        from baseline_model.swinunet2 import swin_transformer_v2_g,swin_transformer_v2_l
        from baseline_model.swinwrapper import ClassificationModelWrapper
        model = ClassificationModelWrapper(swin_transformer_v2_l(in_channels=4,input_resolution=(160, 160),\
                window_size=5),number_of_classes=3,output_channels=1536)

    get_param_num(model)
    model = model.to(device)
    # from torchsummary import summary
    # summary(model, input_size=(4, 160, 160))
    # 选择特定层
    target_layers = [] # [model.bottleneck]#,model.sgcn2,model.sgcn3]

    # 统计参数数量

    for target_layer in target_layers:
        total_params = 0
        for name, param in target_layer.named_parameters():
            if param.requires_grad:
                total_params += param.numel()

        print("参数数量:",str(target_layer), total_params)
    print("summary:",)
    if args.pretrain==True:
        model = load_vgunet(device, use_amp=args.use_amp)
        
    model = torch.compile(model)
    print(count_params(model))
    
    if args.pretrain==True:
        val_log = validate(args, val_loader, model, criterion, is_test=True)

        print(f"original: loss {val_log['loss']:.4f} - iou {val_log['iou']:.4f} - dice avg {val_log['dice']:.4f} - wt {val_log['wt_dice']:.4f} - et {val_log['et_dice']:.4f} - tc {val_log['tc_dice']:.4f}")
        print(f"PPV SCORES: ppv avg {val_log['ppv']:.4f} - wt_ppv {val_log['wt_ppv']:.4f} - et_ppv {val_log['et_ppv']:.4f} - tc_ppv {val_log['tc_ppv']:.4f}")
        print(f"Hausdorff: Hausdorff avg {val_log['Hausdorff']:.4f} - wt_Hausdorff {val_log['wt_Hausdorff']:.4f} - et_Hausdorff {val_log['et_Hausdorff']:.4f} - tc_Hausdorff {val_log['tc_Hausdorff']:.4f}")
        
    

    if args.optimizer == 'Adam':
        # not 1e-8, so that we don't run into NaN errors with amp autocast.
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, eps=1e-4 if args.use_amp else 1e-8)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
            momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.95, patience=3, verbose=True,min_lr=1e-7)

    ##parallel
    # dist.init_process_group(backend='nccl')
    # torch.cuda.set_device(args.local_rank)
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
    # val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler,shuffle=False,pin_memory=True,drop_last=False)
    # model = SyncBatchNorm.convert_sync_batchnorm(model)
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
    # print("start parallel!!")

    log = pd.DataFrame(index=[], columns=['epoch', 'lr', 'loss', 'iou', 'val_loss', 'val_iou'])

    best_iou = 0
    trigger = 0  ###triger for early stop
    
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(args.epochs):
        print('Epoch [%d/%d]' %(epoch, args.epochs))

        # train for one epoch
        train_log = train(args, train_loader, model, criterion, optimizer, epoch, scaler)
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

        log = pd.concat([log, tmp])
        log.to_csv('models/%s/log.csv' %args.name, index=False)

        trigger += 1

        if val_log['iou'] > best_iou: # and torch.distributed.get_rank() == 0:####specify the first node to save the model
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


def load_vgunet(device, use_amp=True, model_path="models/vgunet/vgunet_best.pth"):
    model = VGUNet()
    try:
        # Load the saved state dictionary
        state_dict = torch.load(model_path)

        # Strip the "_orig_mod." prefix
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("_orig_mod.", "")  # Remove the prefix
            new_state_dict[new_key] = value

        # Load the updated state dictionary into the model
        model.load_state_dict(new_state_dict)
    except Exception as e:
        print(e)
        print(f"Failed to load model at {model_path}")
        
    if not use_amp:
        model = model.float()
            
    return model.to(device)


def test():
    args = parse_args()
    #args.dataset = "datasets"
    device = torch.device("cuda", args.local_rank)

    if args.name is None:
        if args.deepsupervision:
            args.name = '%s_%s_wDS' %(args.dataset, args.name)
        else:
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
    # img_paths = glob(r'./datasets/2-MICCAI_BraTS_2018/BraTS2018_trainImage/*')
    mask_paths = glob(r'./datasets/BraTs2019/testMask/*')
    # mask_paths = glob(r'./datasets/2-MICCAI_BraTS_2018/BraTS2018_trainMask/*')

    
    if args.name == "unet++":
        model = UNet_2Plus(in_channels=4, n_classes=3)
        model = model.to(device)
    if args.name == "deeplabv3":
        model = DeepLabV3(class_num=3)
        model = model.to(device)
    if args.name == "attunet":
        model = AttU_Net(img_ch=4, output_ch=3)
        model = model.to(device)
    if args.name == "swinunet":
        config = get_config(args)
        model = ViT_seg(config,  num_classes=3)
        model = model.to(device)
    # create model
    if args.name == "vgunet":
        model = load_vgunet(device, use_amp=args.use_amp)

    print(count_params(model))

    test_dataset = Dataset(args, img_paths, mask_paths)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        drop_last=False)
    test_log = validate(args, test_loader, model, criterion,save_output=args.saveout, is_test=True)
    
    print(f"loss {test_log['loss']:.4f} - iou {test_log['iou']:.4f} - dice avg {test_log['dice']:.4f} - wt {test_log['wt_dice']:.4f} - et {test_log['et_dice']:.4f} - tc {test_log['tc_dice']:.4f}")
    print(f"PPV SCORES: ppv avg {test_log['ppv']:.4f} - wt_ppv {test_log['wt_ppv']:.4f} - et_ppv {test_log['et_ppv']:.4f} - tc_ppv {test_log['tc_ppv']:.4f}")
    print(f"Hausdorff: Hausdorff avg {test_log['Hausdorff']:.4f} - wt_Hausdorff {test_log['wt_Hausdorff']:.4f} - et_Hausdorff {test_log['et_Hausdorff']:.4f} - tc_Hausdorff {test_log['tc_Hausdorff']:.4f}")

    torch.cuda.empty_cache()
if __name__ == '__main__':
    args = parse_args()
    if args.action=="train":
        main()
    if args.action == "test":
        test()
