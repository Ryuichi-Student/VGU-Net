# -*- coding: utf-8 -*-

import time
import os
import math
import argparse
from glob import glob
import warnings
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from skimage.io import imread, imsave
from torch.utils.data import DataLoader
from metrics import dice_coef, batch_iou, mean_iou, iou_score ,ppv,sensitivity
from utils.utils import str2bool, count_params
from hausdorff import hausdorff_distance
import imageio
from skimage.io import imread
from baseline_model.unet2plus import UNet_2Plus
from baseline_model.deeplabv3 import DeepLabV3
from baseline_model.attention_unet import AttU_Net
from VGUNet import *
from dataset import Dataset
from baseline_model.vision_transformer import SwinUnet as ViT_seg
from config import get_config

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default="vgunet",
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--mode', default="Calculate", )
    parser.add_argument('--deepsupervision', default=False, type=str2bool)
    parser.add_argument('--saveout', default=False, type=str2bool)
    parser.add_argument('--input-channels', default=4, type=int,
                        help='input channels')
    parser.add_argument('-b', '--batch-size', default=1, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')
    parser.add_argument('--pretrain', default=False, type=str2bool,
                        help='nesterov')

    args = parser.parse_args()

    return args


def main():
    args =parse_args()# joblib.load('models/%s/args.pkl' %val_args.name)

    if not os.path.exists('output/%s' %args.name):
        os.makedirs('output/%s' %args.name)

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')

    # create model
    print("=> creating model %s" % args.name)
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
        model = ViT_seg(config,  num_classes=3).cuda()
    if args.name=="dual":
        from baseline_model.DualGCN import DualSeg_res50
        import torch.nn.functional as F
        model = DualSeg_res50(num_classes=3)
    model = model.cuda()

    # Data loading code
    img_paths = glob(r'./datasets/BraTs2019/testImage/*')
    mask_paths = glob(r'./datasets/BraTs2019/testMask/*')
    val_img_paths = img_paths
    val_mask_paths = mask_paths

    print("testing moe:",args.mode)
    if args.mode == "GetPicture":
        model = model.cuda()
        model = nn.DataParallel(model)##parallel train need parallel test
        pretrain_pth = ("./models/" + args.name + "/" + args.name + "_23.10.11_unet_skiponbottleneck.pth")# swinunet
        pretrained_model_dict = torch.load(pretrain_pth,map_location="cuda:0")
        model.load_state_dict(pretrained_model_dict)
        model.eval()
        test_dataset = Dataset(args, val_img_paths, val_mask_paths)
        plt_test = False
        if plt_test == True:
            select_index = 40  # ,54,26 long distance,40 two class
            test_dataset = Dataset(args, img_paths, mask_paths, select_index)

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True,drop_last=False)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            with torch.no_grad():
                for i, (input, target) in tqdm(enumerate(test_loader), total=len(test_loader)):
                    index=i
                    input = input.cuda()
                    #target = target.cuda()

                    # compute output
                    if args.deepsupervision:
                        output = model(input)[-1]
                        if args.name == "dual":
                            b, c, h, w = input.shape
                            output = F.interpolate(output, size=(h, w), mode='bilinear', align_corners=True)
                    else:
                        output = model(input)
                        if args.name == "dual":
                            b, c, h, w = input.shape
                            output = F.interpolate(output, size=(h, w), mode='bilinear', align_corners=True)
                    output = torch.sigmoid(output).data.cpu().numpy()
                    img_paths = val_img_paths[args.batch_size*i:args.batch_size*(i+1)]
                    for i in range(output.shape[0]):
                        if plt_test == False:
                            npName = os.path.basename(img_paths[i])
                            overNum = npName.find(".npy")
                            rgbName = npName[0:overNum]
                            rgbName = rgbName  + ".png"
                        else:####single test
                            idx = select_index
                            img_path = val_img_paths[idx]
                            name = os.path.basename(img_path)
                            overNum = name.find(".npy")
                            name = name[0:overNum]
                            rgbName = name + ".png"

                        rgbPic = np.zeros([160, 160, 3], dtype=np.uint8)
                        for idx in range(output.shape[2]):
                            for idy in range(output.shape[3]):
                                if output[i,0,idx,idy] > 0.5: ##green
                                    rgbPic[idx, idy, 0] = 0
                                    rgbPic[idx, idy, 1] = 128
                                    rgbPic[idx, idy, 2] = 0
                                if output[i,1,idx,idy] > 0.5: ##red
                                    rgbPic[idx, idy, 0] = 255
                                    rgbPic[idx, idy, 1] = 0
                                    rgbPic[idx, idy, 2] = 0
                                if output[i,2,idx,idy] > 0.5: ##yellow
                                    rgbPic[idx, idy, 0] = 255
                                    rgbPic[idx, idy, 1] = 255
                                    rgbPic[idx, idy, 2] = 0
                        os.makedirs('datasets/BraTs2019/rgb_results/%s/'%args.name,exist_ok=True)
                        imsave('datasets/BraTs2019/rgb_results/%s/'%args.name + str(index)+".png",rgbPic)

            torch.cuda.empty_cache()
            
        save_gt = False
        if save_gt==True:
            print("Saving GT,numpy to picture")
            val_gt_path = 'output/%s/'%args.name + "GT/"
            if not os.path.exists(val_gt_path):
                os.mkdir(val_gt_path)
            if plt_test == True:
                length = 1
            else:
                length= len(val_mask_paths)
            for idx in tqdm(range(length)):#)):
                if plt_test==True:
                    idx=select_index
                mask_path = val_mask_paths[idx]
                name = os.path.basename(mask_path)
                overNum = name.find(".npy")
                name = name[0:overNum]
                rgbName = name + ".png"
                npmask = np.load(mask_path)

                GtColor = np.zeros([npmask.shape[0],npmask.shape[1],3], dtype=np.uint8)
                for idx in range(npmask.shape[0]):
                    for idy in range(npmask.shape[1]):
                        # NET, non-enhancing tumor
                        if npmask[idx, idy] == 1:
                            GtColor[idx, idy, 0] = 255
                            GtColor[idx, idy, 1] = 0
                            GtColor[idx, idy, 2] = 0
                        # ED, peritumoral edema
                        elif npmask[idx, idy] == 2:
                            GtColor[idx, idy, 0] = 0
                            GtColor[idx, idy, 1] = 128
                            GtColor[idx, idy, 2] = 0
                        # ET, enhancing tumor
                        elif npmask[idx, idy] == 4:
                            GtColor[idx, idy, 0] = 255
                            GtColor[idx, idy, 1] = 255
                            GtColor[idx, idy, 2] = 0

                imageio.imwrite(val_gt_path + "gt_"+rgbName, GtColor)
            
        print("Done!")

    if args.mode == "Calculate":
        """
        计算各种指标:Dice、Sensitivity、PPV
        """
        # wt_dices = []
        # tc_dices = []
        # et_dices = []
        # wt_sensitivities = []
        # tc_sensitivities = []
        # et_sensitivities = []
        # wt_ppvs = []
        # tc_ppvs = []
        # et_ppvs = []
        # wt_Hausdorf = []
        # tc_Hausdorf = []
        # et_Hausdorf = []
        #
        # wtMaskList = []
        # tcMaskList = []
        # etMaskList = []
        # wtPbList = []
        # tcPbList = []
        # etPbList = []

        maskPath = glob("./datasets/BraTs2019/rgb_results/attunet/" + "*gt.png")  ###
        saved_list = [

                      "/home/jyh_temp1/Downloads/BRaTS2d_send/UNet2D_BraTs-master/datasets/BraTs2019/rgb_results/unet_ET0.70",
                      "/home/jyh_temp1/Downloads/BRaTS2d_send/UNet2D_BraTs-master/datasets/BraTs2019/rgb_results/unet++",
                      "/home/jyh_temp1/Downloads/BRaTS2d_send/UNet2D_BraTs-master/datasets/BraTs2019/rgb_results/attunet",
                      "/home/jyh_temp1/Downloads/BRaTS2d_send/UNet2D_BraTs-master/datasets/BraTs2019/rgb_results/swinunet7686ET",
                      "/home/jyh_temp1/Downloads/BRaTS2d_send/UNet2D_BraTs-master/datasets/BraTs2019/rgb_results/swinunet_ET78.17_23.10.11",
                      "/home/jyh_temp1/Downloads/BRaTS2d_send/UNet2D_BraTs-master/datasets/BraTs2019/rgb_results/msugnet_best"]
        pbPath = glob("./datasets/BraTs2019/rgb_results/%s/" % args.name + "*.png")
        
        saved_list = ["/home/jyh_temp1/Downloads/BRaTS2d_send/UNet2D_BraTs-master/datasets/BraTs2019/rgb_results/unet"]
        for model_path in saved_list:
            wt_dices = []
            tc_dices = []
            et_dices = []
            wt_sensitivities = []
            tc_sensitivities = []
            et_sensitivities = []
            wt_ppvs = []
            tc_ppvs = []
            et_ppvs = []
            wt_Hausdorf = []
            tc_Hausdorf = []
            et_Hausdorf = []
            pbPath = glob(model_path+"/"+"*.png")
            maskPath.sort()
            pbPath.sort()
            if len(maskPath) == 0:
                print("请先生成图片!")
                return
            for myi in tqdm(range(len(maskPath))):
                # mask = imread(maskPath[myi])
                # pb = imread(pbPath[myi])
                ##single index
                mask = imread("./datasets/BraTs2019/rgb_results/attunet/" + str(myi)+"gt.png")
                pb = imread(model_path + "/" + str(myi) + ".png")
                # pb = imread("./datasets/BraTs2019/rgb_results/"+str(args.name)+"/" + str(myi) + ".png")
                wtmaskregion = np.zeros([mask.shape[0], mask.shape[1]], dtype=np.float32)
                wtpbregion = np.zeros([mask.shape[0], mask.shape[1]], dtype=np.float32)

                tcmaskregion = np.zeros([mask.shape[0], mask.shape[1]], dtype=np.float32)
                tcpbregion = np.zeros([mask.shape[0], mask.shape[1]], dtype=np.float32)

                etmaskregion = np.zeros([mask.shape[0], mask.shape[1]], dtype=np.float32)
                etpbregion = np.zeros([mask.shape[0], mask.shape[1]], dtype=np.float32)

                for idx in range(mask.shape[0]):
                    for idy in range(mask.shape[1]):
                        # 只要这个像素的任何一个通道有值,就代表这个像素不属于前景,即属于WT区域;yellow=255,255,0; red=255,0,0;green=0,128,0
                        if mask[idx, idy, :].any() != 0:
                            wtmaskregion[idx, idy] = 1
                        if pb[idx, idy, :].any() != 0:
                            wtpbregion[idx, idy] = 1
                        # 只要第一个通道是255,即可判断是TC区域,因为红色和黄色的第一个通道都是255,区别于绿色
                        if mask[idx, idy, 0] == 255:
                            tcmaskregion[idx, idy] = 1
                        if pb[idx, idy, 0] == 255:
                            tcpbregion[idx, idy] = 1
                        # 只要第二个通道是128,即可判断是ET区域
                        if mask[idx, idy, 1] == 128:
                            etmaskregion[idx, idy] = 1
                        if pb[idx, idy, 1] == 128:
                            etpbregion[idx, idy] = 1
                #开始计算WT
                dice = dice_coef(wtpbregion,wtmaskregion)
                wt_dices.append(dice)
                ppv_n = ppv(wtpbregion, wtmaskregion)
                wt_ppvs.append(ppv_n)
                Hausdorff = hausdorff_distance(wtmaskregion, wtpbregion)
                wt_Hausdorf.append(Hausdorff)
                sensitivity_n = sensitivity(wtpbregion, wtmaskregion)
                wt_sensitivities.append(sensitivity_n)
                # 开始计算TC
                dice = dice_coef(tcpbregion, tcmaskregion)
                tc_dices.append(dice)
                ppv_n = ppv(tcpbregion, tcmaskregion)
                tc_ppvs.append(ppv_n)
                Hausdorff = hausdorff_distance(tcmaskregion, tcpbregion)
                tc_Hausdorf.append(Hausdorff)
                sensitivity_n = sensitivity(tcpbregion, tcmaskregion)
                tc_sensitivities.append(sensitivity_n)
                # 开始计算ET
                dice = dice_coef(etpbregion, etmaskregion)
                et_dices.append(dice)
                ppv_n = ppv(etpbregion, etmaskregion)
                et_ppvs.append(ppv_n)
                Hausdorff = hausdorff_distance(etmaskregion, etpbregion)
                et_Hausdorf.append(Hausdorff)
                sensitivity_n = sensitivity(etpbregion, etmaskregion)
                et_sensitivities.append(sensitivity_n)

            print("model:",model_path.split("/")[-1])
            print('WT Dice: %.4f' % np.mean(wt_dices),round(np.var(wt_dices),4))
            print('TC Dice: %.4f' % np.mean(tc_dices),round(np.var(tc_dices),4))
            print('ET Dice: %.4f' % np.mean(et_dices),round(np.var(et_dices),4))
            print("=============")
            print('WT PPV: %.4f' % np.mean(wt_ppvs),round(np.var(wt_ppvs),4))
            print('TC PPV: %.4f' % np.mean(tc_ppvs),round(np.var(tc_ppvs),4))
            print('ET PPV: %.4f' % np.mean(et_ppvs),round(np.var(et_ppvs),4))
            print("=============")
            print('WT sensitivity: %.4f' % np.mean(wt_sensitivities),round(np.var(wt_sensitivities),4))
            print('TC sensitivity: %.4f' % np.mean(tc_sensitivities),round(np.var(tc_sensitivities),4))
            print('ET sensitivity: %.4f' % np.mean(et_sensitivities),round(np.var(et_sensitivities),4))
            print("=============")
            print('WT Hausdorff: %.4f' % np.mean(wt_Hausdorf),round(np.var(wt_Hausdorf),4))
            print('TC Hausdorff: %.4f' % np.mean(tc_Hausdorf),round(np.var(tc_Hausdorf),4))
            print('ET Hausdorff: %.4f' % np.mean(et_Hausdorf),round(np.var(et_Hausdorf),4))
            print("=============")


if __name__ == '__main__':
    main( )
