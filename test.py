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
from train import DISTRIBUTED, USE_AMP, COMPILE

SAVE_GT = False

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


def create_rgb_image(data, shape, label_mapping):
    """
    Create an RGB image based on given data and label mapping.

    Args:
        data (numpy.ndarray): The input data, typically a mask or model output.
        shape (tuple): The desired output image shape (height, width, channels).
        label_mapping (dict): A dictionary mapping conditions to RGB colors.

    Returns:
        numpy.ndarray: The generated RGB image.
    """

    rgb_image = np.zeros(shape, dtype=np.uint8)
    for label, color in label_mapping.items():
        mask = label(data)  # Apply label-specific mask

        rgb_image[mask] = color
    return rgb_image


def process_output(output, img_paths, index, args, plt_test=False):
    """
    Process and save model output as RGB images.

    Args:
        output (numpy.ndarray): The model's output.
        img_paths (list): List of image file paths corresponding to the batch.
        index (int): Batch index for naming files.
        args: Additional arguments, including `args.name`.
    """
    os.makedirs(f'datasets/BraTs2019/rgb_results/{args.name}/', exist_ok=True)

    for i, img_path in enumerate(img_paths):  # Handle each path in the list
        np_name = os.path.basename(img_path)
        rgb_name = os.path.splitext(np_name)[0] + ".png"

        label_mapping = {
            lambda data: data[0] > 0.5: [0, 128, 0],   # Green
            lambda data: data[1] > 0.5: [255, 0, 0],  # Red
            lambda data: data[2] > 0.5: [255, 255, 0] # Yellow
        }
        rgb_pic = create_rgb_image(output[i], (output.shape[2], output.shape[3], 3), label_mapping)

        output_path = f'datasets/BraTs2019/rgb_results/{args.name}/{index*args.batch_size+i}.png'
        imageio.imwrite(output_path, rgb_pic)
    

def save_ground_truth(args, val_mask_paths, plt_test=False, select_index=None):
    save_path = f'datasets/BraTs2019/rgb_results/{args.name}/'
    os.makedirs(save_path, exist_ok=True)
    mask_paths = [val_mask_paths[select_index]] if plt_test else val_mask_paths

    label_mapping = {
        lambda data: data == 1: [255, 0, 0],   # NET
        lambda data: data == 2: [0, 128, 0],  # ED
        lambda data: data == 4: [255, 255, 0] # ET
    }

    for i, mask_path in enumerate(tqdm(mask_paths, desc="Saving GT")):
        name = os.path.basename(mask_path)
        rgb_name = f"{i}gt.png"
        npmask = np.load(mask_path)

        gt_color = create_rgb_image(npmask, (npmask.shape[0], npmask.shape[1], 3), label_mapping)
        imageio.imwrite(f'{save_path}/{rgb_name}', gt_color)
        

def main():
    args = parse_args()

    if not os.path.exists('output/%s' %args.name):
        os.makedirs('output/%s' %args.name)

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')

    # create model
    print("=> creating model %s" % args.name)
    if args.name == "vgunet":
        model = VGUNet.load(in_ch=4, out_ch=3)
    model = model.cuda()

    # Data loading code
    img_paths = glob(r'./datasets/BraTs2019/testImage/*')
    mask_paths = glob(r'./datasets/BraTs2019/testMask/*')
    val_img_paths = img_paths
    val_mask_paths = mask_paths

    print("testing mode:",args.mode)
    if args.mode == "GetPicture":
        model = model.cuda()
        if DISTRIBUTED:
            model = nn.DataParallel(model)
        if COMPILE:
            model = torch.compile(model)
        model.eval()
        test_dataset = Dataset(args, val_img_paths, val_mask_paths)
        plt_test = False
        if plt_test == True:
            select_index = 40  # ,54,26 long distance,40 two class
            test_dataset = Dataset(args, img_paths, mask_paths, select_index)

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True,drop_last=False)
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            with torch.no_grad():
                for i, (input, target) in tqdm(enumerate(test_loader), total=len(test_loader)):
                    index=i
                    input = input.cuda()
                    with torch.cuda.amp.autocast():
                        output = model(input)
                        output = torch.sigmoid(output).data.cpu().numpy()
                    img_paths = val_img_paths[args.batch_size*i:args.batch_size*(i+1)]
                    process_output(output, img_paths, index, args)

            torch.cuda.empty_cache()
            
        
        if SAVE_GT:
            save_ground_truth(args, val_mask_paths)
            
        print("Done!")

    if args.mode == "Calculate":
        """
        Dice, Sensitivity, PPV
        """

        maskPath = glob("./datasets/BraTs2019/rgb_results/vgunet/" + "*gt.png")
        pbPath = glob("./datasets/BraTs2019/rgb_results/%s/" % args.name + "*.png")
        
        saved_list = ["./datasets/BraTs2019/rgb_results/vgunet"]
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
                mask = imread("./datasets/BraTs2019/rgb_results/vgunet/" + str(myi)+"gt.png")
                pb = imread(model_path + "/" + str(myi) + ".png")
                
                wtmaskregion = (mask.sum(axis=-1) != 0).astype(np.float32)  # Any non-zero pixel
                wtpbregion = (pb.sum(axis=-1) != 0).astype(np.float32)

                tcmaskregion = (mask[:, :, 0] == 255).astype(np.float32)  # Red channel
                tcpbregion = (pb[:, :, 0] == 255).astype(np.float32)

                etmaskregion = (mask[:, :, 1] == 128).astype(np.float32)  # Green channel
                etpbregion = (pb[:, :, 1] == 128).astype(np.float32)

                #WT
                dice = dice_coef(wtpbregion,wtmaskregion)
                wt_dices.append(dice)
                ppv_n = ppv(wtpbregion, wtmaskregion)
                wt_ppvs.append(ppv_n)
                Hausdorff = hausdorff_distance(wtmaskregion, wtpbregion)
                wt_Hausdorf.append(Hausdorff)
                sensitivity_n = sensitivity(wtpbregion, wtmaskregion)
                wt_sensitivities.append(sensitivity_n)
                # TC
                dice = dice_coef(tcpbregion, tcmaskregion)
                tc_dices.append(dice)
                ppv_n = ppv(tcpbregion, tcmaskregion)
                tc_ppvs.append(ppv_n)
                Hausdorff = hausdorff_distance(tcmaskregion, tcpbregion)
                tc_Hausdorf.append(Hausdorff)
                sensitivity_n = sensitivity(tcpbregion, tcmaskregion)
                tc_sensitivities.append(sensitivity_n)
                # ET
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
