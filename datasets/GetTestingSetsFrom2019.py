import pandas as pd
import numpy as np


def compare_site_names(one,two):
    df1 = pd.read_csv(one, header=None)
    df2 = pd.read_csv(two, header=None)
    data1 = df1.to_dict(orient='list')[0]
    data2 = df2.to_dict(orient='list')[0]
    diff1 = set(data1).difference(data2)
    diff2 = set(data2).difference(data1)
    return list(diff2)


hgg_diff = compare_site_names("datasets/18hgg.csv","datasets/19hgg.csv")
len(hgg_diff)
lgg_diff = compare_site_names("datasets/18lgg.csv","datasets/19lgg.csv")
len(lgg_diff)


flair_name = "_flair.nii.gz"
t1_name = "_t1.nii.gz"
t1ce_name = "_t1ce.nii.gz"
t2_name = "_t2.nii.gz"
mask_name = "_seg.nii.gz"


bratshgg_path = r"datasets/MICCAI_BraTS_2019_Data_Training/HGG"
bratslgg_path = r"datasets/MICCAI_BraTS_2019_Data_Training/LGG"


outputImg_path = r"datasets/BraTs2019/testImage"
outputMask_path = r"datasets/BraTs2019/testMask"


import os

if not os.path.exists(outputImg_path):
    os.mkdir(outputImg_path)
if not os.path.exists(outputMask_path):
    os.mkdir(outputMask_path)

pathhgg_list = []
pathlgg_list = []


for idx in range(len(hgg_diff)):
    mystr = "BraTS19" + hgg_diff[idx]
    pathhgg_list.append(mystr)

for idx in range(len(lgg_diff)):
    mystr = "BraTS19" + lgg_diff[idx]
    pathlgg_list.append(mystr)


import SimpleITK as sitk


def normalize(slice, bottom=99, down=1):
    """
    normalize image with mean and std for regionnonzero,and clip the value into range
    :param slice:
    :param bottom:
    :param down:
    :return:
    """
    b = np.percentile(slice, bottom)
    t = np.percentile(slice, down)
    slice = np.clip(slice, t, b)

    image_nonzero = slice[np.nonzero(slice)]
    if np.std(slice) == 0 or np.std(image_nonzero) == 0:
        return slice
    else:
        tmp = (slice - np.mean(image_nonzero)) / np.std(image_nonzero)
        # since the range of intensities is between 0 and 5000 ,
        # the min in the normalized slice corresponds to 0 intensity in unnormalized slice
        # the min is replaced with -9 just to keep track of 0 intensities
        # so that we can discard those intensities afterwards when sampling random patches
        tmp[tmp == tmp.min()] = -9 #黑色背景区域
        return tmp


def crop_ceter(img,croph,cropw):   
    #for n_slice in range(img.shape[0]):
    height,width = img[0].shape 
    starth = height//2-(croph//2)
    startw = width//2-(cropw//2)        
    return img[:,starth:starth+croph,startw:startw+cropw]


for subsetindex in range(len(pathhgg_list)):
    brats_subset_path = bratshgg_path + "/" + (pathhgg_list[subsetindex]) + "/"
    flair_image = brats_subset_path + (pathhgg_list[subsetindex]) + flair_name
    t1_image = brats_subset_path + (pathhgg_list[subsetindex]) + t1_name
    t1ce_image = brats_subset_path + (pathhgg_list[subsetindex]) + t1ce_name
    t2_image = brats_subset_path + (pathhgg_list[subsetindex]) + t2_name
    mask_image = brats_subset_path + (pathhgg_list[subsetindex]) + mask_name
    flair_src = sitk.ReadImage(flair_image, sitk.sitkInt16)
    t1_src = sitk.ReadImage(t1_image, sitk.sitkInt16)
    t1ce_src = sitk.ReadImage(t1ce_image, sitk.sitkInt16)
    t2_src = sitk.ReadImage(t2_image, sitk.sitkInt16)
    mask = sitk.ReadImage(mask_image, sitk.sitkUInt8)
    flair_array = sitk.GetArrayFromImage(flair_src)
    t1_array = sitk.GetArrayFromImage(t1_src)
    t1ce_array = sitk.GetArrayFromImage(t1ce_src)
    t2_array = sitk.GetArrayFromImage(t2_src)
    mask_array = sitk.GetArrayFromImage(mask)
    flair_array_nor = normalize(flair_array)
    t1_array_nor = normalize(t1_array)
    t1ce_array_nor = normalize(t1ce_array)
    t2_array_nor = normalize(t2_array)
    flair_crop = crop_ceter(flair_array_nor,160,160)
    t1_crop = crop_ceter(t1_array_nor,160,160)
    t1ce_crop = crop_ceter(t1ce_array_nor,160,160)
    t2_crop = crop_ceter(t2_array_nor,160,160)
    mask_crop = crop_ceter(mask_array,160,160) 
    print((pathhgg_list[subsetindex]))
    for n_slice in range(flair_crop.shape[0]):
        if np.max(mask_crop[n_slice,:,:]) != 0:
            maskImg = mask_crop[n_slice,:,:]
            
            FourModelImageArray = np.zeros((flair_crop.shape[1],flair_crop.shape[2],4),np.float64)
            flairImg = flair_crop[n_slice,:,:]
            flairImg = flairImg.astype(np.float64)
            FourModelImageArray[:,:,0] = flairImg
            t1Img = t1_crop[n_slice,:,:]
            t1Img = t1Img.astype(np.float64)
            FourModelImageArray[:,:,1] = t1Img
            t1ceImg = t1ce_crop[n_slice,:,:]
            t1ceImg = t1ceImg.astype(np.float64)
            FourModelImageArray[:,:,2] = t1ceImg
            t2Img = t2_crop[n_slice,:,:]
            t2Img = t2Img.astype(np.float64)
            FourModelImageArray[:,:,3] = t2Img       
        
            imagepath = outputImg_path + "//" + (pathhgg_list[subsetindex]) + "_" + str(n_slice) + ".npy"
            maskpath = outputMask_path + "//" + (pathhgg_list[subsetindex]) + "_" + str(n_slice) + ".npy"
            np.save(imagepath,FourModelImageArray)#(160,160,4) np.float dtype('float64')
            np.save(maskpath,maskImg)# (160, 160) dtype('uint8') 值为0 1 2 4
print("Done！")


for subsetindex in range(len(pathlgg_list)):
    brats_subset_path = bratslgg_path + "/" + (pathlgg_list[subsetindex]) + "/"
    flair_image = brats_subset_path + (pathlgg_list[subsetindex]) + flair_name
    t1_image = brats_subset_path + (pathlgg_list[subsetindex]) + t1_name
    t1ce_image = brats_subset_path + (pathlgg_list[subsetindex]) + t1ce_name
    t2_image = brats_subset_path + (pathlgg_list[subsetindex]) + t2_name
    mask_image = brats_subset_path + (pathlgg_list[subsetindex]) + mask_name
    flair_src = sitk.ReadImage(flair_image, sitk.sitkInt16)
    t1_src = sitk.ReadImage(t1_image, sitk.sitkInt16)
    t1ce_src = sitk.ReadImage(t1ce_image, sitk.sitkInt16)
    t2_src = sitk.ReadImage(t2_image, sitk.sitkInt16)
    mask = sitk.ReadImage(mask_image, sitk.sitkUInt8)
    flair_array = sitk.GetArrayFromImage(flair_src)
    t1_array = sitk.GetArrayFromImage(t1_src)
    t1ce_array = sitk.GetArrayFromImage(t1ce_src)
    t2_array = sitk.GetArrayFromImage(t2_src)
    mask_array = sitk.GetArrayFromImage(mask)
    flair_array_nor = normalize(flair_array)
    t1_array_nor = normalize(t1_array)
    t1ce_array_nor = normalize(t1ce_array)
    t2_array_nor = normalize(t2_array)
    flair_crop = crop_ceter(flair_array_nor,160,160)
    t1_crop = crop_ceter(t1_array_nor,160,160)
    t1ce_crop = crop_ceter(t1ce_array_nor,160,160)
    t2_crop = crop_ceter(t2_array_nor,160,160)
    mask_crop = crop_ceter(mask_array,160,160) 
    print((pathlgg_list[subsetindex]))
    for n_slice in range(flair_crop.shape[0]):
        if np.max(mask_crop[n_slice,:,:]) != 0:
            maskImg = mask_crop[n_slice,:,:]
            
            FourModelImageArray = np.zeros((flair_crop.shape[1],flair_crop.shape[2],4),np.float64)
            flairImg = flair_crop[n_slice,:,:]
            flairImg = flairImg.astype(np.float64)
            FourModelImageArray[:,:,0] = flairImg
            t1Img = t1_crop[n_slice,:,:]
            t1Img = t1Img.astype(np.float64)
            FourModelImageArray[:,:,1] = t1Img
            t1ceImg = t1ce_crop[n_slice,:,:]
            t1ceImg = t1ceImg.astype(np.float64)
            FourModelImageArray[:,:,2] = t1ceImg
            t2Img = t2_crop[n_slice,:,:]
            t2Img = t2Img.astype(np.float64)
            FourModelImageArray[:,:,3] = t2Img       
        
            imagepath = outputImg_path + "//" + (pathlgg_list[subsetindex]) + "_" + str(n_slice) + ".npy"
            maskpath = outputMask_path + "//" + (pathlgg_list[subsetindex]) + "_" + str(n_slice) + ".npy"
            np.save(imagepath,FourModelImageArray)#(160,160,4) np.float dtype('float64')
            np.save(maskpath,maskImg)# (160, 160) dtype('uint8') 值为0 1 2 4
print("Done！")

