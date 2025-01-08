import os
import numpy as np
import SimpleITK as sitk

flair_name = "_flair.nii.gz"
t1_name = "_t1.nii.gz"
t1ce_name = "_t1ce.nii.gz"
t2_name = "_t2.nii.gz"
mask_name = "_seg.nii.gz"

bratshgg_path = r"./datasets/2-MICCAI_BraTS_2018/MICCAI_BraTS_2018_Data_Training/HGG"
bratslgg_path = r"./datasets/2-MICCAI_BraTS_2018/MICCAI_BraTS_2018_Data_Training/LGG"

outputImg_path = r"./datasets/2-MICCAI_BraTS_2018/BraTS2018_trainImage"
outputMask_path = r"./datasets/2-MICCAI_BraTS_2018/BraTS2018_trainMask"


def file_name_path(file_dir, dir=True, file=False):
    """
    get root path,sub_dirs,all_sub_files
    :param file_dir:
    :return: dir or file
    """
    for root, dirs, files in os.walk(file_dir):
        if len(dirs) and dir:
            print("sub_dirs:", dirs)
            return dirs
        if len(files) and file:
            print("files:", files)
            return files

if not os.path.exists(outputImg_path):
    os.mkdir(outputImg_path)
if not os.path.exists(outputMask_path):
    os.mkdir(outputMask_path)

pathhgg_list = file_name_path(bratshgg_path)
pathlgg_list = file_name_path(bratslgg_path)


def normalize(slice, bottom=99, down=1):
    """
    normalize image with mean and std for regionnonzero,and clip the value into range
    :param slice:
    :param bottom:
    :param down:
    :return:
    """
    #有点像“去掉最低分去掉最高分”的意思,使得数据集更加“公平”
    b = np.percentile(slice, bottom)
    t = np.percentile(slice, down)
    slice = np.clip(slice, t, b)#限定范围numpy.clip(a, a_min, a_max, out=None)

    #除了黑色背景外的区域要进行标准化
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
    height,width = img[0].shape 
    starth = height//2-(croph//2)
    startw = width//2-(cropw//2)        
    return img[:,starth:starth+croph,startw:startw+cropw]

from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from tqdm import tqdm

# Functions to process each case
def process_case(case_index, path_list, base_path, output_img_path, output_mask_path, flair_name, t1_name, t1ce_name, t2_name, mask_name):
    brats_subset_path = os.path.join(base_path, str(path_list[case_index]))
    flair_image = os.path.join(brats_subset_path, str(path_list[case_index]) + flair_name)
    t1_image = os.path.join(brats_subset_path, str(path_list[case_index]) + t1_name)
    t1ce_image = os.path.join(brats_subset_path, str(path_list[case_index]) + t1ce_name)
    t2_image = os.path.join(brats_subset_path, str(path_list[case_index]) + t2_name)
    mask_image = os.path.join(brats_subset_path, str(path_list[case_index]) + mask_name)

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

    flair_crop = crop_ceter(flair_array_nor, 160, 160)
    t1_crop = crop_ceter(t1_array_nor, 160, 160)
    t1ce_crop = crop_ceter(t1ce_array_nor, 160, 160)
    t2_crop = crop_ceter(t2_array_nor, 160, 160)
    mask_crop = crop_ceter(mask_array, 160, 160)

    for n_slice in range(flair_crop.shape[0]):
        if np.max(mask_crop[n_slice, :, :]) != 0:
            mask_img = mask_crop[n_slice, :, :]
            FourModelImageArray = np.zeros((flair_crop.shape[1], flair_crop.shape[2], 4), np.float64)

            FourModelImageArray[:, :, 0] = flair_crop[n_slice, :, :].astype(np.float64)
            FourModelImageArray[:, :, 1] = t1_crop[n_slice, :, :].astype(np.float64)
            FourModelImageArray[:, :, 2] = t1ce_crop[n_slice, :, :].astype(np.float64)
            FourModelImageArray[:, :, 3] = t2_crop[n_slice, :, :].astype(np.float64)

            image_path = os.path.join(output_img_path, f"{path_list[case_index]}_{n_slice}.npy")
            mask_path = os.path.join(output_mask_path, f"{path_list[case_index]}_{n_slice}.npy")
            np.save(image_path, FourModelImageArray)
            np.save(mask_path, mask_img)

    return f"Processed case {path_list[case_index]}"

def process_dataset(path_list, base_path, output_img_path, output_mask_path, flair_name, t1_name, t1ce_name, t2_name, mask_name, num_workers=None):
    if num_workers is None:
        num_workers = cpu_count() * 2
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for index in range(len(path_list)):
            futures.append(
                executor.submit(process_case, index, path_list, base_path, output_img_path, output_mask_path, flair_name, t1_name, t1ce_name, t2_name, mask_name)
            )

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing cases"):
            pass

process_dataset(pathhgg_list, bratshgg_path, outputImg_path, outputMask_path, flair_name, t1_name, t1ce_name, t2_name, mask_name)
process_dataset(pathlgg_list, bratslgg_path, outputImg_path, outputMask_path, flair_name, t1_name, t1ce_name, t2_name, mask_name)
print("Done!")