import argparse
import os
import re
import shutil

import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm
from networks.deeplabv3 import *
# from networks.efficientunet import UNet
from networks.net_factory import net_factory
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default="Entropy_Minimization", help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--fold', type=str,
                    default='fold4', help='fold')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument('--sup_type', type=str, default="scribble",
                    help='label')


def get_fold_ids(fold):
    all_cases_set = ["patient{:0>3}".format(i) for i in range(1, 101)]
    fold1_testing_set = [
        "patient{:0>3}".format(i) for i in range(1, 21)]
    fold1_training_set = [
        i for i in all_cases_set if i not in fold1_testing_set]

    fold2_testing_set = [
        "patient{:0>3}".format(i) for i in range(11, 31)]
    fold2_training_set = [
        i for i in all_cases_set if i not in fold2_testing_set]

    fold3_testing_set = [
        "patient{:0>3}".format(i) for i in range(41, 61)]
    fold3_training_set = [
        i for i in all_cases_set if i not in fold3_testing_set]

    fold4_testing_set = [
        "patient{:0>3}".format(i) for i in range(61, 81)]
    fold4_training_set = [
        i for i in all_cases_set if i not in fold4_testing_set]

    fold5_testing_set = [
        "patient{:0>3}".format(i) for i in range(81, 101)]
    fold5_training_set = [
        i for i in all_cases_set if i not in fold5_testing_set]
    if fold == "fold1":
        return [fold1_training_set, fold1_testing_set]
    elif fold == "fold2":
        return [fold2_training_set, fold2_testing_set]
    elif fold == "fold3":
        return [fold3_training_set, fold3_testing_set]
    elif fold == "fold4":
        return [fold4_training_set, fold4_testing_set]
    elif fold == "fold5":
        return [fold5_training_set, fold5_testing_set]
    else:
        return "ERROR KEY"


def calculate_metric_percase(pred, gt, spacing):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    #asd = metric.binary.asd(pred, gt, voxelspacing=spacing)
    asd = metric.binary.asd(pred, gt, voxelspacing=spacing)
    hd95 =metric.binary.hd95(pred, gt, voxelspacing=spacing)
    #hd95 = metric.binary.hd95(pred, gt, voxelspacing=spacing)
    return dice, hd95, asd

def test_single_volume_superpixel(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            slice = zoom(
                slice, (patch_size[0] / x, patch_size[1] / y), order=0)
            superpixel_slice = image2[ind, :, :]
        
            input = torch.from_numpy(slice).unsqueeze(
                0).float().cuda()
            
            net.eval()
            with torch.no_grad():
                out = torch.argmax(torch.softmax(
                    net(input), dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                pred = zoom(
                    out, (x / patch_size[0], y / patch_size[1]), order=0)
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(
                net(input)[0], dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list
def test_single_volume(case, net, test_save_path, FLAGS):
    h5f = h5py.File(FLAGS.root_path +
                    "/ACDC_training_volumes/{}".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (256 / x, 256 / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        #input = torch.cat([input] * 3, dim=1)
        net.eval()
        with torch.no_grad():
            out_main = net(input)
            out_main = out_main[0]
            out = torch.argmax(torch.softmax(
                out_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / 256, y / 256), order=0)
            prediction[ind] = pred
    case = case.replace(".h5", "")
    print(case)
    print(case[0:10])
    
    org_img_path = "training/{}".format(case[0:10])
    org_img_path = org_img_path + "/{}.nii.gz".format(case)
    org_img_itk = sitk.ReadImage(org_img_path)
    spacing = org_img_itk.GetSpacing()

    first_metric = calculate_metric_percase(
        prediction == 1, label == 1, (spacing[2], spacing[0], spacing[1]))
    second_metric = calculate_metric_percase(
        prediction == 2, label == 2, (spacing[2], spacing[0], spacing[1]))
    third_metric = calculate_metric_percase(
        prediction == 3, label == 3, (spacing[2], spacing[0], spacing[1]))

    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.CopyInformation(org_img_itk)
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.CopyInformation(org_img_itk)
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.CopyInformation(org_img_itk)
    sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
    sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
    sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
    return first_metric, second_metric, third_metric


def Inference(FLAGS):
    train_ids, test_ids = get_fold_ids(FLAGS.fold)
    all_volumes = os.listdir(
        FLAGS.root_path + "/ACDC_training_volumes")
    image_list = []
    for ids in test_ids:
        new_data_list = list(filter(lambda x: re.match(
            '{}.*'.format(ids), x) != None, all_volumes))
        image_list.extend(new_data_list)
    snapshot_path = "iter_40000_dice_0.8763.pth"
    test_save_path = "prediction/"
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    net =net_factory(net_type='BoundaryUNet', in_chns=1, class_num=4)
    
    if FLAGS.fold == "fold6":
        save_mode_path = os.path.join(snapshot_path, 'iter_39000.pth')
    else:
        save_mode_path = os.path.join(snapshot_path)
    
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()

    first_total = 0.0
    second_total = 0.0
    third_total = 0.0
    metric_dice = []
    metric_hd = []
    metric_asd = []
    metric_dice2 = []
    metric_hd2 = []
    metric_asd2 = []
    metric_dice3 = []
    metric_hd3 = []
    metric_asd3 = []
    for case in tqdm(image_list):
        #print(case)
        first_metric, second_metric, third_metric = test_single_volume(
            case, net, test_save_path, FLAGS)
        first_total += np.asarray(first_metric)
        second_total += np.asarray(second_metric)
        third_total += np.asarray(third_metric)
        metric_dice.append(first_metric[0])
        metric_dice2.append(second_metric[0])
        metric_dice3.append(third_metric[0])
        metric_hd.append(first_metric[1])
        metric_hd2.append(second_metric[1])
        metric_hd3.append(third_metric[1])
        metric_asd.append(first_metric[2])
        metric_asd2.append(second_metric[2])
        metric_asd3.append(third_metric[2])
    avg_metric = [first_total / len(image_list), second_total /
                  len(image_list), third_total / len(image_list)]
    std1 = [np.std(metric_dice), np.std(metric_hd), np.std(metric_asd)]
    std2 = [np.std(metric_dice2), np.std(metric_hd2), np.std(metric_asd2)]
    std3 = [np.std(metric_dice3), np.std(metric_hd3), np.std(metric_asd3)]
    print(avg_metric)
    print(std1)
    print(std2)
    print(std3)
    print((avg_metric[0] + avg_metric[1] + avg_metric[2]) / 3)
    return avg_metric


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    total = 0.0
    total_dice1 =0.0
    total_hd1 =0.0
    total_dice2 =0.0
    total_hd2 =0.0
    total_dice3 =0.0
    total_hd3 =0.0
    total_std_dice1 =0.0
    total_std_hd1 =0.0
    total_std_dice2 =0.0
    total_std_hd2 =0.0
    total_std_dice3 =0.0
    total_std_hd3 =0.0
    total_dice =0.0
    total_hd =0.0
    for i in [1]:
        # for i in [5]:
        FLAGS.fold = "fold{}".format(i)
        print(FLAGS.fold)
        print("Inference fold{}".format(i))
        metrics = Inference(FLAGS)
       