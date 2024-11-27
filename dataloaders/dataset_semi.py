import itertools
import os
import random
import re
from glob import glob

import cv2
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
random.seed(42)

class BaseDataSets(Dataset):
    def __init__(self, base_dir=None, num=4, labeled_type="labeled", split='train', transform=None, fold="fold1", sup_type="label"):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.sup_type = sup_type
        self.transform = transform
        self.num = num
        self.labeled_type = labeled_type
        self.ignore_class = 4
        train_ids, val_ids = self._get_fold_ids(fold)
        #all_labeled_ids = ["patient{:0>3}".format(
            #10 * i) for i in range(1, 11)]
      
        all_patient_ids = train_ids
        num_samples = int(len(all_patient_ids) * 0.05) 
        all_labeled_ids = random.sample(all_patient_ids, num_samples)
        if self.split == 'train':
            self.all_slices = os.listdir(
                self._base_dir + "/ACDC_training_slices")
            self.sample_list = []
            labeled_ids = [i for i in all_labeled_ids if i in train_ids]
            unlabeled_ids = [i for i in train_ids if i not in labeled_ids]
            if self.labeled_type == "labeled":
                print("Labeled patients IDs", labeled_ids)
                for ids in labeled_ids:
                    new_data_list = list(filter(lambda x: re.match(
                        '{}.*'.format(ids), x) != None, self.all_slices))
                    self.sample_list.extend(new_data_list)
                print("total labeled {} samples".format(len(self.sample_list)))
            else:
                print("Unlabeled patients IDs", unlabeled_ids)
                for ids in unlabeled_ids:
                    new_data_list = list(filter(lambda x: re.match(
                        '{}.*'.format(ids), x) != None, self.all_slices))
                    self.sample_list.extend(new_data_list)
                print("total unlabeled {} samples".format(len(self.sample_list)))

        elif self.split == 'val':
            self.all_volumes = os.listdir(
                self._base_dir + "/ACDC_training_volumes")
            self.sample_list = []
            for ids in val_ids:
                new_data_list = list(filter(lambda x: re.match(
                    '{}.*'.format(ids), x) != None, self.all_volumes))
                self.sample_list.extend(new_data_list)

        # if num is not None and self.split == "train":
        #     self.sample_list = self.sample_list[:num]

    def _get_fold_ids(self, fold):
        all_cases_set = ["patient{:0>3}".format(i) for i in range(1, 101)]
        fold1_testing_set = [
            "patient{:0>3}".format(i) for i in range(1, 21)]
        fold1_training_set = [
            i for i in all_cases_set if i not in fold1_testing_set]

        fold2_testing_set = [
            "patient{:0>3}".format(i) for i in range(21, 41)]
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
    def get_cls_label(self, cls_label):
        cls_label_set = list(cls_label)

        if self.ignore_class in cls_label_set:
            cls_label_set.remove(self.ignore_class)
        if 255 in cls_label_set:
            cls_label_set.remove(255)

        cls_label = np.zeros(self.ignore_class)
        for i in cls_label_set:
            cls_label[i] += 1
        cls_label = torch.from_numpy(cls_label).float()
        return cls_label
    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            #h5f = h5py.File(self._base_dir +
                            #"/ACDC_training_slices/{}".format(case), 'r')
            h5f = h5py.File("/home/DATAsda/zmdata/WSL4MIS/data/ACDC_add_Super/ACDC_training_slices/{}".format(case), 'r')
        else:
            h5f = h5py.File(self._base_dir +
                            "/ACDC_training_volumes/{}".format(case), 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        sample = {'image': image, 'label': label}
        if self.split == "train":
            image = h5f['image'][:]
            label = h5f[self.sup_type][:]
            super_label = np.array(h5f['super_scribble'][:])
            sample = {'image': image, 'label': label, 'super_label': super_label }
            sample = self.transform(sample)
        else:
            image = h5f['image'][:]
            label = h5f['label'][:]
            sample = {'image': image, 'label': label }
        sample["idx"] = case.split("_")[0]
        return sample


def random_rot_flip(image, label, super_label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    super_label = np.rot90(super_label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    super_label = np.flip(super_label, axis=axis).copy()
    
    return image, label,super_label

def random_rotate(image, label, super_label, cval):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0,
                           reshape=False, mode="constant", cval=cval)
    super_label = ndimage.rotate(super_label, angle, order=0,
                           reshape=False, mode="constant", cval=cval)
    
    return image, label,super_label 

class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size
        
    def get_boundary(self,mask):
    # 将 mask 转换为灰度图像
        mask_gray = np.argmax(mask, axis=1)  # 在通道维度上找到最大值所在的索引，即类别标签
        mask_gray = np.uint8(mask_gray)  # 将数据类型转换为 uint8

    # 初始化空白图像用于存储边界信息
        boundary = np.zeros_like(mask_gray)

    # 对每个 batch 中的每个图像提取边界信息
        for i in range(mask_gray.shape[0]):
        # 使用 Canny 边缘检测算法提取边界
            edges = cv2.Canny(mask_gray[i], 0, 1)  # 第二个参数是低阈值，第三个参数是高阈值

        # 将边界信息叠加到结果图像中
            boundary[i] = edges

        return boundary

    def __call__(self, sample):
        image, label,super_label = sample['image'], sample['label'], sample['super_label']
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]
        if random.random() > 0.5:
            image, label, super_label = random_rot_flip(image, label, super_label)
        elif random.random() > 0.5:
            if 4 in np.unique(label):
                image, label,super_label = random_rotate(image, label, super_label, cval=4)
            else:
                image, label,super_label = random_rotate(image, label,super_label, cval=0)
        x, y = image.shape
        image = zoom(
            image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(
            label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        super_label = zoom(
            super_label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(
            image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        super_label = torch.from_numpy(super_label.astype(np.uint8))        
        sample = {'image': image, 'label': label,'super_label': super_label}
        return sample


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
