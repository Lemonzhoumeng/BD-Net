#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 15:52:39 2023

@author: zhoumeng
"""

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
from skimage import color

def pseudo_label_generator_acdc(data, seed, beta=100, mode='bf'):
    from skimage.exposure import rescale_intensity
    from skimage.segmentation import random_walker
    if 1 not in np.unique(seed) or 2 not in np.unique(seed) or 3 not in np.unique(seed):
        pseudo_label = np.zeros_like(seed)
    else:
        markers = np.ones_like(seed)
        markers[seed == 4] = 0
        markers[seed == 0] = 1
        markers[seed == 1] = 2
        markers[seed == 2] = 3
        markers[seed == 3] = 4
        sigma = 0.35
        data = rescale_intensity(data, in_range=(-sigma, 1 + sigma),
                                 out_range=(-1, 1))
        segmentation = random_walker(data, markers, beta, mode)
        pseudo_label = segmentation - 1
    return pseudo_label


class BaseDataSets(Dataset):
    def __init__(self, base_dir=None, split='train', transform=None, fold="fold1", sup_type="label"):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.sup_type = sup_type
        self.transform = transform
        train_ids, test_ids = self._get_fold_ids(fold)
        if self.split == 'train':
            self.all_slices = os.listdir(
                self._base_dir + "/ACDC_add_Super/ACDC_training_slices")
            self.sample_list = []
            for ids in train_ids:
                new_data_list = list(filter(lambda x: re.match(
                    '{}.*'.format(ids), x) != None, self.all_slices))
                self.sample_list.extend(new_data_list)

        elif self.split == 'val':
            self.all_volumes = os.listdir(
                self._base_dir + "/ACDC/ACDC_training_volumes")
            self.sample_list = []
            for ids in test_ids:
                new_data_list = list(filter(lambda x: re.match(
                    '{}.*'.format(ids), x) != None, self.all_volumes))
                self.sample_list.extend(new_data_list)

        # if num is not None and self.split == "train":
        #     self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

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

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir +
                            "/ACDC_add_Super/ACDC_training_slices/{}".format(case), 'r')
            h5f2 = h5py.File(self._base_dir +
                            "/ACDC_add_Super_M150_casuper/ACDC_training_slices/{}".format(case), 'r')
        else:
            h5f = h5py.File(self._base_dir +
                            "/ACDC/ACDC_training_volumes/{}".format(case), 'r')

        #image = h5f['image'][:]
        #label = h5f['label'][:]
        #super_label = np.array(h5f['super_scribble'][:])
        #sample = {'image': image, 'label': label}
        if self.split == "train":
            image = h5f['image'][:]
            img = h5f2['image_superpixel'][:]
            img_scaled = cv2.convertScaleAbs(img, alpha=(255.0/np.max(img)))
            image2= cv2.convertScaleAbs(img_scaled)
            image2 = cv2.cvtColor(image2, cv2.COLOR_LAB2RGB)
            image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
            if self.sup_type == "random_walker":
                full_label = np.array(h5f['label'][:])
                label1 = pseudo_label_generator_acdc(image, h5f["scribble"][:])
                #label1 = np.array(h5f['super_scribble'][:])
                
                
            else:
                label = h5f[self.sup_type][:]
                super_label = np.array(h5f['super_scribble'][:])
                
                
            sample = {'image': image, 'image2': image2, 'label': label,'super_label': super_label}
            sample = self.transform(sample)
        else:
            image = h5f['image'][:]
            label = h5f['label'][:]
            arr_list = []
            filename_prefix = case[:18] + "_slice_"
            for ind in range(image.shape[0]):
                slice = image[ind, :, :]
                new_case = filename_prefix + str(ind) + ".h5"
                h5f2 = h5py.File(self._base_dir +
                            "/ACDC_add_Super_M150_casuper/ACDC_training_slices/{}".format(new_case), 'r')
                img = h5f2['image_superpixel'][:]
                img_scaled = cv2.convertScaleAbs(img, alpha=(255.0/np.max(img)))
                image2= cv2.convertScaleAbs(img_scaled)
                image2 = cv2.cvtColor(image2, cv2.COLOR_LAB2RGB)
                image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
                arr_list.append(image2)
            image2 = np.stack(arr_list)
            sample = {'image': image, 'image2' : image2, 'label': label,'super_label': super_label }
        sample["idx"] = idx
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
    
def random_rot_flip2(image,image2,label,super_label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    image2 = np.rot90(image2, k)
    label = np.rot90(label, k)
    super_label = np.rot90(super_label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    image2 = np.flip(image2, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    super_label = np.flip(super_label, axis=axis).copy()
    return image,image2, label,super_label

def random_rotate(image, label, super_label, cval):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0,
                           reshape=False, mode="constant", cval=cval)
    super_label = ndimage.rotate(super_label, angle, order=0,
                           reshape=False, mode="constant", cval=cval)
    return image, label,super_label
    
def random_rotate2(image,image2, label,super_label, cval):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    image2 = ndimage.rotate(image2, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0,
                           reshape=False, mode="constant", cval=cval)
    super_label = ndimage.rotate(super_label, angle, order=0,
                           reshape=False, mode="constant", cval=cval)
    return image,image2, label,super_label


class RandomGenerator2(object):
    def __init__(self, output_size1, output_size2, output_size3):
        self.output_size1 = output_size1
        self.output_size2 = output_size2
        self.output_size3 = output_size3

    def __call__(self, sample):
        image, label,super_label = sample['image'], sample['label'], sample['super_label']
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]
        if random.random() > 0.5:
            image, label, super_label = random_rot_flip(image, label,super_label)
        elif random.random() > 0.5:
            if 4 in np.unique(label):
                image, label,super_label = random_rotate(image, label,super_label, cval=4)
            else:
                image, label,super_label = random_rotate(image, label,super_label, cval=0)
        x, y = image.shape
        image1 = zoom(
            image, (self.output_size1[0] / x, self.output_size1[1] / y), order=0)
        label1 = zoom(
            label, (self.output_size1[0] / x, self.output_size1[1] / y), order=0)
        super_label1 = zoom(
            super_label, (self.output_size1[0] / x, self.output_size1[1] / y), order=0)
        image1 = torch.from_numpy(
            image1.astype(np.float32)).unsqueeze(0)
        label1 = torch.from_numpy(label1.astype(np.uint8))
        super_label1 = torch.from_numpy(super_label1.astype(np.uint8))
        
        image2 = zoom(
            image, (self.output_size2[0] / x, self.output_size2[1] / y), order=0)
        label2 = zoom(
            label, (self.output_size2[0] / x, self.output_size2[1] / y), order=0)
        super_label2 = zoom(
            super_label, (self.output_size2[0] / x, self.output_size2[1] / y), order=0)
        image2 = torch.from_numpy(
            image2.astype(np.float32)).unsqueeze(0)
        label2 = torch.from_numpy(label2.astype(np.uint8))
        super_label2 = torch.from_numpy(super_label2.astype(np.uint8))
        
        image3 = zoom(
            image, (self.output_size3[0] / x, self.output_size3[1] / y), order=0)
        label3 = zoom(
            label, (self.output_size3[0] / x, self.output_size3[1] / y), order=0)
        super_label3 = zoom(
            super_label, (self.output_size3[0] / x, self.output_size3[1] / y), order=0)
        image3 = torch.from_numpy(
            image3.astype(np.float32)).unsqueeze(0)
        label3 = torch.from_numpy(label3.astype(np.uint8))
        super_label3 = torch.from_numpy(super_label3.astype(np.uint8))
        

        
        sample = {'image': image1, 'label': label1,'super_label': super_label1,'image2': image2, 'label2': label2,'super_label2': super_label2, 'image3': image3, 'label3': label3,'super_label3': super_label3 }
        return sample

class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size
      

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
    
        
        sample = {'image': image, 'label': label,'super_label': super_label }
        return sample


class RandomGenerator3(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image,image2, label,super_label = sample['image'], sample['image2'], sample['label'], sample['super_label']
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]
        if random.random() > 0.5:
            image,image2, label, super_label = random_rot_flip2(image,image2,label,super_label)
        elif random.random() > 0.5:
            if 4 in np.unique(label):
                image,image2, label,super_label = random_rotate2(image,image2, label,super_label, cval=4)
            else:
                image, label,super_label = random_rotate2(image,image2, label,super_label, cval=0)
        x, y = image.shape
        image = zoom(
            image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image2 = zoom(
            image2, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(
            label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        super_label = zoom(
            super_label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(
            image.astype(np.float32)).unsqueeze(0)
        image2 = torch.from_numpy(
            image2.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        super_label = torch.from_numpy(super_label.astype(np.uint8))

        sample = {'image': image,'image2': image2, 'label': label,'super_label': super_label }
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