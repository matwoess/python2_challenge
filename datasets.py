# -*- coding: utf-8 -*-
"""datasets.py

Author -- Mathias Wöß
Contact -- k11709064@students.jku.at
"""
import glob
import os
import random

import numpy as np
from PIL import Image
import dill as pickle
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from tqdm import tqdm


def get_random_image_values():
    border = 20
    rand_x = random.randrange(70, 100)  # maximum pixel index = 99
    rand_y = random.randrange(70, 100)  # maximum pixel index = 99
    rand_w = random.randrange(5, 21 + 1, 2)
    rand_h = random.randrange(5, 21 + 1, 2)
    rand_cx = random.randrange(border + (rand_w // 2), rand_x - border - (rand_w // 2), 2)
    rand_cy = random.randrange(border + (rand_h // 2), rand_y - border - (rand_h // 2), 2)
    return rand_x, rand_y, rand_w, rand_h, rand_cx, rand_cy


def create_cropped_data(image_array: np.ndarray, crop_size: tuple, crop_center: tuple, crop_only: bool = True):
    if not crop_only:
        # check parameters
        if not isinstance(image_array, np.ndarray) or len(image_array.shape) != 2:
            raise ValueError('image_array is not a 2D numpy array')
        elif len(crop_size) != 2 or len(crop_center) != 2:
            raise ValueError('crop size or crop center tuples have invalid amount of values')
        elif crop_size[0] % 2 == 0 or crop_size[1] % 2 == 0:
            raise ValueError('crop size contains an even number')
    # check rectangle position
    min_x = crop_center[0] - crop_size[0] // 2
    max_x = crop_center[0] + crop_size[0] // 2
    min_y = crop_center[1] - crop_size[1] // 2
    max_y = crop_center[1] + crop_size[1] // 2
    if not crop_only:
        crop_margin = 20
        if not (crop_margin <= min_x and max_x < image_array.shape[0] - crop_margin and
                crop_margin <= min_y and max_y < image_array.shape[1] - crop_margin):
            raise ValueError('the crop rectangle is too close to the edges')
    if crop_only:
        # create crop array
        crop_array = np.zeros_like(image_array)
        crop_array[min_x:max_x + 1, min_y:max_y + 1] = 1
        return crop_array
    else:
        # target_array = crop region in image_array
        target_array = np.copy(image_array[min_x:max_x + 1, min_y:max_y + 1])
        # set image_array values in crop region to 0 (in-place)
        image_array[min_x:max_x + 1, min_y:max_y + 1] = 0
        return image_array, target_array


def create_dataset(data_folder: str, dataset_file: str, targets_file: str = os.path.join('data', 'targets.pkl')):
    files = sorted(glob.glob(os.path.join(data_folder, '**/*.jpg'), recursive=True))
    images = []
    crop_sizes = []
    crop_centers = []
    targets = []
    for image in tqdm(files, desc='creating dataset', total=len(files)):
        img = Image.open(image)
        # quadruple dataset by vertical and horizontal flipping
        for i in range(4):
            if i == 1 or i == 3:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if i == 2:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
            x, y, w, h, cx, cy = get_random_image_values()
            resized = img.resize((y, x), Image.LANCZOS)  # mind thee: x and y swapped
            arr = np.array(resized, dtype=np.float32)
            arr, target_array = create_cropped_data(np.copy(arr), (w, h), (cx, cy), crop_only=False)
            images.append(arr)
            crop_sizes.append((w, h))
            crop_centers.append((cx, cy))
            targets.append(target_array)
    data = {'images': images, 'crop_sizes': crop_sizes, 'crop_centers': crop_centers}
    # safe for next iteration
    with open(dataset_file, 'wb') as f:
        pickle.dump(data, f)
    with open(targets_file, 'wb') as f:
        pickle.dump(targets, f)
    print(f'created datset and saved it to {dataset_file} and targets to {targets_file}')


class CropDataset(Dataset):
    def __init__(self, data_folder: str = os.path.join('data', 'user_images'),
                 dataset_file: str = os.path.join('data', 'dataset.pkl'),
                 targets: str = os.path.join('data', 'dataset.pkl')):
        """Grayscale image dataset as provided by the python II lecture"""
        # check for existing dataset
        if not os.path.exists(dataset_file):
            create_dataset(data_folder, dataset_file)
        with open(dataset_file, 'rb') as f:
            data = pickle.load(f)
        print(f'loaded dataset from {dataset_file}')
        self.images = data['images']
        self.crop_sizes = data['crop_sizes']
        self.crop_centers = data['crop_centers']

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.crop_sizes[idx], self.crop_centers[idx], idx


class AugmentedDataset(Dataset):
    def __init__(self, dataset: Dataset):
        """Provides images from 'dataset' as inputs and images cropped as targets"""
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image_data, crop_size, crop_center, idx = self.dataset.__getitem__(idx)
        crop_array = create_cropped_data(image_data, crop_size, crop_center, crop_only=True)
        image_data = image_data.astype(np.float32)
        mean = image_data.mean()
        std = image_data.std()
        image_data[:] -= mean
        image_data[:] /= std
        # Add information about relative position in image to inputs
        full_inputs = np.zeros(shape=(*image_data.shape, 3), dtype=image_data.dtype)
        full_inputs[..., 0] = image_data
        # create a layer showing approximately how far away from the crop_center each pixel is
        x_offset = (crop_center[0] - image_data.shape[0] / 2) / image_data.shape[0]
        y_offset = (crop_center[1] - image_data.shape[1] / 2) / image_data.shape[1]
        closeness_array = np.zeros(image_data.shape, dtype=np.float32)
        closeness_array.T[:, ] += 0.5 - np.abs(
            np.linspace(-0.5 - x_offset, 0.5 - x_offset, num=image_data.shape[0], endpoint=True))
        closeness_array[:, ] += 0.5 - np.abs(
            np.linspace(-0.5 - y_offset, 0.5 - y_offset, num=image_data.shape[1], endpoint=True))
        # scale layer to interval  [0, 1]
        closeness_array -= np.min(closeness_array)
        closeness_array /= np.max(closeness_array)
        # crop out blank array

        full_inputs[..., 1] = closeness_array
        full_inputs[..., 2] = crop_array

        # Convert numpy arrays to tensors
        full_inputs = TF.to_tensor(full_inputs)
        # target_data = TF.to_tensor(target_array)

        return full_inputs, crop_size, mean, std, idx


class TrainingDataset(Dataset):
    def __init__(self, dataset: Dataset, targets_file: str = os.path.join('data', 'targets.pkl')):
        """Provides images from 'dataset' as inputs and images cropped as targets"""
        self.dataset = dataset
        with open(targets_file, 'rb') as f:
            target_data = pickle.load(f)
        self.targets = target_data

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        full_inputs, crop_size, mean, std, idx = self.dataset.__getitem__(idx)
        target = self.targets[idx]
        target = target.astype(np.float32)
        target[:] -= mean
        target[:] /= std
        return full_inputs, crop_size, mean, std, target, idx
