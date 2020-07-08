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
import torchvision
import torchvision.transforms as transforms
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


class GreyscaleDataset(Dataset):
    def __init__(self, data_folder: str = 'data', dataset_file: str = 'dataset.pkl'):
        """Grayscale image dataset as provided by the python II lecture"""
        # check for existing dataset
        if os.path.exists(dataset_file):
            with open(dataset_file, 'rb') as f:
                data = pickle.load(f)
            print(f'loaded dataset from {dataset_file}')
        else:
            # create the data
            files = sorted(glob.glob(os.path.join(data_folder, '**/*.jpg'), recursive=True))
            images = []
            crop_sizes = []
            crop_centers = []
            for image in tqdm(files, desc='creating dataset', total=len(files)):
                img = Image.open(image)
                x, y, w, h, cx, cy = get_random_image_values()
                img = img.resize((y, x), Image.LANCZOS)  # mind thee: x and y swapped
                arr = np.array(img, dtype=np.float32)
                images.append(arr)
                crop_sizes.append((w, h))
                crop_centers.append((cx, cy))
            data = {'images': images, 'crop_sizes': crop_sizes, 'crop_centers': crop_centers}
            # safe for next iteration
            with open(dataset_file, 'wb') as f:
                pickle.dump(data, f)
            print(f'created datset and saved it to {dataset_file}')

        self.images = data['images']
        self.crop_sizes = data['crop_sizes']
        self.crop_centers = data['crop_centers']

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.crop_sizes[idx], self.crop_centers[idx], idx


def create_cropped_data(image_array: np.ndarray, crop_size: tuple, crop_center: tuple, debug=False) -> tuple:
    # check parameters
    if debug:
        if not isinstance(image_array, np.ndarray) or len(image_array.shape) != 2:
            raise ValueError('image_array is not a 2D numpy array')
        elif len(crop_size) != 2 or len(crop_center) != 2:
            raise ValueError('crop size or crop center tuples have invalid amount of values')
        elif crop_size[0] % 2 == 0 or crop_size[1] % 2 == 0:
            raise ValueError('crop size contains an even number')
    # check rectangle position
    crop_margin = 20
    min_x = crop_center[0] - crop_size[0] // 2
    max_x = crop_center[0] + crop_size[0] // 2
    min_y = crop_center[1] - crop_size[1] // 2
    max_y = crop_center[1] + crop_size[1] // 2
    if not (crop_margin <= min_x and max_x < image_array.shape[0] - crop_margin and
            crop_margin <= min_y and max_y < image_array.shape[1] - crop_margin):
        raise ValueError('the crop rectangle is too close to the edges')
    # create crop array
    crop_array = np.zeros_like(image_array)
    crop_array[min_x:max_x + 1, min_y:max_y + 1] = 1
    # target_array = crop region in image_array
    target_array = np.copy(image_array[min_x:max_x + 1, min_y:max_y + 1])
    # set image_array values in crop region to 0 (in-place)
    image_array[min_x:max_x + 1, min_y:max_y + 1] = 0
    return image_array, crop_array, target_array


class CroppedImages(Dataset):
    def __init__(self, dataset: Dataset, transform_chain: transforms.Compose = None, debug=False):
        """Provides images from 'dataset' as inputs and images cropped as targets"""
        self.dataset = dataset
        self.transform_chain = transform_chain
        self.debug = debug

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image_data, crop_size, crop_center, idx = self.dataset.__getitem__(idx)
        image_data = TF.to_pil_image(image_data)
        if self.transform_chain is not None:
            image_data = self.transform_chain(image_data)
        image_data = np.copy(np.asarray(image_data, dtype=np.float32))
        image_data, crop_array, target_array = create_cropped_data(
            np.copy(image_data), crop_size, crop_center, debug=self.debug)
        # Perform normalization based on input values of individual sample
        mean = image_data.mean()
        std = image_data.std()
        image_data[:] -= mean
        image_data[:] /= std
        target_array[:] -= mean
        target_array[:] /= std
        # Add information about relative position in image to inputs
        # full_inputs = image_data  # Not feeding information about the position in the image would be bad for our CNN
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
        full_inputs[..., 1] = closeness_array
        full_inputs[..., 2] = crop_array

        # Convert numpy arrays to tensors
        full_inputs = TF.to_tensor(full_inputs)
        # target_data = TF.to_tensor(target_array)

        return full_inputs, target_array, idx
