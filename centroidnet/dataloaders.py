import os
from typing import Any, Tuple, Optional

from torch.utils.data import Dataset
import numpy as np
import cv2
import random

from tqdm import tqdm

from .centroidnet_core import encode


class CentroidNetDataset(Dataset):
    """
    CentroidNetDataset Dataset
        Load centroids from txt file and apply vector aware data augmentation

    Arguments:
        filename: filename of the input data format: (image_file_name,xmin,ymin,xmax,ymax,class_id \\lf)
        crop (h, w): Random crop size
        transpose ((dim2, dim3)): List of random transposes to choose from
        stride ((dim2, dim3)): List of random strides to choose from
    """
    def convert_path(self, path):
        if os.path.isabs(path):
            return path
        else:
            return os.path.join(self.data_path, path)

    def load_and_convert_data(self, filename, max_dist):
        with open(filename) as f:
            lines = f.readlines()
        lines = [x.strip().split(",") for x in lines]
        self.count = len(lines)
        img_boxes = {}
        for line in lines:
            fn, xmin, xmax, ymin, ymax, class_id = line
            xmin, xmax, ymin, ymax, class_id = int(xmin), int(xmax), int(ymin), int(ymax), int(class_id)
            x, y = (xmin + xmax) // 2, (ymin + ymax) // 2
            if fn not in img_boxes:
                img_boxes[fn] = list()

            self.num_classes = class_id+2 if class_id+2 > self.num_classes else self.num_classes
            img_boxes[fn].append(np.array([y, x, ymin, ymax, xmin, xmax, class_id], dtype=int))

        for key in tqdm(img_boxes.keys()):
            img_boxes[key] = np.stack(img_boxes[key])
            fn = self.convert_path(key)
            img = cv2.imread(fn)

            if img is None:
                raise Exception("Could not read {}".format(fn))

            if self.crop is not None:
                crop = min(img.shape[0], img.shape[1], self.crop[0], self.crop[1])
                if crop != self.crop[0]:
                    print(f"Warning: random crop adjusted to {[crop, crop]}")
                    self.set_crop([crop, crop])

            target = encode(img_boxes[key], img.shape[0], img.shape[1], max_dist, self.num_classes)
            img = (np.transpose(img, [2, 0, 1]).astype(np.float32) - self.sub) / self.div
            target = target.astype(np.float32)

            self.images.append(img)
            self.targets.append(target)

    def __init__(self, filename: str, crop=(256, 256), max_dist=100, repeat=1, sub=127, div=256,
                 transpose=np.array([[0, 1], [1, 0]]),
                 stride=np.array([[1, 1], [-1, -1], [-1, 1], [1, -1]]), data_path=None):
        self.count = 0
        self.crop = None
        self.transpose: Optional[Tuple[Any, Any]] = None
        self.stride: Optional[Tuple[Any, Any]] = None
        self.train_mode = None

        if data_path is None:
            self.data_path = os.path.dirname(filename)
        else:
            self.data_path = data_path

        self.filename = filename
        self.images = list()
        self.targets = list()

        self.sub = sub
        self.div = div
        self.repeat = repeat
        self.set_crop(crop)
        self.set_repeat(repeat)
        self.set_transpose(transpose)
        self.set_stride(stride)
        self.num_classes = 0
        self.train()
        self.load_and_convert_data(filename, max_dist=max_dist)

    def eval(self):
        self.train_mode = False

    def train(self):
        self.train_mode = True

    def set_repeat(self, repeat):
        if repeat < 0:
            self.repeat = 1
        self.repeat = repeat

    def set_crop(self, crop):
        self.crop = crop

    def set_transpose(self, transpose):
        if np.all(transpose == np.array([[0, 1]])):
            self.transpose = transpose

    def set_stride(self, stride):
        if np.all(stride == np.array([[1, 1]])):
            self.stride = stride

    @staticmethod
    def adjust_vectors(img, transpose, stride):
        if transpose is not None:
            img2 = img.copy()
            img[0] = img2[transpose[0]]
            img[1] = img2[transpose[1]]
        if stride is not None:
            img2 = img.copy()
            img[0] = img2[0] * stride[0]
            img[1] = img2[1] * stride[1]
        return img

    @staticmethod
    def adjust_image(img, transpose, slice, crop, stride):
        if transpose is not None:
            img = np.transpose(img, (0, transpose[0] + 1, transpose[1] + 1))
        if slice is not None:
            img = img[:, slice[0]:slice[0] + crop[0], slice[1]:slice[1] + crop[1]]
        if stride is not None:
            img = img[:, ::stride[0], ::stride[1]]
        return img

    def get_target(self, img: np.array, transpose, slice, crop, stride):
        img[0:2] = self.adjust_vectors(img[0:2], transpose, stride)
        img[2:4] = self.adjust_vectors(img[2:4], transpose, stride)
        img = self.adjust_image(img, transpose, slice, crop, stride)
        return img

    def get_input(self, img: np.array, transpose, slice, crop, stride):
        img = self.adjust_image(img, transpose, slice, crop, stride)
        return img

    def __getitem__(self, index):
        index = index // self.repeat
        input, target = self.images[index], self.targets[index]

        if self.stride is None and self.transpose is None and self.crop is None or not self.train_mode:
            return input.astype(np.float32), target.astype(np.float32)

        if self.transpose is not None:
            transpose = random.choice(self.transpose)
        else:
            transpose = None

        if self.stride is not None:
            stride = random.choice(self.stride)
        else:
            stride = None

        if self.crop is not None:
            min = np.array([0, 0])
            if transpose is not None:
                max = np.array([input.shape[transpose[0] + 1], input.shape[transpose[1] + 1]], dtype=int) - self.crop
            else:
                max = np.array([input.shape[1] - self.crop[0], input.shape[2] - self.crop[1]])
            slice = [random.randint(mn, mx) for mn, mx in zip(min, max)]
        else:
            slice = None

        input = self.get_input(input, transpose, slice, self.crop, stride).astype(np.float32)
        target = self.get_target(target, transpose, slice, self.crop, stride).astype(np.float32)

        return input, target

    def __len__(self):
        return len(self.images) * self.repeat
