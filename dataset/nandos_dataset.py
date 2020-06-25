from torch.utils.data import Dataset
import os
import cv2
import numpy as np
import torch
import dataset.image_processing as im_proc
from efficientdet.utils import build_label
import torchvision
from .augmentation import ToTensor

IMAGE_SIZE = (1280, 1280)

def resize_image(image, size=IMAGE_SIZE):
    image = im_proc.resize_image(image, max(size))

    image = im_proc.pad_image(image, size)[0]
    return image


class NandosDataset(Dataset):

    def __init__(self, image_dir, label_path, device='cpu', transform=ToTensor()):
        super().__init__()
        self.image_dir = image_dir
        self.label_path = label_path
        with open(label_path, 'r') as f:
            lines = f.readlines()
            records = {}
            for l in lines:
                frame_path, x1, y1, x2, y2, label = l.split(' ')
                records.setdefault(frame_path, [])
                records[frame_path].append((int(x1), int(y1), int(x2), int(y2), int(label)))
            self.records = records
        self.frames = list(self.records.keys())
        self.transform = transform
        self.device = device

    def __getitem__(self, index):
        frame_path = self.frames[index]
        annot = np.array(self.records[frame_path]).astype(float)
        image = cv2.imread(os.path.join(self.image_dir, frame_path))
        sample = self.transform({'image': image, 'annotations': annot})
        image = sample['image']
        annot = sample['annotations']
        rects, classes = build_label(annot, image.shape[1:], [0.5, 1, 2], [0.5, 1], self.num_classes())
        image = image.to(self.device)
        rects = rects.to(self.device)
        classes = classes.to(self.device)

        return image, rects, classes

    def __len__(self):
        return len(self.frames)

    def num_classes(self):
        return 1


if __name__ == '__main__':
    ds = NandosDataset("/Users/UnicornKing/sciforce/ActionRecognitionAnnotator/data/detection_images",
                       "/Users/UnicornKing/sciforce/ActionRecognitionAnnotator/data/detection_label.txt")
    print(ds[0][0].shape)

