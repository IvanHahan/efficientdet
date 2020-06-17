from torch.utils.data import Dataset
import os
import cv2
import numpy as np
import torch
from .augmentation import ToTensor
from efficientdet.utils import build_label


class LetterDataset(Dataset):

    def __init__(self, image_dir, annot_dir, device='cpu', transform=ToTensor()):
        self.image_dir = image_dir
        self.annot_dir = annot_dir
        self.annot_names = [os.path.splitext(name)[0] for name in os.listdir(annot_dir)]
        self.transform = transform
        self.device = device

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.annot_names[index] + '.png')
        image = cv2.imread(image_path)

        annot_path = os.path.join(self.annot_dir, self.annot_names[index] + '.npy')
        letter_boxes = np.load(annot_path).astype('float32')
        letter_boxes[:, 2] = letter_boxes[:, 0] + letter_boxes[:, 2]
        letter_boxes[:, 3] = letter_boxes[:, 1] + letter_boxes[:, 3]
        letter_boxes = np.column_stack([letter_boxes, np.zeros((letter_boxes.shape[0], 1))])

        sample = self.transform({'image': image, 'annotations': letter_boxes})
        image = sample['image']
        annot = sample['annotations']

        keep_boxes = (annot[:, 0] < annot[:, 2]) & (annot[:, 1] < annot[:, 3])
        annot = annot[keep_boxes]

        rects, classes = build_label(annot, image.shape[1:], [0.5, 1, 2], self.num_classes())

        image = image.to(self.device)
        rects = rects.to(self.device)
        classes = classes.to(self.device)

        return image, rects, classes

    def __len__(self):
        return len(self.annot_names)

    def num_classes(self):
        return 1


if __name__ == '__main__':
    ds = LetterDataset("/dataset/images/raw",
                       "/dataset/annotations")
    print(ds[0][1].shape)

