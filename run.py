import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset.nandos_dataset import NandosDataset
from efficientdet import EfficientDet
from efficientdet.losses import total_loss
from ranger import ranger
from util import make_dir_if_needed
import cv2
from dataset.nandos_dataset import resize_image
from efficientdet.utils import postprocess
import matplotlib.pyplot as plt
from torchvision import transforms
from dataset.augmentation import MaxSizeResizer, ToTensor, SquarePad, Augmenter

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='data/screenshot/train/images/0_mainframe.png')
parser.add_argument('--output', default='output.jpg')
parser.add_argument('--weights', default='model/efficientdet-d2-e50.pth')
parser.add_argument('--network', default='efficientdet-d2')
parser.add_argument('--num_classes', default=1)
args = parser.parse_args()

if __name__ == '__main__':

    model = EfficientDet(args.num_classes, network=args.network)
    model.load_state_dict(torch.load(args.weights))
    model.eval()

    image = cv2.imread(args.input)
    transforms = transforms.Compose(
        [
            MaxSizeResizer(1024),
            SquarePad(),
            ToTensor(),
        ]
    )
    image = transforms({'image': image})['image'].float()
    image = image.view(1, *image.size())
    classes_, _, train_rects, output_rects = model(image.float())
    out_classes, out_rects = postprocess(classes_[0], output_rects[0], 0.3)
    image = image[0].int().cpu().numpy().transpose([1, 2, 0]).copy().astype('uint8')
    for rect in out_rects.data.numpy():
        x1, y1, x2, y2 = rect.astype(int).tolist()
        image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 1)
    cv2.imwrite(args.output, image)
    plt.imshow(image)
    plt.show()
