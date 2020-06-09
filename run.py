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

parser = argparse.ArgumentParser()
parser.add_argument('--input')
parser.add_argument('--output', default='output.jpg')
parser.add_argument('--weights')
parser.add_argument('--num_classes', default=3)
args = parser.parse_args()

if __name__ == '__main__':

    model = EfficientDet(args.num_classes, network=args.network)
    model.load_state_dict(torch.load(args.weights))
    model.eval()

    image = cv2.imread(args.image)

    image = resize_image(image)
    input_ = torch.from_numpy(np.expand_dims(image.transpose([2, 0, 1]), 0))
    classes_, _, train_rects, output_rects = model(input_.float())
    out_classes, out_rects = postprocess(classes_[0], output_rects[0])
    for rect in out_rects.data.numpy():
        x1, y1, x2, y2 = rect.astype(int).tolist()
        image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imwrite(args.output, image)
