from efficientdet import EfficientDet
from dataset.nandos_dataset import NandosDataset
import torch
from torchvision import transforms
from torch import nn
from dataset.letter_dataset import LetterDataset
from torch.utils.data import DataLoader
import argparse
import numpy as np
from efficientdet.losses import total_loss, focal_loss
from efficientdet.utils import build_label, postprocess
from torch.optim import Adam
from matplotlib import pyplot as plt
import cv2
import os
from ranger import ranger
from util import make_dir_if_needed

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--dataset', choices=['nandos'], default='nandos')
parser.add_argument('--image_dir', default='/Users/UnicornKing/sciforce/ActionRecognitionAnnotator/data/detection_images')
parser.add_argument('--label_path', default='/Users/UnicornKing/sciforce/ActionRecognitionAnnotator/data/detection_label.txt')
parser.add_argument('--model_dir', default='model/')
parser.add_argument('--verbosity', default=10, type=int)
parser.add_argument('--network', default='efficientdet-d0')
args = parser.parse_args()

if __name__ == '__main__':

    make_dir_if_needed(args.model_dir)

    calc_loss = total_loss()

    if args.dataset == 'nandos':
        train_dataset = NandosDataset(args.image_dir, args.label_path,
                                      transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])
                                      ])
                                      )

    model = EfficientDet(train_dataset.num_classes(), network=args.network)

    optimizer = ranger(model.parameters(), 1e-3)

    for e in range(args.epochs):
        losses = []
        for images, rects, classes in DataLoader(train_dataset, 4, False):
            optimizer.zero_grad()

            classes_, activations, train_rects, output_rects = model(images.float())

            loss = calc_loss(train_rects, classes_, rects, classes)
            losses.append(loss.item())

            loss.backward()
            # torch.nn.utils.clip_grad_value_(model.parameters(), 0.1)
            optimizer.step()
        print(f'EPOCH {e}: Loss - {np.mean(losses)}')

        if e % args.verbosity == 0:
            torch.save(model.state_dict(), os.path.join(args.model_dir, f'{args.network}-e{e}.pth'))



