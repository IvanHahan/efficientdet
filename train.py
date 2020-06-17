import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset.nandos_dataset import NandosDataset
from dataset.letter_dataset import LetterDataset
from efficientdet import EfficientDet
from efficientdet.losses import total_loss
from ranger import ranger
from util import make_dir_if_needed
from efficientdet.utils import postprocess
import cv2
import matplotlib.pyplot as plt
from dataset.augmentation import Augmenter, MaxSizeResizer, ToTensor, SquarePad
from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=501)
parser.add_argument('--dataset', choices=['nandos', 'letters'], default='nandos')
parser.add_argument('--image_dir',
                    default='data/nandos/detection_images')
parser.add_argument('--label_path',
                    default='data/nandos/detection_label.txt')
parser.add_argument('--model_dir', default='model/')
parser.add_argument('--verbosity', default=50, type=int)
parser.add_argument('--network', default='efficientdet-d0')
parser.add_argument('--device', default='cuda')
parser.add_argument('--checkpoint', default=None)
args = parser.parse_args()

if __name__ == '__main__':

    device = args.device

    make_dir_if_needed(args.model_dir)

    calc_loss = total_loss()

    if args.dataset == 'nandos':
        train_dataset = NandosDataset(args.image_dir, args.label_path, device=device, transform=transforms.Compose([
            Augmenter(),
            MaxSizeResizer(1024),
            SquarePad(),
            ToTensor(),
        ]))
    elif args.dataset == 'letters':
        train_dataset = LetterDataset(args.image_dir, args.label_path, device=device, transform=transforms.Compose([
            MaxSizeResizer(1024),
            SquarePad(),
            ToTensor(),
        ]))

    model = EfficientDet(train_dataset.num_classes(), network=args.network, device=device).to(device)
    model.eval()

    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint))

    model.train()
    optimizer = ranger(model.parameters(), 1e-3)

    for e in range(args.epochs):
        losses = []
        for images, rects, classes in DataLoader(train_dataset, 2, False):
            optimizer.zero_grad()

            classes_, activations, train_rects, output_rects = model(images.float())

            loss = calc_loss(train_rects, classes_, rects, classes)
            losses.append(loss.item())

            loss.backward()
            optimizer.step()

        print(f'EPOCH {e}: Loss - {np.mean(losses)}')

        if e % args.verbosity == 0:
            torch.save(model.state_dict(), os.path.join(args.model_dir, f'{args.dataset}-{args.network}-e{e}.pth'))
            out_classes, out_rects = postprocess(classes_[0], output_rects[0])
            image = images[0].int().cpu().numpy().transpose([1,2,0]).copy().astype('uint8')
            for rect in out_rects.cpu().data.numpy():
                x1, y1, x2, y2 = rect.astype(int).tolist()
                image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            plt.imshow(image)
            plt.show()
