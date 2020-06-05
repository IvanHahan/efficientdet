from efficientdet import EfficientDet
from data.nandos_dataset import NandosDataset
import torch
from torchvision import transforms
from torch import nn
from data.letter_dataset import LetterDataset
from torch.utils.data import DataLoader
import argparse
import numpy as np
from efficientdet.losses import total_loss, focal_loss
from efficientdet.utils import build_label, postprocess
from torch.optim import Adam
from matplotlib import pyplot as plt
import cv2
from ranger import ranger

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--dataset', choices=['nandos'], default='nandos')
parser.add_argument('--image_dir', default='/Users/UnicornKing/sciforce/ActionRecognitionAnnotator/data/detection_images')
parser.add_argument('--label_path', default='/Users/UnicornKing/sciforce/ActionRecognitionAnnotator/data/detection_label.txt')
args = parser.parse_args()

if __name__ == '__main__':

    calc_loss = total_loss()

    if args.dataset == 'nandos':
        train_dataset = NandosDataset(args.image_dir, args.label_path,
                                      transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])
                                      ])
                                      )

    model = EfficientDet(train_dataset.num_classes())

    optimizer = ranger(model.parameters(), 1e-3)

    for e in range(args.epochs):
        for images, rects, classes in DataLoader(train_dataset, 4, False):
            optimizer.zero_grad()

            classes_, activations, train_rects, output_rects = model(images.float())

            loss = calc_loss(train_rects, classes_, rects, classes)
            print('loss:', loss.item())

            loss.backward()
            # torch.nn.utils.clip_grad_value_(model.parameters(), 0.1)
            optimizer.step()

        if e % 2 == 0:
            image = (images[0].data.numpy().transpose([1, 2, 0]) * 127.5) + 127.5
            image = image.copy().astype('uint8')
            out_classes, out_rects = postprocess(classes_[0], output_rects[0])
            for rect in out_rects.data.numpy():
                x1, y1, x2, y2 = rect.astype(int).tolist()
                image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)
            plt.imshow(image)
            plt.show()


        # if e > 40:
        #     optimizer = Adam(model.parameters(), 0.001)


