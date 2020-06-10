import imgaug.augmenters as iaa
import numpy as np
import torch
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

from .image_processing import resize_image, pad_image


def boxes_numpy2imgaug(annotations, image_shape):
    boxes = []
    for annot in annotations:
        x1, y1, x2, y2, label = annot
        box = BoundingBox(x1, y1, x2, y2, label)
        boxes.append(box)
    return BoundingBoxesOnImage(boxes, image_shape)


def boxes_imgaug2numpy(boxes):
    annotations = []
    for box in boxes.bounding_boxes:
        annotations.append((box.x1, box.y1, box.x2, box.y2, box.label))
    return np.array(annotations)


def get_augmentations():
    def sometimes(aug, p=0.5): return iaa.Sometimes(p, aug)

    return iaa.Sequential(
        [
            iaa.SomeOf(2, [
                sometimes(iaa.Multiply()),
                sometimes(iaa.HorizontalFlip()),
                sometimes(iaa.GammaContrast()),
                sometimes(iaa.AddToHueAndSaturation(5)),
                sometimes(iaa.CLAHE())
            ]
                       )
        ]
    )


class Augmenter(object):
    def __init__(self, augmentations=get_augmentations()):
        self.augmentations = augmentations

    def __call__(self, sample):
        """
        :param image: numpy image
        :param annotations: (x1, x2, y1, y2, label) unscaled
        """
        image = sample['image']
        if 'annotations' in sample:
            annotations = sample['annotations']
            boxes = boxes_numpy2imgaug(annotations, image.shape[:2])
            image_aug, boxes_aug = self.augmentations(image=image, bounding_boxes=boxes)
            sample['image'] = image_aug
            sample['annotations'] = boxes_imgaug2numpy(boxes_aug)
        else:
            image_aug = self.augmentations(image=image)
            sample['image'] = image_aug

        return sample


class MaxSizeResizer(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        """
        :param image: numpy image
        :param annotations: (x1, x2, y1, y2, label) unscaled
        """
        image = sample['image']
        if 'annotations' in sample:
            annotations = sample['annotations']
            d = max(image.shape) / self.size
            annotations[:, :4] /= d
            annotations = annotations.astype(int)
            sample['annotations'] = annotations

        image = resize_image(image, self.size)
        sample['image'] = image

        return sample


class SquarePad(object):

    def __call__(self, sample):
        """
        :param image: numpy image
        :param annotations: (x1, x2, y1, y2, label) unscaled
        """
        image = sample['image']

        image = pad_image(image, (max(image.shape), max(image.shape)))[0]
        sample['image'] = image

        return sample


class ToTensor(object):
    def __call__(self, sample):
        sample['image'] = torch.from_numpy(sample['image'].transpose([2, 0, 1]))
        if 'annotations' in sample:
            sample['annotations'] = torch.from_numpy(sample['annotations'])
        return sample
