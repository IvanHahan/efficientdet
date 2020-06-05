from torch.utils.data import Dataset
import os
import cv2
import numpy as np
import torch

def build_label(image_shape, boxes):
    lev_8 = np.zeros((5, *np.array(image_shape) // 8))
    lev_16 = np.zeros((5, *np.array(image_shape) // 16))
    lev_32 = np.zeros((5, *np.array(image_shape) // 32))
    lev_64 = np.zeros((5, *np.array(image_shape) // 64))
    lev_128 = np.zeros((5, *np.array(image_shape) // 128))
    levels = [lev_8, lev_16, lev_32, lev_64, lev_128]

    anchor_boxes = np.array([
        [[8, 8]],
        [[16, 16]],
        [[32, 32]],
        [[64, 64]],
        [[128, 128]],
    ])
    anchor_boxes_areas = np.multiply.reduce(anchor_boxes, axis=-1) # [5, ?]

    for b in boxes:
        x, y, w, h = b
        area = w * h
        if area == 0:
            continue
        c_x, c_y = x + w / 2, y + h / 2
        ratios = anchor_boxes_areas / area
        ratios[ratios > 1] = 1 / ratios[ratios > 1]
        max_ratios = np.max(ratios, axis=1)
        best_level = int(np.argmax(max_ratios))
        cell_h, cell_w = np.array(image_shape) // np.array(levels[best_level].shape[1:])

        lev_x, lev_y = int(c_x // cell_w), int(c_y // cell_h)

        cell_rel_x = x - lev_x * cell_w + (w/2)
        cell_rel_y = y - lev_y * cell_h + (h/2)

        w = np.log(w / cell_w)
        h = np.log(h / cell_h)

        levels[best_level][0, lev_y, lev_x] = cell_rel_x
        levels[best_level][1, lev_y, lev_x] = cell_rel_y
        levels[best_level][2, lev_y, lev_x] = w
        levels[best_level][3, lev_y, lev_x] = h
        levels[best_level][4, lev_y, lev_x] = 1

    return np.column_stack([level.reshape(5, -1) for level in levels]).transpose()


class LetterDataset(Dataset):

    def __init__(self, image_dir, annot_dir):
        self.image_dir = image_dir
        self.annot_dir = annot_dir
        self.annot_names = [os.path.splitext(name)[0] for name in os.listdir(annot_dir)]

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.annot_names[index] + '.png')
        image = cv2.imread(image_path)
        h, w = image.shape[:2]
        image = cv2.resize(image, (1280, 768))
        dh, dw = h / image.shape[0], w / image.shape[1]
        image = np.transpose(image, [2, 0, 1])

        annot_path = os.path.join(self.annot_dir, self.annot_names[index] + '.npy')
        letter_boxes = np.load(annot_path).astype('float32')
        letter_boxes[:, 0] /= dw
        letter_boxes[:, 1] /= dh
        letter_boxes[:, 2] /= dw
        letter_boxes[:, 3] /= dw

        return image, letter_boxes

    def __len__(self):
        return len(self.annot_names)


if __name__ == '__main__':
    ds = LetterDataset("/Users/UnicornKing/PyCharmProjects/efficientdet/data/images/raw",
                       "/Users/UnicornKing/PyCharmProjects/efficientdet/data/annotations")
    print(ds[0][1].shape)

