import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

from efficientdet.bifpn import BIFPN
from efficientdet.my_retinahead import RetinaHead
from efficientnet.model import EfficientNet

MODEL_MAP = {
    'efficientdet-d0': 'efficientnet-b0',
    'efficientdet-d1': 'efficientnet-b1',
    'efficientdet-d2': 'efficientnet-b2',
    'efficientdet-d3': 'efficientnet-b3',
    'efficientdet-d4': 'efficientnet-b4',
    'efficientdet-d5': 'efficientnet-b5',
    'efficientdet-d6': 'efficientnet-b6',
    'efficientdet-d7': 'efficientnet-b6',
}


class EfficientDet(nn.Module):
    def __init__(self,
                 num_classes,
                 network='efficientdet-d0',
                 D_bifpn=3,
                 W_bifpn=88,
                 is_training=True,
                 threshold=0.01,
                 iou_threshold=0.5,
                 transform=transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])):
        super(EfficientDet, self).__init__()
        self.backbone = EfficientNet.from_pretrained(MODEL_MAP[network])
        self.is_training = is_training
        self.neck = BIFPN(in_channels=self.backbone.get_list_features()[-5:],
                          out_channels=W_bifpn,
                          stack=D_bifpn,
                          num_outs=5)
        self.bbox_head = RetinaHead(num_classes=num_classes,
                                    in_channels=W_bifpn)

        self.threshold = threshold
        self.iou_threshold = iou_threshold
        self.transform = transform

    def forward(self, inputs):
        xs = []
        for x_ in inputs:
            x_ = self.transform(x_)
            xs.append(x_)
        xs = torch.stack(xs)
        x = self.extract_feat(xs)
        # classes, train_boxes, output_boxes = self.bbox_head(x, inputs.shape[2:])
        classes, activations, train_boxes, output_boxes = self.bbox_head(x, inputs.shape[2:])
        return classes, activations, train_boxes, output_boxes

    def extract_feat(self, img):
        """
            Directly extract features from the backbone+neck
        """
        x = self.backbone(img)
        x = self.neck(x[-5:])
        return x


if __name__ == '__main__':
    # print(EfficientNet.from_pretrained('efficientnet-b0')(torch.from_numpy(np.ones((1, 3, 128, 128))).float())[2].shape)
    # model = nn.Sequential(*EfficientNet.from_pretrained('efficientnet-b0').get_list_features())
    # print(model(torch.from_numpy(np.ones((1, 3, 128, 128))).float()))
    model = EfficientDet(2)
    a = model(torch.from_numpy(np.ones((1, 3, 1280, 1024))).float())
    print(a[0].shape)
