import torch
from torch import nn
import numpy as np


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class RetinaHead(nn.Module):

    def __init__(self, in_channels, num_classes, anchor_ratios=[0.5, 1, 2], anchor_scales=[0.5, 1], device='cpu'):
        super().__init__()
        self.anchor_ratios = anchor_ratios
        self.anchor_scales = anchor_scales
        self.num_classes = num_classes
        self.class_branch = nn.Sequential(
            DoubleConv(in_channels, 128),
            nn.Conv2d(128, num_classes * len(anchor_ratios) * len(anchor_scales), 3, 1, 1),
            # nn.BatchNorm2d(num_classes * len(anchor_ratios)),
            nn.Sigmoid()
        )
        self.boxes_branch = nn.Sequential(
            DoubleConv(in_channels, 128),
            nn.Conv2d(128, 4 * len(anchor_ratios) * len(anchor_scales), 3, 1, 1),
        )
        self.device = device

    def forward_single(self, x, img_shape):
        classes = self.class_branch(x)
        classes = classes.permute(0, 2, 3, 1)
        activations = classes.clone()
        classes = classes.contiguous().view(x.shape[0], -1, self.num_classes)

        boxes = self.boxes_branch(x)
        boxes = boxes.permute(0, 2, 3, 1)
        boxes = boxes.contiguous().view(*boxes.shape[:3], len(self.anchor_ratios) * len(self.anchor_scales), 4)  # B x H x W x A x (xywh)
        # print((boxes[..., :2]<0).sum())
        boxes[..., :2] = torch.sigmoid(boxes[..., :2])

        cell_shape = img_shape // np.array(list(boxes.shape[1:3]))
        anchors = []
        for ratio in self.anchor_ratios:
            for scale in self.anchor_scales:
                anchor = [max(cell_shape) * scale, max(cell_shape) * scale]
                if ratio < 1:
                    anchor[0] /= ratio
                else:
                    anchor[1] *= ratio
                anchors.append(anchor)

        output_boxes = boxes.clone()
        output_boxes[..., 0] = output_boxes[..., 0] * cell_shape[1] + \
                                  torch.arange(0, boxes.shape[2]).repeat(boxes.shape[1], 1) \
                                      .view(1, boxes.shape[1], boxes.shape[2], 1).to(self.device) * cell_shape[1]
        output_boxes[..., 1] = output_boxes[..., 1] * cell_shape[0] + \
                                  torch.arange(0, boxes.shape[1]).repeat(boxes.shape[2], 1).t() \
                                      .view(1, boxes.shape[1], boxes.shape[2], 1).to(self.device) * cell_shape[0]
        for i, anchor in enumerate(anchors):
            output_boxes[..., i, 2] = torch.exp(output_boxes[..., i, 2]) * anchor[1]
            output_boxes[..., i, 3] = torch.exp(output_boxes[..., i, 3]) * anchor[0]

        output_boxes = output_boxes.contiguous().view(output_boxes.size(0), -1, 4)
        train_boxes = boxes.contiguous().view(boxes.size(0), -1, 4)

        return classes, activations, train_boxes, output_boxes

    def forward(self, features, img_shape):
        # return [self.forward_single(x) for x in features][0]
        classes, activations, train_boxes, output_boxes = list(zip(*[self.forward_single(x, img_shape) for x in features]))
        classes = torch.cat(classes, 1)
        train_boxes = torch.cat(train_boxes, 1)
        output_boxes = torch.cat(output_boxes, 1)
        return classes, activations, train_boxes, output_boxes
