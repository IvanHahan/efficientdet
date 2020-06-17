import torch
from torch import nn
from torch.nn import functional as F


def focal_loss(alpha, gamma):
    def loss(pred_classes, annot_classes):
        # boxes: samples x boxes x (xywh)
        # classes: samples x boxes x classes
        # print(pred_classes.contiguous().view(-1).sort()[0])
        # print(pred_classes.contiguous().view(-1).sort()[0][-2:], pred_classes.contiguous().view(-1).argsort()[-2:], annot_classes.contiguous().view(-1).argsort()[-2:])
        entropy = F.binary_cross_entropy(pred_classes.contiguous().view(-1),
                                  annot_classes.contiguous().view(-1), reduce=False)
        inv_entropy = torch.exp(entropy)
        f_loss = (alpha * (1 - inv_entropy) ** gamma) * entropy

        return f_loss.mean()
    return loss


def total_loss():
    def loss(pred_boxes, pred_classes, annot_boxes, annot_classes):
        object_i = torch.sum(annot_classes, dim=-1) > 0

        object_pred_boxes = pred_boxes[object_i]
        object_annot_boxes = annot_boxes[object_i]
        # print(object_pred_boxes[..., 2:].data, object_annot_boxes[..., 2:])

        f_loss = focal_loss(0.25, 2)(pred_classes, annot_classes)

        x_loss = F.binary_cross_entropy(object_pred_boxes[..., 0], object_annot_boxes[..., 0])  # x
        y_loss = F.binary_cross_entropy(object_pred_boxes[..., 1], object_annot_boxes[..., 1])  # y
        w_loss = F.mse_loss(object_pred_boxes[..., 2], object_annot_boxes[..., 2])
        h_loss = F.mse_loss(object_pred_boxes[..., 3], object_annot_boxes[..., 3])
        if torch.isinf(f_loss) or torch.isnan(f_loss):
            print(torch.isinf(pred_classes), torch.isnan(pred_classes))
        # if torch.isinf(f_loss) or torch.isinf(x_loss) or torch.isinf(w_loss) or torch.isinf(h_loss):
        #     print( f_loss , x_loss , y_loss , w_loss , h_loss)
        # if torch.isinf(w_loss):
        #     print(object_annot_boxes[..., 2])

        return f_loss + x_loss + y_loss + w_loss + h_loss
    return loss
