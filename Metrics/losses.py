import torch
import torch.nn as nn

class loss_function(nn.Module):
    def __init__(self):
        super(loss_function, self).__init__()
        self.bce_loss = nn.BCELoss()

    def soft_dice_loss(self, y_true, y_pred):
        smooth = 1e-5
        i = torch.sum(y_true)
        j = torch.sum(y_pred)
        intersection = torch.sum(y_true * y_pred)
        score = (2.0 * intersection + smooth) / (i + j + smooth)
        loss = 1.0 - score.mean()
        return loss

    def iou_loss(self, inputs, targets):
        smooth = 1e-5
        intersection = torch.sum(inputs * targets)
        total = torch.sum(inputs + targets)
        union = total - intersection
        score = (intersection + smooth) / (union + smooth)
        loss = 1 - score.mean()
        return loss

    def forward(self, y_true, y_pred, loss_weights=[1,1,1]):
        assert len(loss_weights) == 3, 'Inappropriate number of weights!'
        dsc = self.soft_dice_loss(y_true, y_pred)
        bce = self.bce_loss(y_pred, y_true)
        iou = self.iou_loss(y_pred, y_true)
        loss = bce * loss_weights[0] + iou * loss_weights[1] + dsc * loss_weights[2]
        return loss