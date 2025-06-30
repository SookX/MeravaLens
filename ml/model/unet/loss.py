import torch
import torch.nn as nn

class IoULoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(IoULoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        """
        preds: [B, C, H, W] - logits
        targets: [B, H, W]  - class indices
        """
        num_classes = preds.shape[1]
        preds = torch.softmax(preds, dim=1)
        preds = preds.argmax(dim=1)

        iou = 0.0
        for cls in range(num_classes):
            pred_cls = (preds == cls).float()
            target_cls = (targets == cls).float()

            intersection = (pred_cls * target_cls).sum()
            union = pred_cls.sum() + target_cls.sum() - intersection

            iou += (intersection + self.smooth) / (union + self.smooth)

        return 1 - iou / num_classes 