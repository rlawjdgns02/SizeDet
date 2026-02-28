import torch
import torch.nn as nn

from mmdet.models.builder import HEADS
from .obbox_head import OBBoxHead


@HEADS.register_module()
class OBBDecoupBBoxHead(OBBoxHead):
    r"""Decoupled bbox head with separate branches for cls, center(+angle),
    and size.

    .. code-block:: none

        RoI feat (flatten)
            |-> fc_cls_1 -> fc_cls_2 -> fc_cls        -> class prediction
            |-> fc_center -> fc_center_out             -> cx, cy, angle (delta)
            \-> fc_size   -> fc_size_out               -> w, h (delta)
    """  # noqa: W605

    def __init__(self,
                 fc_out_channels=1024,
                 *args,
                 **kwargs):
        super(OBBDecoupBBoxHead, self).__init__(*args, **kwargs)

        self.fc_out_channels = fc_out_channels
        in_features = self.in_channels * self.roi_feat_area  # 256 * 49 = 12544

        self.relu = nn.ReLU(inplace=True)

        # cls branch: fc_cls_1 -> fc_cls_2 -> fc_cls
        self.fc_cls_1 = nn.Linear(in_features, fc_out_channels)
        self.fc_cls_2 = nn.Linear(fc_out_channels, fc_out_channels)
        self.fc_cls = nn.Linear(fc_out_channels, self.num_classes + 1)

        # center branch: fc_center_1 -> LN -> fc_center_2 -> LN -> fc_center_out (cx, cy, angle)
        self.fc_center_1 = nn.Linear(in_features, fc_out_channels//2)
        self.ln_center_1 = nn.LayerNorm(fc_out_channels//2)
        self.fc_center_2 = nn.Linear(fc_out_channels//2, fc_out_channels//2)
        self.ln_center_2 = nn.LayerNorm(fc_out_channels//2)
        center_out_dim = 3 if self.reg_class_agnostic else 3 * self.num_classes
        self.fc_center_out = nn.Linear(fc_out_channels//2, center_out_dim)

        # size branch: fc_size_1 -> LN -> fc_size_2 -> LN -> fc_size_out (w, h)
        self.fc_size_1 = nn.Linear(in_features, fc_out_channels//2)
        self.ln_size_1 = nn.LayerNorm(fc_out_channels//2)
        self.fc_size_2 = nn.Linear(fc_out_channels//2, fc_out_channels//2)
        self.ln_size_2 = nn.LayerNorm(fc_out_channels//2)
        size_out_dim = 2 if self.reg_class_agnostic else 2 * self.num_classes
        self.fc_size_out = nn.Linear(fc_out_channels//2, size_out_dim)

        # remove parent's fc_reg (replaced by fc_center_out + fc_size_out)
        if hasattr(self, 'fc_reg'):
            del self.fc_reg

    def init_weights(self):
        # cls branch
        nn.init.normal_(self.fc_cls.weight, 0, 0.01)
        nn.init.constant_(self.fc_cls.bias, 0)
        for m in [self.fc_cls_1, self.fc_cls_2]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

        # center branch
        for m in [self.fc_center_1, self.fc_center_2]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.fc_center_out.weight, 0, 0.001)
        nn.init.constant_(self.fc_center_out.bias, 0)

        # size branch
        for m in [self.fc_size_1, self.fc_size_2]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.fc_size_out.weight, 0, 0.001)
        nn.init.constant_(self.fc_size_out.bias, 0)

    def forward(self, x):
        x = x.flatten(1)  # (N, 256, 7, 7) -> (N, 12544)

        # cls branch
        x_cls = self.relu(self.fc_cls_1(x))
        x_cls = self.relu(self.fc_cls_2(x_cls))
        cls_score = self.fc_cls(x_cls)

        # center branch (cx, cy, angle)
        x_center = self.relu(self.ln_center_1(self.fc_center_1(x)))
        x_center = self.relu(self.ln_center_2(self.fc_center_2(x_center)))
        center_pred = self.fc_center_out(x_center)

        # size branch (w, h)
        x_size = self.relu(self.ln_size_1(self.fc_size_1(x)))
        x_size = self.relu(self.ln_size_2(self.fc_size_2(x_size)))
        size_pred = self.fc_size_out(x_size)

        # assemble bbox_pred: [dx, dy, dw, dh, dtheta]
        if self.reg_class_agnostic:
            bbox_pred = torch.cat([
                center_pred[:, :2],   # dx, dy
                size_pred,            # dw, dh
                center_pred[:, 2:]    # dtheta
            ], dim=-1)  # (N, 5)
        else:
            N = center_pred.size(0)
            center_pred = center_pred.view(N, -1, 3)  # (N, C, 3)
            size_pred = size_pred.view(N, -1, 2)      # (N, C, 2)
            bbox_pred = torch.cat([
                center_pred[:, :, :2],  # dx, dy
                size_pred,              # dw, dh
                center_pred[:, :, 2:]   # dtheta
            ], dim=-1)  # (N, C, 5)
            bbox_pred = bbox_pred.view(N, -1)  # (N, C*5)

        return cls_score, bbox_pred, x_center
