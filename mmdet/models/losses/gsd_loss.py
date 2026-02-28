import torch
import torch.nn as nn

from ..builder import LOSSES
from .utils import weighted_loss


@weighted_loss
def gsd_loss(pred, target, area_beta=0.2, ratio_beta=0.4):
    """GSD (Geometric Size Decomposition) Loss
    면적비 오차 + 종횡비 오차를 델타 공간에서 계산

    Args:
        pred (torch.Tensor): shape (N, 5) - [dx, dy, dw, dh, dtheta]
        target (torch.Tensor): shape (N, 5) - 동일 형식
        beta (float): area_loss와 ratio_loss 간의 가중치 밸런스

    Returns:
        torch.Tensor: shape (N, 5) - element-wise loss
    """
    assert pred.size() == target.size() and target.numel() > 0

    # ============================================================
    # pred, target: [dx, dy, dw, dh, dtheta]
    # 각각 index 2가 dw, index 3이 dh
    # ============================================================

    dw_p = pred[:, 2]
    dh_p = pred[:, 3]
    dw_t = target[:, 2]
    dh_t = target[:, 3]

    # ============================================================
    # STEP 1: s와 r 계산
    # ============================================================

    s = (dw_p - dw_t) + (dh_p - dh_t)  # TODO: 면적 오차 (dw, dh의 ___)
    r = (dw_p - dw_t) - (dh_p - dh_t)  # TODO: 종횡비 오차 (dw, dh의 ___)

    # ============================================================
    # STEP 2: 면적비 오차
    # ============================================================

    area_loss = torch.exp(torch.abs(s)) - 1

    # ============================================================
    # STEP 3: 종횡비 오차
    # ============================================================

    ratio_loss = torch.exp(torch.abs(r)) - 1

    # ============================================================
    # STEP 4: 나머지 성분 (dx, dy, dtheta)은 기존 SmoothL1 사용
    # ============================================================

    diff_other = torch.abs(pred - target)  # 전체 (N, 5)
    smooth_other = torch.where(
        diff_other < 1.0,
        0.5 * diff_other * diff_other,
        diff_other - 0.5
    )

    # ============================================================
    # STEP 5: 최종 loss 조합
    # - dw, dh 자리 (index 2, 3)에 area_loss, ratio_loss 배치
    # - 나머지 (index 0, 1, 4)는 smooth_other 그대로
    # ============================================================

    loss = smooth_other.clone()
    loss[:, 2] = smooth_other[:, 2] + area_beta * area_loss  # TODO: dw 자리에 뭘 넣을지
    loss[:, 3] = smooth_other[:, 3] + ratio_beta * ratio_loss  # TODO: dh 자리에 뭘 넣을지

    return loss

@weighted_loss
def gsd_smoothl1_loss(pred, target, beta=1.0):
    """GSD SmoothL1 Loss
    면적비 오차 + 종횡비 오차를 SmoothL1 형태로 계산
    
    Args:
        pred (torch.Tensor): shape (N, 5) - [dx, dy, dw, dh, dtheta]
        target (torch.Tensor): shape (N, 5) - 동일 형식
        area_beta (float): area_loss 가중치
        ratio_beta (float): ratio_loss 가중치
        beta (float): SmoothL1 threshold
        
    Returns:
        torch.Tensor: shape (N, 5) - element-wise loss
    """
    assert pred.size() == target.size() and target.numel() > 0
    
    # ============================================================
    # STEP 1: dw, dh 추출
    # ============================================================
    
    dw_p = pred[:, 2]
    dh_p = pred[:, 3]
    dw_t = target[:, 2]
    dh_t = target[:, 3]
    
    # ============================================================
    # STEP 2: 면적과 종횡비를 log space로 표현
    # ============================================================
    
    pred_area_log = dw_p + dh_p  # TODO: dw_p와 dh_p를 어떻게 조합?
    target_area_log = dw_t + dh_t  # TODO: dw_t와 dh_t를 어떻게 조합?
    
    pred_ratio_log = dw_p - dh_p  # TODO: dw_p와 dh_p를 어떻게 조합?
    target_ratio_log = dw_t - dh_t  # TODO: dw_t와 dh_t를 어떻게 조합?
    
    # ============================================================
    # STEP 3: SmoothL1 계산
    # SmoothL1(x) = 0.5 * x^2 / beta  if |x| < beta
    #             = |x| - 0.5 * beta  otherwise
    # ============================================================
    
    # 면적 오차
    area_diff = torch.abs(pred_area_log - target_area_log)
    area_loss = torch.where(
        area_diff < beta,
        0.5 * area_diff * area_diff / beta,  # TODO: |diff| < beta일 때 공식
        area_diff - 0.5 * beta   # TODO: |diff| >= beta일 때 공식
    )
    
    # 종횡비 오차
    ratio_diff = torch.abs(pred_ratio_log - target_ratio_log)
    ratio_loss = torch.where(
        ratio_diff < beta,
        0.5 * ratio_diff * ratio_diff / beta,  # TODO: |diff| < beta일 때 공식
        ratio_diff - 0.5 * beta   # TODO: |diff| >= beta일 때 공식
    )
    
    # ============================================================
    # STEP 4: 나머지 성분 (dx, dy, dtheta)은 기존 SmoothL1
    # ============================================================
    
    diff_other = torch.abs(pred - target)
    smooth_other = torch.where(
        diff_other < beta,
        0.5 * diff_other * diff_other / beta,
        diff_other - 0.5 * beta
    )
    
    # ============================================================
    # STEP 5: 최종 loss 조합
    # dw, dh 자리에 area_loss, ratio_loss 추가
    # ============================================================
    
    loss = smooth_other.clone()
    loss[:, 2] = smooth_other + area_loss  # TODO: dw 자리 (smooth_other + area)
    loss[:, 3] = smooth_other + ratio_loss  # TODO: dh 자리 (smooth_other + ratio)
    
    return loss



@LOSSES.register_module()
class GSDLoss(nn.Module):
    """GSD Loss Module

    Args:
        area_beta (float): area_loss 가중치. Defaults to 0.2.
        ratio_beta (float): ratio_loss 가중치. Defaults to 0.4.
        reduction (str): 'none', 'mean', 'sum'. Defaults to 'mean'.
        loss_weight (float): loss 전체 가중치. Defaults to 1.0.
    """

    def __init__(self, area_beta=0.2, ratio_beta=0.4,
                 reduction='mean', loss_weight=1.0):
        super(GSDLoss, self).__init__()
        self.area_beta = area_beta
        self.ratio_beta = ratio_beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_bbox = self.loss_weight * gsd_loss(
            pred,
            target,
            weight,
            area_beta=self.area_beta,
            ratio_beta=self.ratio_beta,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_bbox

@LOSSES.register_module()
class GSDSmoothL1Loss(nn.Module):
    """GSD SmoothL1 Loss Module
    
    Args:
        area_beta (float): area_loss 가중치. Defaults to 0.2.
        ratio_beta (float): ratio_loss 가중치. Defaults to 0.4.
        beta (float): SmoothL1 threshold. Defaults to 1.0.
        reduction (str): 'none', 'mean', 'sum'. Defaults to 'mean'.
        loss_weight (float): loss 전체 가중치. Defaults to 1.0.
    """
    
    def __init__(self, area_beta=0.2, ratio_beta=0.4, beta=1.0,
                 reduction='mean', loss_weight=1.0):
        super(GSDSmoothL1Loss, self).__init__()
        self.area_beta = area_beta
        self.ratio_beta = ratio_beta
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight
    
    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_bbox = self.loss_weight * gsd_smoothl1_loss(
            pred,
            target,
            weight,
            area_beta=self.area_beta,
            ratio_beta=self.ratio_beta,
            beta=self.beta,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_bbox