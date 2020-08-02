import torch

from mmdet.ops.nms import nms_wrapper
from mmdet.core.bbox.geometry import bbox_overlaps


def multiclass_nms(multi_bboxes,
                   multi_scores,
                   score_thr,
                   k,
                   agg_thr,
                   nms_cfg,
                   max_num=-1,
                   score_factors=None):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class)
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels
            are 0-based.
    """
    num_classes = multi_scores.shape[1]
    bboxes, labels = [], []
    nms_cfg_ = nms_cfg.copy()
    nms_type = nms_cfg_.pop('type', 'nms')
    nms_op = getattr(nms_wrapper, nms_type)
    for i in range(1, num_classes):
        cls_inds = multi_scores[:, i] > score_thr
        if not cls_inds.any():
            continue
        # get bboxes and scores of this class
        if multi_bboxes.shape[1] == 4:
            _bboxes = multi_bboxes[cls_inds, :]
        else:
            _bboxes = multi_bboxes[cls_inds, i * 4:(i + 1) * 4]
        _scores = multi_scores[cls_inds, i]
        if score_factors is not None:
            _scores *= score_factors[cls_inds]
        cls_dets = torch.cat([_bboxes, _scores[:, None]], dim=1)
        cls_dets = agg_scores(cls_dets, k, agg_thr)
        cls_dets, _ = nms_op(cls_dets, **nms_cfg_)
        cls_labels = multi_bboxes.new_full((cls_dets.shape[0], ),
                                           i - 1,
                                           dtype=torch.long)
        bboxes.append(cls_dets)
        labels.append(cls_labels)
    if bboxes:
        bboxes = torch.cat(bboxes)
        labels = torch.cat(labels)
        if bboxes.shape[0] > max_num:
            _, inds = bboxes[:, -1].sort(descending=True)
            inds = inds[:max_num]
            bboxes = bboxes[inds]
            labels = labels[inds]
    else:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)

    return bboxes, labels


def agg_scores(cls_dets, k, agg_thr):

    cls_dets_agg = cls_dets.clone()
    iou = calc_iou(cls_dets[:, :4], cls_dets[:, :4])

    for i,det in enumerate(cls_dets):

        agg_ind = (iou[i] >= agg_thr)
        iou_weight = iou[i, agg_ind] - 1
        iou_weight = k ** iou_weight
        agg_score = (iou_weight * cls_dets[agg_ind, 4]).sum()
        cls_dets_agg[i, 4] = agg_score

    return cls_dets_agg


def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih

    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua

    return IoU

