import torch.nn as nn
import torch

from mmdet.core import bbox2result
from .. import builder
from ..registry import DETECTORS
from .base import BaseDetector
from mmdet.core import (bbox2roi, bbox_mapping, merge_aug_bboxes,
                        merge_aug_masks, merge_aug_proposals, multiclass_nms)

@DETECTORS.register_module
class SingleStageDetector(BaseDetector):

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SingleStageDetector, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self.bbox_head = builder.build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super(SingleStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.bbox_head.init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, self.train_cfg)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def simple_test(self, img, img_meta, rescale=False):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results[0]

    def aug_test(self, imgs, img_metas, rescale=False):

        aug_proposals = []
        for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
            x = self.extract_feat(img)
            outs = self.bbox_head(x)
            bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
            bbox_list = self.bbox_head.get_bboxes_aug(*bbox_inputs)
            aug_proposals.append(bbox_list)

        # here now first merge detections, then run NMS
        det_bboxes, det_scores = [], []
        for prop in aug_proposals:
            det_bboxes.append(prop[0][0])
            det_scores.append(prop[0][1])

        det_bboxes = torch.cat(det_bboxes)
        det_scores = torch.cat(det_scores)
        bbox_list = []
        bbox_list.append(multiclass_nms(
            det_bboxes,
            det_scores,
            self.test_cfg.score_thr,
            self.test_cfg.k,
            self.test_cfg.agg_thr,
            self.test_cfg.nms,
            self.test_cfg.max_per_img))

        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results[0]
