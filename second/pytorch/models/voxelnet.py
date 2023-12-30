import time
from enum import Enum
from functools import reduce
import contextlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import torchplus
from second.pytorch.core import box_torch_ops
from second.pytorch.core.losses import (WeightedSigmoidClassificationLoss,
                                        WeightedSmoothL1LocalizationLoss,
                                        WeightedSoftmaxClassificationLoss)
from second.pytorch.models import middle, pointpillars, rpn, voxel_encoder
from torchplus import metrics
from second.pytorch.utils import torch_timer


def _get_pos_neg_loss(cls_loss, labels):
    # cls_loss: [N, num_anchors, num_class]
    # labels: [N, num_anchors]
    batch_size = cls_loss.shape[0]
    if cls_loss.shape[-1] == 1 or len(cls_loss.shape) == 2:
        cls_pos_loss = (labels > 0).type_as(cls_loss) * cls_loss.view(
            batch_size, -1)
        cls_neg_loss = (labels == 0).type_as(cls_loss) * cls_loss.view(
            batch_size, -1)
        cls_pos_loss = cls_pos_loss.sum() / batch_size
        cls_neg_loss = cls_neg_loss.sum() / batch_size
    else:
        cls_pos_loss = cls_loss[..., 1:].sum() / batch_size
        cls_neg_loss = cls_loss[..., 0].sum() / batch_size
    return cls_pos_loss, cls_neg_loss

REGISTERED_NETWORK_CLASSES = {}

def register_voxelnet(cls, name=None):
    global REGISTERED_NETWORK_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_NETWORK_CLASSES, f"exist class: {REGISTERED_NETWORK_CLASSES}"
    REGISTERED_NETWORK_CLASSES[name] = cls
    return cls

def get_voxelnet_class(name):
    global REGISTERED_NETWORK_CLASSES
    assert name in REGISTERED_NETWORK_CLASSES, f"available class: {REGISTERED_NETWORK_CLASSES}"
    return REGISTERED_NETWORK_CLASSES[name]

class LossNormType(Enum):
    NormByNumPositives = "norm_by_num_positives"
    NormByNumExamples = "norm_by_num_examples"
    NormByNumPosNeg = "norm_by_num_pos_neg"
    DontNorm = "dont_norm"

@register_voxelnet
class VoxelNet(nn.Module):
    def __init__(self,
                 output_shape,
                 num_class=2,
                 num_input_features=4,
                 vfe_class_name="VoxelFeatureExtractor",
                 vfe_num_filters=[32, 128],
                 with_distance=False,
                 middle_class_name="SparseMiddleExtractor",
                 middle_num_input_features=-1,
                 middle_num_filters_d1=[64],
                 middle_num_filters_d2=[64, 64],
                 rpn_class_name="RPN",
                 rpn_num_input_features=-1,
                 rpn_layer_nums=[3, 5, 5],
                 rpn_layer_strides=[2, 2, 2],
                 rpn_num_filters=[128, 128, 256],
                 rpn_upsample_strides=[1, 2, 4],
                 rpn_num_upsample_filters=[256, 256, 256],
                 use_norm=True,
                 use_groupnorm=False,
                 num_groups=32,
                 use_direction_classifier=True,
                 use_sigmoid_score=False,
                 encode_background_as_zeros=True,
                 use_rotate_nms=True,
                 multiclass_nms=False,
                 nms_score_thresholds=None,
                 nms_pre_max_sizes=None,
                 nms_post_max_sizes=None,
                 nms_iou_thresholds=None,
                 target_assigner=None,
                 cls_loss_weight=1.0,
                 loc_loss_weight=1.0,
                 pos_cls_weight=1.0,
                 neg_cls_weight=1.0,
                 direction_loss_weight=1.0,
                 loss_norm_type=LossNormType.NormByNumPositives,
                 encode_rad_error_by_sin=False,
                 loc_loss_ftor=None,
                 cls_loss_ftor=None,
                 measure_time=False,
                 voxel_generator=None,
                 post_center_range=None,
                 dir_offset=0.0,
                 sin_error_factor=1.0,
                 nms_class_agnostic=False,
                 num_direction_bins=2,
                 direction_limit_offset=0,
                 name='voxelnet'):
        super().__init__()
        self.name = name
        self._sin_error_factor = sin_error_factor
        self._num_class = num_class
        self._use_rotate_nms = use_rotate_nms
        self._multiclass_nms = multiclass_nms
        self._nms_score_thresholds = nms_score_thresholds
        self._nms_pre_max_sizes = nms_pre_max_sizes
        self._nms_post_max_sizes = nms_post_max_sizes
        self._nms_iou_thresholds = nms_iou_thresholds
        self._use_sigmoid_score = use_sigmoid_score
        self._encode_background_as_zeros = encode_background_as_zeros
        self._use_direction_classifier = use_direction_classifier
        self._num_input_features = num_input_features
        self._box_coder = target_assigner.box_coder
        self.target_assigner = target_assigner
        self.voxel_generator = voxel_generator
        self._pos_cls_weight = pos_cls_weight
        self._neg_cls_weight = neg_cls_weight
        self._encode_rad_error_by_sin = encode_rad_error_by_sin
        self._loss_norm_type = loss_norm_type
        self._dir_loss_ftor = WeightedSoftmaxClassificationLoss()
        self._diff_loc_loss_ftor = WeightedSmoothL1LocalizationLoss()
        self._dir_offset = dir_offset
        self._loc_loss_ftor = loc_loss_ftor
        self._cls_loss_ftor = cls_loss_ftor
        self._direction_loss_weight = direction_loss_weight
        self._cls_loss_weight = cls_loss_weight
        self._loc_loss_weight = loc_loss_weight
        self._post_center_range = post_center_range or []
        self.measure_time = measure_time
        self._nms_class_agnostic = nms_class_agnostic
        self._num_direction_bins = num_direction_bins
        self._dir_limit_offset = direction_limit_offset
        self.voxel_feature_extractor = voxel_encoder.get_vfe_class(vfe_class_name)(
            num_input_features,
            use_norm,
            num_filters=vfe_num_filters,
            with_distance=with_distance,
            voxel_size=self.voxel_generator.voxel_size,
            pc_range=self.voxel_generator.point_cloud_range,
        )
        self.middle_feature_extractor = middle.get_middle_class(middle_class_name)(
            output_shape,
            use_norm,
            num_input_features=middle_num_input_features,
            num_filters_down1=middle_num_filters_d1,
            num_filters_down2=middle_num_filters_d2)
        self.rpn = rpn.get_rpn_class(rpn_class_name)(
            use_norm=True,
            num_class=num_class,
            layer_nums=rpn_layer_nums,
            layer_strides=rpn_layer_strides,
            num_filters=rpn_num_filters,
            upsample_strides=rpn_upsample_strides,
            num_upsample_filters=rpn_num_upsample_filters,
            num_input_features=rpn_num_input_features,
            num_anchor_per_loc=target_assigner.num_anchors_per_location,
            encode_background_as_zeros=encode_background_as_zeros,
            use_direction_classifier=use_direction_classifier,
            use_groupnorm=use_groupnorm,
            num_groups=num_groups,
            box_code_size=target_assigner.box_coder.code_size,
            num_direction_bins=self._num_direction_bins)
        self.rpn_acc = metrics.Accuracy(
            dim=-1, encode_background_as_zeros=encode_background_as_zeros)
        self.rpn_precision = metrics.Precision(dim=-1)
        self.rpn_recall = metrics.Recall(dim=-1)
        self.rpn_metrics = metrics.PrecisionRecall(
            dim=-1,
            thresholds=[0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95],
            use_sigmoid_score=use_sigmoid_score,
            encode_background_as_zeros=encode_background_as_zeros)

        self.rpn_cls_loss = metrics.Scalar()
        self.rpn_loc_loss = metrics.Scalar()
        self.rpn_total_loss = metrics.Scalar()
        self.register_buffer("global_step", torch.LongTensor(1).zero_())

        self._time_dict = {}
        self._time_total_dict = {}
        self._time_count_dict = {}

    def start_timer(self, *names):
        if not self.measure_time:
            return
        torch.cuda.synchronize()
        for name in names:
            self._time_dict[name] = time.time()

    def end_timer(self, name):
        if not self.measure_time:
            return
        torch.cuda.synchronize()
        time_elapsed = time.time() - self._time_dict[name]
        if name not in self._time_count_dict:
            self._time_count_dict[name] = 1
            self._time_total_dict[name] = time_elapsed
        else:
            self._time_count_dict[name] += 1
            self._time_total_dict[name] += time_elapsed
        self._time_dict[name] = 0

    def clear_timer(self):
        self._time_count_dict.clear()
        self._time_dict.clear()
        self._time_total_dict.clear()

    @contextlib.contextmanager
    def profiler(self):
        old_measure_time = self.measure_time
        self.measure_time = True
        yield
        self.measure_time = old_measure_time

    def get_avg_time_dict(self):
        ret = {}
        for name, val in self._time_total_dict.items():
            count = self._time_count_dict[name]
            ret[name] = val / max(1, count)
        return ret

    def update_global_step(self):
        self.global_step += 1

    def get_global_step(self):
        return int(self.global_step.cpu().numpy()[0])

    def clear_global_step(self):
        self.global_step.zero_()

    def loss(self, example, preds_dict):
        box_preds = preds_dict["box_preds"]
        cls_preds = preds_dict["cls_preds"]
        batch_size_dev = cls_preds.shape[0]
        self.start_timer("loss forward")
        labels = example['labels']
        reg_targets = example['reg_targets']
        importance = example['importance']
        self.start_timer("prepare weight forward")
        cls_weights, reg_weights, cared = prepare_loss_weights(
            labels,
            pos_cls_weight=self._pos_cls_weight,
            neg_cls_weight=self._neg_cls_weight,
            loss_norm_type=self._loss_norm_type,
            dtype=box_preds.dtype)

        cls_targets = labels * cared.type_as(labels)
        cls_targets = cls_targets.unsqueeze(-1)
        self.end_timer("prepare weight forward")
        self.start_timer("create_loss forward")
        loc_loss, cls_loss = create_loss(
            self._loc_loss_ftor,
            self._cls_loss_ftor,
            box_preds=box_preds,
            cls_preds=cls_preds,
            cls_targets=cls_targets,
            cls_weights=cls_weights * importance,
            reg_targets=reg_targets,
            reg_weights=reg_weights * importance,
            num_class=self._num_class,
            encode_rad_error_by_sin=self._encode_rad_error_by_sin,
            encode_background_as_zeros=self._encode_background_as_zeros,
            box_code_size=self._box_coder.code_size,
            sin_error_factor=self._sin_error_factor,
            num_direction_bins=self._num_direction_bins,
        )
        loc_loss_reduced = loc_loss.sum() / batch_size_dev
        loc_loss_reduced *= self._loc_loss_weight
        cls_pos_loss, cls_neg_loss = _get_pos_neg_loss(cls_loss, labels)
        cls_pos_loss /= self._pos_cls_weight
        cls_neg_loss /= self._neg_cls_weight
        cls_loss_reduced = cls_loss.sum() / batch_size_dev
        cls_loss_reduced *= self._cls_loss_weight
        loss = loc_loss_reduced + cls_loss_reduced
        self.end_timer("create_loss forward")
        if self._use_direction_classifier:
            dir_targets = get_direction_target(
                example['anchors'],
                reg_targets,
                dir_offset=self._dir_offset,
                num_bins=self._num_direction_bins)
            dir_logits = preds_dict["dir_cls_preds"].view(
                batch_size_dev, -1, self._num_direction_bins)
            weights = (labels > 0).type_as(dir_logits) * importance
            weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
            dir_loss = self._dir_loss_ftor(
                dir_logits, dir_targets, weights=weights)
            dir_loss = dir_loss.sum() / batch_size_dev
            loss += dir_loss * self._direction_loss_weight
        self.end_timer("loss forward")
        res = {
            "loss": loss,
            