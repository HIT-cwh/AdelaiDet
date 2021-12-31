# -*- coding: utf-8 -*-
import logging
from skimage import color
import copy
from scipy.optimize import linear_sum_assignment
import cv2
import os.path as osp
import random

import torch
from torch import nn
import torch.nn.functional as F

from detectron2.structures import ImageList
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.structures.instances import Instances
from detectron2.structures.masks import PolygonMasks, polygons_to_bitmask
from detectron2.utils.comm import get_rank
from mmcv.runner import load_checkpoint
import mmcv

from .condinst import CondInst
from .dynamic_mask_head import build_dynamic_mask_head
from .mask_branch import build_mask_branch

from adet.utils.comm import aligned_bilinear


LOSS_DICT = dict()
def loss_register(name, parser_dict, force=False):

    def _register(parser_func):
        if (name not in parser_dict) or force:
            parser_dict[name] = parser_func
        else:
            raise KeyError(
                f'{name} is already registered in task_dict, '
                'add "force=True" if you want to override it')
        return parser_func

    return _register


def build_loss(cfg):
    type = cfg['type']
    cfg.pop('type')
    return LOSS_DICT[type](**cfg)


def dice_coefficient(x, target):
    eps = 1e-5
    n_inst = x.size(0)
    x = x.reshape(n_inst, -1)
    target = target.reshape(n_inst, -1)
    intersection = (x * target).sum(dim=1)
    union = (x ** 2.0).sum(dim=1) + (target ** 2.0).sum(dim=1) + eps
    loss = 1. - (2 * intersection / union)
    return loss


def nms(loss, gt_inds):
    assert len(loss.shape) == 1
    loss_inds = torch.arange(loss.size(0), dtype=torch.int, device=loss.device)
    inds_unique = gt_inds.unique()
    nms_loss_inds = []
    for ind in inds_unique:
        loss_per_ins = loss[gt_inds==ind]
        loss_inds_per_ins = loss_inds[gt_inds==ind]
        nms_loss_inds.append(loss_inds_per_ins[loss_per_ins.argmin()])
    return torch.tensor(nms_loss_inds, dtype=torch.long, device=loss.device), inds_unique


def weighted_sum_teacher(preds_T, loss, gt_inds):
    """利用dice_coefficient得到加权后的preds_T"""
    assert len(loss.shape) == 1
    # loss_inds = torch.arange(loss.size(0), dtype=torch.int, device=loss.device)
    inds_unique = gt_inds.unique()
    preds = []
    for ind in inds_unique:
        preds_T_per_ins = preds_T[gt_inds == ind]
        loss_per_ins = loss[gt_inds == ind]
        loss_per_ins = nn.Softmax(dim=0)(1/loss_per_ins)
        preds.append((preds_T_per_ins * loss_per_ins.reshape(-1, 1, 1)).sum(dim=0, keepdim=True))
    preds = torch.cat(preds, dim=0).to(preds_T.device)
    return preds, inds_unique


def weighted_choice_teacher(preds_T, loss, gt_inds):
    """利用dice_coefficient得到loss加权抽样preds_T"""
    assert len(loss.shape) == 1
    # loss_inds = torch.arange(loss.size(0), dtype=torch.int, device=loss.device)
    inds_unique = gt_inds.unique()
    preds = []
    for ind in inds_unique:
        preds_T_per_ins = preds_T[gt_inds == ind]
        loss_per_ins = loss[gt_inds == ind]
        loss_per_ins = nn.Softmax(dim=0)(loss_per_ins)
        ind = torch.multinomial(loss_per_ins, 1).to(preds_T_per_ins.device)
        preds.append(preds_T_per_ins[ind])
    preds = torch.cat(preds, dim=0).to(preds_T.device)
    return preds, inds_unique


def visualization(im_ind, root, *preds):
    assert len(preds) > 0
    shape = preds[0].shape
    for pred in preds:
        assert pred.shape == shape
    preds = list(preds)
    for i, pred in enumerate(preds):
        preds[i] = torch.clamp((pred.sigmoid() * 255).int(), 0, 255)
    output = torch.cat(preds, dim=-1).cpu().numpy()
    mmcv.mkdir_or_exist(root)
    for i, img in enumerate(output):
        cv2.imwrite(osp.join(root, f'img_{im_ind}_mask_{i}.png'), img)


@loss_register('cwd', LOSS_DICT)
class ChannelWiseDivergence(nn.Module):
    def __init__(
        self,
        tau=1.0,
        loss_weight=1.0,
    ):
        super(ChannelWiseDivergence, self).__init__()
        self.tau = tau
        self.loss_weight = loss_weight
        self.visualization = True

    def match(self, nms_preds_T, preds_S, nms_gt_inds_T, gt_inds_S):
        t_inds = []
        s_inds = []
        for gt_ind in gt_inds_S:
            arg = torch.where(nms_gt_inds_T==gt_ind)[0]
            # 当前stu输出对应的gt也有对应的tea的mask，由于nms_gt_inds_T不重复，
            # 所以arg要么是空，即tea没有对应的mask，要么是有一个值即第几个tea mask
            if arg.size(0):
                s_inds.append(arg.size(0) == 1)
                t_inds.append(arg)
        t_inds = torch.tensor(t_inds, dtype=torch.long, device=nms_preds_T.device)
        s_inds = torch.tensor(s_inds, dtype=torch.bool, device=preds_S.device)
        return t_inds, s_inds
        # return nms_preds_T[t_inds], preds_S[s_inds], gt_S[s_inds]

    def cal_loss(self, channel_T, channel_S, **kwargs):
        N, W, H = channel_S.shape
        softmax_pred_T = F.softmax(channel_T.reshape(-1, W * H) / self.tau, dim=1)

        logsoftmax = torch.nn.LogSoftmax(dim=1)
        loss = torch.sum(softmax_pred_T *
                         logsoftmax(channel_T.reshape(-1, W * H) / self.tau) -
                         softmax_pred_T *
                         logsoftmax(channel_S.reshape(-1, W * H) / self.tau)) * (
                             self.tau**2)
        return loss

    def forward(self, preds_T, preds_S, im_ind, gt_T, gt_S, iter, gt_inds_T, gt_inds_S):
        iou_loss = dice_coefficient(preds_T, gt_T)
        nms_preds_T_inds, nms_gt_inds_T = nms(iou_loss, gt_inds_T)
        nms_preds_T = preds_T[nms_preds_T_inds]
        t_inds, s_inds = self.match(nms_preds_T, preds_S, nms_gt_inds_T, gt_inds_S)
        preds_T, preds_S, gt_S = nms_preds_T[t_inds], preds_S[s_inds], gt_S[s_inds]
        # preds_T, preds_S, gt_S = self.match(nms_preds_T, preds_S, nms_gt_inds_T, gt_inds_S, gt_S)
        if self.visualization and iter % 1000 == 0 and get_rank() == 0:
            visualization(im_ind, f'visualization_iter{iter}', preds_T, preds_S, gt_S)
        loss = self.loss_weight * self.cal_loss(preds_T, preds_S)
        return loss


@loss_register('cwd_roi', LOSS_DICT)
class ChannelWiseDivergence_roi(ChannelWiseDivergence):
    def __init__(
        self,
        tau=1.0,
        loss_weight=1.0
    ):
        super(ChannelWiseDivergence_roi, self).__init__(tau, loss_weight)
        self.visualization = True

    def cal_loss(self, preds_T, preds_S, gts_S):
        gts_S = gts_S.bool().detach()
        loss = 0.
        for pred_T, pred_S, gt_S in zip(preds_T, preds_S, gts_S):
            arg = torch.where(gt_S == True)
            try:
                x_min, x_max, y_min, y_max = arg[0].min(), arg[0].max(), arg[1].min(), arg[1].max()
            except:
                continue
            assert len(pred_T.shape) == 2 and len(pred_S.shape) == 2
            pred_T, pred_S = pred_T[x_min:x_max+1, y_min:y_max+1], pred_S[x_min:x_max+1, y_min:y_max+1]
            pred_T, pred_S = pred_T.flatten(), pred_S.flatten()
            softmax_pred_T = F.softmax(pred_T / self.tau)
            logsoftmax = torch.nn.LogSoftmax(dim=0)
            loss += torch.sum(softmax_pred_T * logsoftmax(pred_T / self.tau) -
                              softmax_pred_T * logsoftmax(pred_S / self.tau)) * (self.tau ** 2)
        return loss

    def forward(self, preds_T, preds_S, im_ind, gt_T, gt_S, iter, gt_inds_T, gt_inds_S):
        iou_loss = dice_coefficient(preds_T, gt_T)
        # get weighted teacher mask corresponding to a gt
        nms_preds_T, nms_gt_inds_T = weighted_sum_teacher(preds_T, iou_loss, gt_inds_T)
        # match the teacher mask and the student mask according to the gt of this student mask
        t_inds, s_inds = self.match(nms_preds_T, preds_S, nms_gt_inds_T,gt_inds_S)
        preds_T, preds_S, gt_S = nms_preds_T[t_inds], preds_S[s_inds], gt_S[s_inds]
        # preds_T, preds_S, gt_S = self.match(nms_preds_T, preds_S, nms_gt_inds_T,gt_inds_S)
        if self.visualization and iter % 5000 == 0 and get_rank() == 0:
            visualization(im_ind, f'dice_cwd_iter{iter}', preds_T, preds_S, gt_S)
        loss = self.loss_weight * self.cal_loss(preds_T, preds_S, gt_S)
        return loss


@loss_register('cwd_all_insts', LOSS_DICT)
class ChannelWiseDivergence_all_insts(ChannelWiseDivergence):
    def __init__(
        self,
        tau=1.0,
        loss_weight=1.0
    ):
        super(ChannelWiseDivergence_all_insts, self).__init__(tau, loss_weight)
        self.visualization = True
        self.frequency = 5000
        self.use_adaptive_tau = False

    def cal_loss(self, preds_T, preds_S, gts_S):
        gts_S = gts_S.bool().detach()
        # 对gts_S的每个mask按位取或
        if len(gts_S) == 0:
            return torch.tensor(0, device=preds_T.device)
        gt_S_all_insts = gts_S[0]
        for gt in gts_S[1:]:
            gt_S_all_insts = torch.bitwise_or(gt_S_all_insts, gt)
        loss = 0.
        for pred_T, pred_S, gt_S in zip(preds_T, preds_S, gts_S):
            # 自适应的tau
            if gt_S.sum() == 0 or gt_S_all_insts.sum() == 0:
                continue
            tau = gt_S_all_insts.sum() / gt_S.sum() * self.tau if self.use_adaptive_tau else self.tau
            # tau = gt_S_all_insts.sum() / gt_S.sum() * self.tau
            # if get_rank() == 0:
            #     print(tau)
            pred_T, pred_S = pred_T[gt_S_all_insts], pred_S[gt_S_all_insts]
            assert len(pred_T.shape) == 1 and len(pred_S.shape) == 1
            softmax_pred_T = F.softmax(pred_T / tau)
            logsoftmax = torch.nn.LogSoftmax(dim=0)
            loss += torch.sum(softmax_pred_T * logsoftmax(pred_T / tau) -
                              softmax_pred_T * logsoftmax(pred_S / tau)) * (tau ** 2)
        return loss

    def forward(self, preds_T, preds_S, im_ind, gt_T, gt_S, iter, gt_inds_T, gt_inds_S):
        iou_loss = dice_coefficient(preds_T, gt_T)
        # get weighted teacher mask corresponding to a gt
        nms_preds_T, nms_gt_inds_T = weighted_sum_teacher(preds_T, iou_loss, gt_inds_T)
        # match the teacher mask and the student mask according to the gt of this student mask
        t_inds, s_inds = self.match(nms_preds_T, preds_S, nms_gt_inds_T, gt_inds_S)
        preds_T, preds_S, gt_S = nms_preds_T[t_inds], preds_S[s_inds], gt_S[s_inds]
        if self.visualization and iter % self.frequency == 0 and get_rank() == 0:
            visualization(im_ind, f'cwd_all_insts_{iter}', preds_T, preds_S, gt_S)
        loss = self.loss_weight * self.cal_loss(preds_T, preds_S, gt_S)
        return loss


@loss_register('cwd_two_insts', LOSS_DICT)
class ChannelWiseDivergence_two_insts(ChannelWiseDivergence_all_insts):
    def __init__(
        self,
        tau=1.0,
        loss_weight=1.0
    ):
        super(ChannelWiseDivergence_two_insts, self).__init__(tau, loss_weight)
        self.visualization = True

    def cal_loss(self, preds_T, preds_S, gts_S):
        gts_S = gts_S.bool().detach()
        l = len(gts_S)
        if l == 0:
            return torch.tensor(0, device=preds_T.device)
        loss = 0.
        for pred_T, pred_S, gt_S in zip(preds_T, preds_S, gts_S):
            gt_S_another = gts_S[random.randrange(0, l)]
            gt_S = torch.bitwise_or(gt_S, gt_S_another)
            pred_T, pred_S = pred_T[gt_S], pred_S[gt_S]
            assert len(pred_T.shape) == 1 and len(pred_S.shape) == 1
            softmax_pred_T = F.softmax(pred_T / self.tau)
            logsoftmax = torch.nn.LogSoftmax(dim=0)
            loss += torch.sum(softmax_pred_T * logsoftmax(pred_T / self.tau) -
                              softmax_pred_T * logsoftmax(pred_S / self.tau)) * (self.tau ** 2)
        return loss
