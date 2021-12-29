# -*- coding: utf-8 -*-
import logging
from skimage import color
import copy
from scipy.optimize import linear_sum_assignment
import cv2
import os.path as osp

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
from .cwd import build_loss

__all__ = ["CondInst_distill"]

logger = logging.getLogger(__name__)

# LOSS_DICT = dict()
# def loss_register(name, parser_dict, force=False):
#
#     def _register(parser_func):
#         if (name not in parser_dict) or force:
#             parser_dict[name] = parser_func
#         else:
#             raise KeyError(
#                 f'{name} is already registered in task_dict, '
#                 'add "force=True" if you want to override it')
#         return parser_func
#
#     return _register
#
# def build_loss(cfg):
#     type = cfg['type']
#     cfg.pop('type')
#     return LOSS_DICT[type](**cfg)
#
#
# def dice_coefficient(x, target):
#     eps = 1e-5
#     n_inst = x.size(0)
#     x = x.reshape(n_inst, -1)
#     target = target.reshape(n_inst, -1)
#     intersection = (x * target).sum(dim=1)
#     union = (x ** 2.0).sum(dim=1) + (target ** 2.0).sum(dim=1) + eps
#     loss = 1. - (2 * intersection / union)
#     return loss
#
#
# def nms(loss, gt_inds):
#     assert len(loss.shape) == 1
#     loss_inds = torch.arange(loss.size(0), dtype=torch.int, device=loss.device)
#     inds_unique = gt_inds.unique()
#     nms_loss_inds = []
#     for ind in inds_unique:
#         loss_per_ins = loss[gt_inds==ind]
#         loss_inds_per_ins = loss_inds[gt_inds==ind]
#         nms_loss_inds.append(loss_inds_per_ins[loss_per_ins.argmin()])
#     return torch.tensor(nms_loss_inds, dtype=torch.long, device=loss.device), inds_unique
#
#
# def weighted_sum_teacher(preds_T, loss, gt_inds):
#     """利用dice_coefficient得到加权后的preds_T"""
#     assert len(loss.shape) == 1
#     # loss_inds = torch.arange(loss.size(0), dtype=torch.int, device=loss.device)
#     inds_unique = gt_inds.unique()
#     preds = []
#     for ind in inds_unique:
#         preds_T_per_ins = preds_T[gt_inds == ind]
#         loss_per_ins = loss[gt_inds == ind]
#         loss_per_ins = nn.Softmax(dim=0)(loss_per_ins)
#         preds.append((preds_T_per_ins * loss_per_ins.reshape(-1, 1, 1)).sum(dim=0, keepdim=True))
#     preds = torch.cat(preds, dim=0).to(preds_T.device)
#     return preds, inds_unique
#
#
# def weighted_choice_teacher(preds_T, loss, gt_inds):
#     """利用dice_coefficient得到loss加权抽样preds_T"""
#     assert len(loss.shape) == 1
#     # loss_inds = torch.arange(loss.size(0), dtype=torch.int, device=loss.device)
#     inds_unique = gt_inds.unique()
#     preds = []
#     for ind in inds_unique:
#         preds_T_per_ins = preds_T[gt_inds == ind]
#         loss_per_ins = loss[gt_inds == ind]
#         loss_per_ins = nn.Softmax(dim=0)(loss_per_ins)
#         ind = torch.multinomial(loss_per_ins, 1).to(preds_T_per_ins.device)
#         preds.append(preds_T_per_ins[ind])
#     preds = torch.cat(preds, dim=0).to(preds_T.device)
#     return preds, inds_unique
#
#
# def visualization(im_ind, root, *preds):
#     assert len(preds) > 0
#     shape = preds[0].shape
#     for pred in preds:
#         assert pred.shape == shape
#     preds = list(preds)
#     for i, pred in enumerate(preds):
#         preds[i] = torch.clamp((pred.sigmoid() * 255).int(), 0, 255)
#     output = torch.cat(preds, dim=-1).cpu().numpy()
#     mmcv.mkdir_or_exist(root)
#     for i, img in enumerate(output):
#         cv2.imwrite(osp.join(root, f'img_{im_ind}_mask_{i}.png'), img)


# @loss_register('cwd', LOSS_DICT)
# class ChannelWiseDivergence(nn.Module):
#     def __init__(
#         self,
#         tau=1.0,
#         loss_weight=1.0,
#     ):
#         super(ChannelWiseDivergence, self).__init__()
#         self.tau = tau
#         self.loss_weight = loss_weight
#         self.visualization = True
#
#     def match(self, nms_preds_T, preds_S, nms_gt_inds_T, gt_inds_S, gt_S):
#         t_inds = []
#         s_inds = []
#         for gt_ind in gt_inds_S:
#             arg = torch.where(nms_gt_inds_T==gt_ind)[0]
#             # 当前stu输出对应的gt也有对应的tea的mask，由于nms_gt_inds_T不重复，
#             # 所以arg要么是空，即tea没有对应的mask，要么是有一个值即第几个tea mask
#             if arg.size(0):
#                 s_inds.append(arg.size(0) == 1)
#                 t_inds.append(arg)
#         t_inds = torch.tensor(t_inds, dtype=torch.long, device=nms_preds_T.device)
#         s_inds = torch.tensor(s_inds, dtype=torch.bool, device=preds_S.device)
#         return nms_preds_T[t_inds], preds_S[s_inds], gt_S[s_inds]
#
#     def cal_loss(self, channel_T, channel_S):
#         N, W, H = channel_S.shape
#         softmax_pred_T = F.softmax(channel_T.reshape(-1, W * H) / self.tau, dim=1)
#
#         logsoftmax = torch.nn.LogSoftmax(dim=1)
#         loss = torch.sum(softmax_pred_T *
#                          logsoftmax(channel_T.reshape(-1, W * H) / self.tau) -
#                          softmax_pred_T *
#                          logsoftmax(channel_S.reshape(-1, W * H) / self.tau)) * (
#                              self.tau**2)
#         return loss
#
#     def forward(self, preds_T, preds_S, im_ind, gt_T, gt_S, iter, gt_inds_T, gt_inds_S):
#         iou_loss = dice_coefficient(preds_T, gt_T)
#         nms_preds_T_inds, nms_gt_inds_T = nms(iou_loss, gt_inds_T)
#         nms_preds_T = preds_T[nms_preds_T_inds]
#         preds_T, preds_S, gt_S = self.match(nms_preds_T, preds_S, nms_gt_inds_T, gt_inds_S, gt_S)
#         if self.visualization and iter % 1000 == 0 and get_rank() == 0:
#             visualization(im_ind, f'visualization_iter{iter}', preds_T, preds_S, gt_S)
#         loss = self.loss_weight * self.cal_loss(preds_T, preds_S)
#         return loss
#
#
# @loss_register('cwd_match', LOSS_DICT)
# class ChannelWiseDivergence_match(nn.Module):
#
#     def __init__(self, tau=1.0, loss_weight=1.0):
#         super(ChannelWiseDivergence_match, self).__init__()
#         self.tau = tau
#         self.loss_weight = loss_weight
#         # self.visualization = False
#
#     def cal_loss(self, channel_T, channel_S):
#         N, W, H = channel_S.shape
#         softmax_pred_T = F.softmax(channel_T.reshape(-1, W * H) / self.tau, dim=1)
#
#         logsoftmax = torch.nn.LogSoftmax(dim=1)
#         loss = torch.sum(softmax_pred_T *
#                          logsoftmax(channel_T.reshape(-1, W * H) / self.tau) -
#                          softmax_pred_T *
#                          logsoftmax(channel_S.reshape(-1, W * H) / self.tau)) * (
#                              self.tau**2)
#
#         # loss = loss / N
#         return loss
#
#     def compute_kl_matrix(self, logits_p, logits_q):
#         N_p, W, H = logits_p.shape
#         N_q = logits_q.size(0)
#         logits_p, logits_q = logits_p.detach(), logits_q.detach()
#
#         prob_p = F.softmax(logits_p.reshape(N_p, W * H) / self.tau, dim=-1)
#
#         log_prob_p = F.log_softmax(
#             logits_p.reshape(N_p, W * H) / self.tau, dim=-1)
#         log_prob_q = F.log_softmax(
#             logits_q.reshape(N_q, W * H) / self.tau, dim=-1)
#
#         cross_entropy_matrix = -torch.einsum('ik, jk -> ij', [prob_p, log_prob_q])
#         # cross_entropy_matrix = -torch.bmm(log_prob_q, prob_p.transpose(1, 2))
#         entropy_p = -(prob_p * log_prob_p).sum(-1, keepdim=True)
#
#         kl_matrix = cross_entropy_matrix - entropy_p
#
#         return kl_matrix
#
#     def forward(self, preds_T, preds_S, im_ind, gt, iter):
#         """Forward function."""
#         assert preds_S.shape[-2:] == preds_T.shape[-2:]
#
#         t_s_kl_matrix = self.compute_kl_matrix(preds_T, preds_S)
#         t_gt_kl_matrix = self.compute_kl_matrix(preds_T, gt)
#         try:
#             t_s_t_ind, t_s_s_ind = linear_sum_assignment(t_s_kl_matrix.cpu())
#             t_gt_t_ind, t_gt_gt_ind = linear_sum_assignment(t_gt_kl_matrix.cpu())
#         except:
#             if get_rank() == 0:
#                 print(preds_S)
#             input()
#         match = t_s_s_ind == t_gt_gt_ind
#         # if get_rank() == 0:
#         #     print(match.sum())
#         if get_rank() == 0 and iter % 100 == 0 and match.sum() > 0:
#             visualization(im_ind, f'visualization_iter{iter}',
#                           preds_T[t_s_t_ind[match]], preds_S[t_s_s_ind[match]], gt[t_s_s_ind[match]])
#
#         if match.sum() > 0:
#             loss = self.loss_weight * self.cal_loss(preds_T[t_s_t_ind[match]],
#                                                     preds_S[t_s_s_ind[match]])
#         else:
#             loss = torch.zeros((), device=preds_S.device)
#         return loss, match.sum()
#
#
# @loss_register('cwd_dice', LOSS_DICT)
# class ChannelWiseDivergence_dice(nn.Module):
#     def __init__(
#         self,
#         tau=1.0,
#         loss_weight=1.0,
#     ):
#         super(ChannelWiseDivergence_dice, self).__init__()
#         self.tau = tau
#         self.loss_weight = loss_weight
#         self.visualization = False
#
#     def match(self, nms_preds_T, preds_S, nms_gt_inds_T, gt_inds_S, gt_S):
#         t_inds = []
#         s_inds = []
#         for gt_ind in gt_inds_S:
#             arg = torch.where(nms_gt_inds_T==gt_ind)[0]
#             if arg.size(0):
#                 s_inds.append(arg.size(0) == 1)
#                 t_inds.append(arg)
#         t_inds = torch.tensor(t_inds, dtype=torch.long, device=nms_preds_T.device)
#         s_inds = torch.tensor(s_inds, dtype=torch.bool, device=preds_S.device)
#         return nms_preds_T[t_inds], preds_S[s_inds], gt_S[s_inds]
#
#     def cal_loss(self, preds_T, preds_S, gts_S):
#         gts_S = gts_S.bool().detach()
#         loss = 0.
#         for pred_T, pred_S, gt_S in zip(preds_T, preds_S, gts_S):
#             pred_T, pred_S = pred_T[gt_S], pred_S[gt_S]
#             assert len(pred_T.shape) == 1 and len(pred_S.shape) == 1
#             softmax_pred_T = F.softmax(pred_T / self.tau)
#             logsoftmax = torch.nn.LogSoftmax(dim=0)
#             loss += torch.sum(softmax_pred_T * logsoftmax(pred_T / self.tau) -
#                               softmax_pred_T * logsoftmax(pred_S / self.tau)) * (self.tau ** 2)
#         return loss
#
#     def forward(self, preds_T, preds_S, im_ind, gt_T, gt_S, iter, gt_inds_T, gt_inds_S):
#         iou_loss = dice_coefficient(preds_T, gt_T)
#         # choose the best teacher mask corresponding to a gt
#         nms_preds_T_inds, nms_gt_inds_T = nms(iou_loss, gt_inds_T)
#         nms_preds_T = preds_T[nms_preds_T_inds]
#         # match the teacher mask and the student mask according to the gt of this student mask
#         preds_T, preds_S, gt_S = self.match(nms_preds_T, preds_S, nms_gt_inds_T,
#                                             gt_inds_S, gt_S)
#         if self.visualization and iter % 1000 == 0 and get_rank() == 0:
#             visualization(im_ind, f'dice_cwd_iter{iter}', preds_T, preds_S, gt_S)
#         loss = self.loss_weight * self.cal_loss(preds_T, preds_S, gt_S)
#         return loss
#
#
# @loss_register('cwd_dice_weighted', LOSS_DICT)
# class ChannelWiseDivergence_dice_weighted_tea(nn.Module):
#     def __init__(
#         self,
#         tau=1.0,
#         loss_weight=1.0,
#         weight_sum=True
#     ):
#         super(ChannelWiseDivergence_dice_weighted_tea, self).__init__()
#         self.tau = tau
#         self.loss_weight = loss_weight
#         self.visualization = True
#         self.weight_sum = weight_sum
#
#     def match(self, nms_preds_T, preds_S, nms_gt_inds_T, gt_inds_S, gt_S):
#         t_inds = []
#         s_inds = []
#         for gt_ind in gt_inds_S:
#             # 当前stu输出对应的gt也有对应的tea的mask，由于nms_gt_inds_T不重复，
#             # 所以arg要么是空，即tea没有对应的mask，要么是有一个值即第几个tea mask
#             arg = torch.where(nms_gt_inds_T==gt_ind)[0]
#             if arg.size(0):
#                 s_inds.append(arg.size(0) == 1)
#                 t_inds.append(arg)
#         t_inds = torch.tensor(t_inds, dtype=torch.long, device=nms_preds_T.device)
#         s_inds = torch.tensor(s_inds, dtype=torch.bool, device=preds_S.device)
#         return nms_preds_T[t_inds], preds_S[s_inds], gt_S[s_inds]
#
#     def cal_loss(self, preds_T, preds_S, gts_S):
#         gts_S = gts_S.bool().detach()
#         loss = 0.
#         for pred_T, pred_S, gt_S in zip(preds_T, preds_S, gts_S):
#             pred_T, pred_S = pred_T[gt_S], pred_S[gt_S]
#             assert len(pred_T.shape) == 1 and len(pred_S.shape) == 1
#             softmax_pred_T = F.softmax(pred_T / self.tau)
#             logsoftmax = torch.nn.LogSoftmax(dim=0)
#             loss += torch.sum(softmax_pred_T * logsoftmax(pred_T / self.tau) -
#                               softmax_pred_T * logsoftmax(pred_S / self.tau)) * (self.tau ** 2)
#         return loss
#
#     def forward(self, preds_T, preds_S, im_ind, gt_T, gt_S, iter, gt_inds_T, gt_inds_S):
#         iou_loss = dice_coefficient(preds_T, gt_T)
#         # get weighted teacher mask corresponding to a gt
#         if self.weight_sum:
#             nms_preds_T, nms_gt_inds_T = weighted_sum_teacher(preds_T, iou_loss, gt_inds_T)
#         else:
#             nms_preds_T, nms_gt_inds_T = weighted_choice_teacher(preds_T, iou_loss,
#                                                               gt_inds_T)
#         # match the teacher mask and the student mask according to the gt of this student mask
#         preds_T, preds_S, gt_S = self.match(nms_preds_T, preds_S, nms_gt_inds_T,
#                                             gt_inds_S, gt_S)
#         if self.visualization and iter % 100 == 0 and get_rank() == 0:
#             visualization(im_ind, f'dice_cwd_iter{iter}', preds_T, preds_S, gt_S)
#         loss = self.loss_weight * self.cal_loss(preds_T, preds_S, gt_S)
#         return loss
#
#
# @loss_register('cwd_roi', LOSS_DICT)
# class ChannelWiseDivergence_roi(ChannelWiseDivergence):
#     def __init__(
#         self,
#         tau=1.0,
#         loss_weight=1.0
#     ):
#         super(ChannelWiseDivergence_roi, self).__init__(tau, loss_weight)
#         self.visualization = True
#
#     def cal_loss(self, preds_T, preds_S, gts_S):
#         gts_S = gts_S.bool().detach()
#         loss = 0.
#         for pred_T, pred_S, gt_S in zip(preds_T, preds_S, gts_S):
#             arg = torch.where(gt_S == True)
#             try:
#                 x_min, x_max, y_min, y_max = arg[0].min(), arg[0].max(), arg[
#                     1].min(), arg[1].max()
#             except:
#                 continue
#             assert len(pred_T.shape) == 2 and len(pred_S.shape) == 2
#             pred_T, pred_S = pred_T[x_min:x_max+1, y_min:y_max+1], pred_S[x_min:x_max+1, y_min:y_max+1]
#             pred_T, pred_S = pred_T.flatten(), pred_S.flatten()
#             softmax_pred_T = F.softmax(pred_T / self.tau)
#             logsoftmax = torch.nn.LogSoftmax(dim=0)
#             loss += torch.sum(softmax_pred_T * logsoftmax(pred_T / self.tau) -
#                               softmax_pred_T * logsoftmax(pred_S / self.tau)) * (self.tau ** 2)
#         return loss
#
#     def forward(self, preds_T, preds_S, im_ind, gt_T, gt_S, iter, gt_inds_T, gt_inds_S):
#         iou_loss = dice_coefficient(preds_T, gt_T)
#         # get weighted teacher mask corresponding to a gt
#         nms_preds_T, nms_gt_inds_T = weighted_sum_teacher(preds_T, iou_loss, gt_inds_T)
#         # match the teacher mask and the student mask according to the gt of this student mask
#         preds_T, preds_S, gt_S = self.match(nms_preds_T, preds_S, nms_gt_inds_T,
#                                             gt_inds_S, gt_S)
#         if self.visualization and iter % 5000 == 0 and get_rank() == 0:
#             visualization(im_ind, f'dice_cwd_iter{iter}', preds_T, preds_S, gt_S)
#         loss = self.loss_weight * self.cal_loss(preds_T, preds_S, gt_S)
#         return loss
#
#
# @loss_register('cwd_all_insts', LOSS_DICT)
# class ChannelWiseDivergence_all_insts(ChannelWiseDivergence):
#     def __init__(
#         self,
#         tau=1.0,
#         loss_weight=1.0
#     ):
#         super(ChannelWiseDivergence_all_insts, self).__init__(tau, loss_weight)
#         self.visualization = True
#
#     def cal_loss(self, preds_T, preds_S, gts_S):
#         gts_S = gts_S.bool().detach()
#         # 对gts_S的每个mask按位取或
#         gt_S = torch.bitwise_or(*gts_S)
#         loss = 0.
#         for pred_T, pred_S in zip(preds_T, preds_S):
#             pred_T, pred_S = pred_T[gt_S], pred_S[gt_S]
#             assert len(pred_T.shape) == 1 and len(pred_S.shape) == 1
#             softmax_pred_T = F.softmax(pred_T / self.tau)
#             logsoftmax = torch.nn.LogSoftmax(dim=0)
#             loss += torch.sum(softmax_pred_T * logsoftmax(pred_T / self.tau) -
#                               softmax_pred_T * logsoftmax(pred_S / self.tau)) * (self.tau ** 2)
#         return loss
#
#     def forward(self, preds_T, preds_S, im_ind, gt_T, gt_S, iter, gt_inds_T, gt_inds_S):
#         iou_loss = dice_coefficient(preds_T, gt_T)
#         # get weighted teacher mask corresponding to a gt
#         nms_preds_T, nms_gt_inds_T = weighted_sum_teacher(preds_T, iou_loss, gt_inds_T)
#         # match the teacher mask and the student mask according to the gt of this student mask
#         preds_T, preds_S, gt_S = self.match(nms_preds_T, preds_S, nms_gt_inds_T,
#                                             gt_inds_S, gt_S)
#         if self.visualization and iter % 5000 == 0 and get_rank() == 0:
#             visualization(im_ind, f'dice_cwd_iter{iter}', preds_T, preds_S, gt_S)
#         loss = self.loss_weight * self.cal_loss(preds_T, preds_S, gt_S)
#         return loss


@META_ARCH_REGISTRY.register()
class CondInst_cwd(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.teacher_trainable = False
        self._iter = 0

        # cfg.defrost()
        # cfg.MODEL.RESNETS.DEPTH = 101
        # cfg.MODEL.CONDINST.MAX_PROPOSALS = -1
        # cfg.MODEL.CONDINST.TOPK_PROPOSALS_PER_IM = 64
        # cfg.freeze()
        self.student = CondInst(cfg)
        tea_cfg = copy.deepcopy(cfg)
        tea_cfg.defrost()
        tea_cfg.MODEL.RESNETS.DEPTH = 101
        tea_cfg.MODEL.CONDINST.MAX_PROPOSALS = -1
        tea_cfg.MODEL.CONDINST.TOPK_PROPOSALS_PER_IM = 64
        tea_cfg.freeze()
        self.teacher = CondInst(tea_cfg)

        # todo model.train的时候会把teacher也改为train mode，weight不会变，但bn的track会受影响
        if self.teacher_trainable:
            self.teacher.train()
        else:
            self.teacher.eval()

        self.components = [
            dict(
                student_module='mask_head.identity_mask',
                teacher_module='mask_head.identity_mask',
                losses=[
                    dict(
                        type=cfg.MODEL.CONDINST.DISTILL_LOSS,
                        name='loss_cwd',
                        tau=1,
                        loss_weight=cfg.MODEL.CONDINST.DISTILL_LOSS_WEIGHT,
                    ),
                ])
        ]
        self.distill_losses = nn.ModuleDict()
        self.align_modules = nn.ModuleDict()

        for i, component in enumerate(self.components):
            align_module_cfg = getattr(component, 'align_module', None)
            if align_module_cfg is not None:
                align_module_name = f'component_{i}'
                align_module = self.build_align_module(align_module_cfg)
                self.align_modules[align_module_name] = align_module

            for loss in component['losses']:
                loss_cfg = loss.copy()
                loss_name = loss_cfg.pop('name')
                self.distill_losses[loss_name] = build_loss(loss_cfg)

        self.student_outputs = {}
        self.teacher_outputs = {}
        self.prepare_from_student()

    def prepare_from_student(self):
        self.student_module2name = {}
        for name, module in self.student.named_modules():
            self.student_module2name[module] = name
        self.student_name2module = dict(self.student.named_modules())

        self.teacher_module2name = {}
        for name, module in self.teacher.named_modules():
            self.teacher_module2name[module] = name
        self.teacher_name2module = dict(self.teacher.named_modules())

        for component in self.components:
            student_module_name = component['student_module']
            teacher_module_name = component['teacher_module']

            student_module = self.student_name2module[student_module_name]
            teacher_module = self.teacher_name2module[teacher_module_name]

            student_module.register_forward_hook(
                self.student_forward_output_hook)
            teacher_module.register_forward_hook(
                self.teacher_forward_output_hook)

        module_name_list = ['mask_head.identity_im_inds', 'mask_head.identity_gt_mask', 'mask_head.identity_gt_inds']
        for module_name in module_name_list:
            student_module = self.student_name2module[module_name]
            teacher_module = self.teacher_name2module[module_name]

            student_module.register_forward_hook(
                self.student_forward_output_hook)
            teacher_module.register_forward_hook(
                self.teacher_forward_output_hook)

    def teacher_forward_output_hook(self, module, inputs, outputs):
        self.teacher_outputs[self.teacher_module2name[module]] = outputs

    def student_forward_output_hook(self, module, inputs, outputs):
        self.student_outputs[self.student_module2name[module]] = outputs

    def exec_teacher_forward(self, data):
        if self.teacher_trainable:
            output = self.teacher(data)
        else:
            with torch.no_grad():
                output = self.teacher(data)
        return output

    def exec_student_forward(self, data):
        output = self.student(data)
        return output

    def get_im_inds(self):
        return (self.teacher_outputs['mask_head.identity_im_inds'],
                self.student_outputs['mask_head.identity_im_inds'])

    def get_mask_gt(self):
        return (self.teacher_outputs['mask_head.identity_gt_mask'],
                self.student_outputs['mask_head.identity_gt_mask'])

    def get_gt_inds(self):
        return (self.teacher_outputs['mask_head.identity_gt_inds'],
                self.student_outputs['mask_head.identity_gt_inds'])

    def forward(self, batched_inputs):
        if not self.training:
            return self.student(batched_inputs)
        self.exec_teacher_forward(batched_inputs)
        losses = self.exec_student_forward(batched_inputs)
        self._iter += 1
        tea_im_inds, stu_im_inds = self.get_im_inds()
        tea_mask_gt, stu_mask_gt = self.get_mask_gt()
        tea_gt_inds, stu_gt_inds = self.get_gt_inds()
        # if get_rank() == 0:
        #     print(tea_gt_inds, stu_gt_inds)

        assert all(tea_im_inds.unique() == stu_im_inds.unique())
        im_inds = tea_im_inds.unique()
        # print('im_inds: ', im_inds)

        tea_mask_gt = tea_mask_gt.squeeze(1)
        stu_mask_gt = stu_mask_gt.squeeze(1)

        for i, component in enumerate(self.components):
            student_module_name = component['student_module']
            student_output = self.student_outputs[student_module_name]
            assert student_output.size(1) == 1
            student_output = student_output.squeeze(1)

            align_module_name = f'component_{i}'
            if align_module_name in self.align_modules:
                align_module = self.align_modules[align_module_name]
                student_output = align_module(student_output)

            teacher_module_name = component['teacher_module']
            teacher_output = self.teacher_outputs[teacher_module_name]
            assert teacher_output.size(1) == 1
            teacher_output = teacher_output.squeeze(1)

            for loss in component['losses']:
                loss_name = loss['name']
                loss_func = self.distill_losses[loss_name]
                loss = 0.
                # match_num = 0
                for im_ind in im_inds:
                    stu_mask_gt_per_im = stu_mask_gt[stu_im_inds == im_ind]
                    tea_mask_gt_per_im = tea_mask_gt[tea_im_inds == im_ind]
                    loss_per_im= loss_func(teacher_output[tea_im_inds==im_ind],
                                                          student_output[stu_im_inds==im_ind],
                                                          im_ind,
                                                          tea_mask_gt_per_im,
                                                          stu_mask_gt_per_im,
                                                          self._iter,
                                                          tea_gt_inds[tea_im_inds==im_ind],
                                                          stu_gt_inds[stu_im_inds==im_ind])
                    loss += loss_per_im
                    # match_num += match_per_im
                losses[loss_name] = loss / stu_mask_gt.size(0)
                # assert False
                # if match_num == 0:
                #     assert loss == 0
                # losses[loss_name] = loss / match_num if match_num else torch.zeros((), device=student_output.device)
        # input()

        return losses
