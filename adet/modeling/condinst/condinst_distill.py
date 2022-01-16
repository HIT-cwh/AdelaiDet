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

__all__ = ["CondInst_distill"]

logger = logging.getLogger(__name__)

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


@loss_register('dice', LOSS_DICT)
class DiceCoefficient(nn.Module):
    def __init__(
        self,
        tau=1.0,
        loss_weight=1.0,
    ):
        super(DiceCoefficient, self).__init__()
        self.tau = tau
        self.loss_weight = loss_weight
        self.visualization = True

    def match(self, nms_preds_T, preds_S, nms_gt_inds_T, gt_inds_S, gt_S):
        t_inds = []
        s_inds = []
        for gt_ind in gt_inds_S:
            arg = torch.where(nms_gt_inds_T==gt_ind)[0]
            s_inds.append(arg.size(0) == 1)
            if arg.size(0):
                t_inds.append(arg)
        t_inds = torch.tensor(t_inds, dtype=torch.long, device=nms_preds_T.device)
        s_inds = torch.tensor(s_inds, dtype=torch.bool, device=preds_S.device)
        return nms_preds_T[t_inds], preds_S[s_inds], gt_S[s_inds]

    def forward(self, preds_T, preds_S, im_ind, gt_T, gt_S, iter, gt_inds_T, gt_inds_S):
        iou_loss = dice_coefficient(preds_T, gt_T)
        nms_preds_T_inds, nms_gt_inds_T = nms(iou_loss, gt_inds_T)
        nms_preds_T = preds_T[nms_preds_T_inds]
        preds_T, preds_S, gt_S = self.match(nms_preds_T, preds_S, nms_gt_inds_T,
                                            gt_inds_S, gt_S)
        if self.visualization and iter % 1000 == 0 and get_rank() == 0:
            visualization(im_ind, f'dice_visualization_iter{iter}', preds_T, preds_S, gt_S)
        loss = self.loss_weight * dice_coefficient(preds_S, preds_T).sum()
        return loss


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


def visualization_single(preds_S, im_ind, gt_ind, root='gt2'):
    preds_S = preds_S[0]
    preds_S = preds_S.sigmoid() * 255
    preds_S = preds_S.int().cpu().numpy()
    mmcv.mkdir_or_exist(root)
    # for i, img in enumerate(preds_S):
    cv2.imwrite(osp.join(root, f'img_{im_ind}_gt_{gt_ind}.png'), preds_S)


@META_ARCH_REGISTRY.register()
class CondInst_distill(nn.Module):
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

        if self.teacher_trainable:
            self.teacher.train()
        else:
            self.teacher.eval()

        self.components = [
            dict(
                student_module='mask_head.identity_mask',
                teacher_module='mask_head.identity_mask',
                losses=[
                    # dict(
                    #     type='cwd',
                    #     name='loss_cwd',
                    #     tau=1,
                    #     loss_weight=1,
                    # ),
                    dict(
                        type='dice',
                        name='mask_loss_dis',
                        tau=1,
                        loss_weight=2,
                    )
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