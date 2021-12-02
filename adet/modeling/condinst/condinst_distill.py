# -*- coding: utf-8 -*-
import logging
from skimage import color
import copy
from scipy.optimize import linear_sum_assignment

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

    def forward(self, preds_S, preds_T):
        """Forward function."""
        assert preds_S.shape[-2:] == preds_T.shape[-2:]
        N, C, W, H = preds_S.shape

        softmax_pred_T = F.softmax(preds_T.reshape(-1, W * H) / self.tau, dim=1)

        logsoftmax = torch.nn.LogSoftmax(dim=1)
        loss = torch.sum(softmax_pred_T *
                         logsoftmax(preds_T.reshape(-1, W * H) / self.tau) -
                         softmax_pred_T *
                         logsoftmax(preds_S.reshape(-1, W * H) / self.tau)) * (
                             self.tau**2)

        loss = self.loss_weight * loss / (C * N)

        return loss


@loss_register('cwd_match', LOSS_DICT)
class ChannelWiseDivergence_match(nn.Module):

    def __init__(self, tau=1.0, loss_weight=1.0):
        super(ChannelWiseDivergence_match, self).__init__()
        self.tau = tau
        self.loss_weight = loss_weight

    def cal_loss(self, channel_T, channel_S):
        N, C, W, H = channel_S.shape
        softmax_pred_T = F.softmax(channel_T.reshape(-1, W * H) / self.tau, dim=1)

        logsoftmax = torch.nn.LogSoftmax(dim=1)
        loss = torch.sum(softmax_pred_T *
                         logsoftmax(channel_T.reshape(-1, W * H) / self.tau) -
                         softmax_pred_T *
                         logsoftmax(channel_S.reshape(-1, W * H) / self.tau)) * (
                             self.tau**2)

        loss = loss / (N * C)
        return loss

    def compute_kl_matrix(self, logits_p, logits_q):
        N, C, W, H = logits_p.shape
        logits_p, logits_q = logits_p.detach(), logits_q.detach()
        prob_p = F.softmax(logits_p.reshape(N, C, W * H) / self.tau, dim=-1)

        log_prob_p = F.log_softmax(
            logits_p.reshape(N, C, W * H) / self.tau, dim=-1)
        log_prob_q = F.log_softmax(
            logits_q.reshape(N, C, W * H) / self.tau, dim=-1)

        cross_entropy_matrix = -torch.bmm(log_prob_q, prob_p.transpose(1, 2))
        entropy_p = -(prob_p * log_prob_p).sum(-1, keepdim=True)

        kl_matrix = cross_entropy_matrix - entropy_p

        return kl_matrix

    def forward(self, preds_S, preds_T):
        """Forward function."""
        assert preds_S.shape[-2:] == preds_T.shape[-2:]

        kl_matrixs = self.compute_kl_matrix(preds_T, preds_S)
        loss = 0.
        for i, matrix in enumerate(kl_matrixs):
            try:
                t_ind, s_ind = linear_sum_assignment(matrix.cpu())
            except:
                if get_rank() == 0:
                    print('*' * 40)
                    print(preds_S)
                    print(preds_T)
                raise RuntimeError
            t_ind = torch.tensor(t_ind, device=preds_S.device)
            s_ind = torch.tensor(s_ind, device=preds_S.device)
            loss += self.loss_weight * self.cal_loss(preds_T[i:i + 1, t_ind],
                                                     preds_S[i:i + 1, s_ind])

        return loss / len(kl_matrixs)


@META_ARCH_REGISTRY.register()
class CondInst_distill(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.teacher_trainable = False
        self._iter = 0

        self.student = CondInst(cfg)
        tea_cfg = copy.deepcopy(cfg)
        tea_cfg.defrost()
        tea_cfg.MODEL.RESNETS.DEPTH = 101
        tea_cfg.freeze()
        self.teacher = CondInst(tea_cfg)
        tea_pretrain_path = "./ckpt/CondInst_MS_R_101_3x.pth"
        ckpt = torch.load(tea_pretrain_path, map_location=cfg.MODEL.DEVICE)
        self.teacher.load_state_dict(ckpt, strict=False)
        print('Successfully load teacher checkpoint.')
        if self.teacher_trainable:
            self.teacher.train()
        else:
            self.teacher.eval()

        self.components = [
            dict(
                student_module='mask_head.identity',
                teacher_module='mask_head.identity',
                losses=[
                    dict(
                        type='cwd_match',
                        name='loss_cwd',
                        tau=1,
                        loss_weight=1,
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

    def forward(self, batched_inputs):
        losses = self.exec_student_forward(batched_inputs)
        self.exec_teacher_forward(batched_inputs)
        self._iter += 1
        # if self._iter % 20 == 0:
        #     print(self.teacher_outputs)
        #     print(self.student_outputs)
        for i, component in enumerate(self.components):

            student_module_name = component['student_module']
            student_output = self.student_outputs[student_module_name]
            align_module_name = f'component_{i}'
            if align_module_name in self.align_modules:
                align_module = self.align_modules[align_module_name]
                student_output = align_module(student_output)

            teacher_module_name = component['teacher_module']
            teacher_output = self.teacher_outputs[teacher_module_name]

            for loss in component['losses']:
                loss_name = loss['name']
                loss_func = self.distill_losses[loss_name]
                losses[loss_name] = loss_func(student_output, teacher_output)

        return losses
