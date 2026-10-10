"""Vendored ST-GCN++ backbone, PYSKL-compatible, torch-only.

PROVENANCE (do not remove): re-implemented from PYSKL at pinned commit
f2bf3a6b08e2e8dec744692d64efdb187fd6719a, primary source
https://github.com/kennymckormick/pyskl/tree/main/configs/stgcn%2B%2B
(MMAction2 configs/skeleton/stgcnpp as secondary reference only).
Upstream files reused (logic mirrored, mmcv/registry parts removed):
  pyskl/models/gcns/stgcn.py            -> STGCNBlock, STGCNBackbone
  pyskl/models/gcns/utils/gcn.py        -> UnitGCN (adaptive='init' path only)
  pyskl/models/gcns/utils/tcn.py        -> UnitTCN, MSTCN (default ms_cfg)
  pyskl/models/heads/simple_head.py     -> GCN-mode pooling + linear head
  pyskl/utils/graph.py                  -> coco_spatial_adjacency
(state_dict key names match upstream exactly so official checkpoints load.)

Why vendored: importing PYSKL pulls mmcv-full/mmpose/mmdet; VIGLANCE stays
lightweight (torch-only). Nothing else from PYSKL is duplicated.

VIGLANCE addition: ViglanceFightModel = frozen-capable backbone +
Linear(256, 2) head with classes 0=NO_FIGHT, 1=FIGHT. The 60-class NTU head
is never instantiated here.

Requires torch (imported at module top; this module itself is only ever
imported lazily by the adapter / tests, never at package import time).
"""

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Graph: PYSKL Graph(layout='coco', mode='spatial') -> A shape (3, 17, 17)
# ---------------------------------------------------------------------------

COCO_INWARD = (
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 5), (12, 6),
    (9, 7), (7, 5), (10, 8), (8, 6), (5, 0), (6, 0),
    (1, 0), (3, 1), (2, 0), (4, 2),
)
NUM_NODES = 17


def _edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


def _normalize_digraph(A, dim=0):
    Dl = np.sum(A, dim)
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    return A.dot(Dn)


def coco_spatial_adjacency():
    """(3, 17, 17) float32 stack (Iden, In, Out), == PYSKL spatial mode."""
    inward = list(COCO_INWARD)
    outward = [(j, i) for (i, j) in inward]
    self_link = [(i, i) for i in range(NUM_NODES)]
    Iden = _edge2mat(self_link, NUM_NODES)
    In = _normalize_digraph(_edge2mat(inward, NUM_NODES))
    Out = _normalize_digraph(_edge2mat(outward, NUM_NODES))
    return np.stack((Iden, In, Out)).astype(np.float32)


# ---------------------------------------------------------------------------
# TCN blocks (PYSKL defaults)
# ---------------------------------------------------------------------------

class UnitTCN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1,
                 dilation=1, norm='BN'):
        super().__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=(kernel_size, 1),
                              padding=(pad, 0),
                              stride=(stride, 1),
                              dilation=(dilation, 1))
        self.bn = nn.BatchNorm2d(out_channels) if norm is not None else nn.Identity()
        self.drop = nn.Dropout(0, inplace=True)

    def forward(self, x):
        return self.drop(self.bn(self.conv(x)))


class MSTCN(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1,
                 ms_cfg=((3, 1), (3, 2), (3, 3), (3, 4), ('max', 3), '1x1')):
        super().__init__()
        num_branches = len(ms_cfg)
        mid_channels = out_channels // num_branches
        rem_mid_channels = out_channels - mid_channels * (num_branches - 1)
        self.act = nn.ReLU()
        branches = []
        for i, cfg in enumerate(ms_cfg):
            branch_c = rem_mid_channels if i == 0 else mid_channels
            if cfg == '1x1':
                branches.append(nn.Conv2d(in_channels, branch_c, kernel_size=1,
                                          stride=(stride, 1)))
                continue
            if cfg[0] == 'max':
                branches.append(nn.Sequential(
                    nn.Conv2d(in_channels, branch_c, kernel_size=1),
                    nn.BatchNorm2d(branch_c), self.act,
                    nn.MaxPool2d(kernel_size=(cfg[1], 1), stride=(stride, 1),
                                 padding=(1, 0))))
                continue
            branches.append(nn.Sequential(
                nn.Conv2d(in_channels, branch_c, kernel_size=1),
                nn.BatchNorm2d(branch_c), self.act,
                UnitTCN(branch_c, branch_c, kernel_size=cfg[0],
                        stride=stride, dilation=cfg[1], norm=None)))
        self.branches = nn.ModuleList(branches)
        tin = mid_channels * (num_branches - 1) + rem_mid_channels
        self.transform = nn.Sequential(
            nn.BatchNorm2d(tin), self.act,
            nn.Conv2d(tin, out_channels, kernel_size=1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout(0., inplace=True)

    def forward(self, x):
        feat = torch.cat([b(x) for b in self.branches], dim=1)
        return self.drop(self.bn(self.transform(feat)))


# ---------------------------------------------------------------------------
# GCN block (adaptive='init', conv_pos='pre', with_res=True)
# ---------------------------------------------------------------------------

class UnitGCN(nn.Module):
    def __init__(self, in_channels, out_channels, A):
        super().__init__()
        self.num_subsets = A.size(0)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()
        self.A = nn.Parameter(A.clone())
        self.conv = nn.Conv2d(in_channels, out_channels * A.size(0), 1)
        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels))
        else:
            self.down = lambda x: x

    def forward(self, x):
        n, c, t, v = x.shape
        res = self.down(x)
        x = self.conv(x)
        x = x.view(n, self.num_subsets, -1, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, self.A)).contiguous()
        return self.act(self.bn(x) + res)


class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super().__init__()
        self.gcn = UnitGCN(in_channels, out_channels, A)
        self.tcn = MSTCN(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = UnitTCN(in_channels, out_channels, kernel_size=1,
                                    stride=stride)

    def forward(self, x):
        return self.relu(self.tcn(self.gcn(x)) + self.residual(x))


# ---------------------------------------------------------------------------
# Backbone: j.py config (in=3, base=64, VC, 10 stages, inflate/down [5, 8])
# ---------------------------------------------------------------------------

class STGCNBackbone(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, num_stages=10,
                 inflate_stages=(5, 8), down_stages=(5, 8)):
        super().__init__()
        A = torch.tensor(coco_spatial_adjacency(), dtype=torch.float32)
        self.data_bn = nn.BatchNorm1d(in_channels * NUM_NODES)
        modules = [STGCNBlock(in_channels, base_channels, A.clone(), 1,
                              residual=False)]
        inflate_times, out = 0, base_channels
        base = base_channels
        for i in range(2, num_stages + 1):
            stride = 1 + (i in down_stages)
            if i in inflate_stages:
                inflate_times += 1
            out = int(base_channels * 2 ** inflate_times + 1e-4)
            modules.append(STGCNBlock(base, out, A.clone(), stride))
            base = out
        self.gcn = nn.ModuleList(modules)
        self.num_stages = num_stages
        self.out_channels = out

    def forward(self, x):
        N, M, T, V, C = x.size()
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = self.data_bn(x.view(N * M, V * C, T))
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)
        for blk in self.gcn:
            x = blk(x)
        return x.reshape((N, M) + x.shape[1:])


# ---------------------------------------------------------------------------
# VIGLANCE fight model: backbone + 2-class head (0=NO_FIGHT, 1=FIGHT)
# ---------------------------------------------------------------------------

CLASS_NAMES = ("NO_FIGHT", "FIGHT")


class ViglanceFightModel(nn.Module):
    def __init__(self, num_classes=2, in_channels=256, dropout=0.0):
        super().__init__()
        self.backbone = STGCNBackbone()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.fight_head = nn.Linear(in_channels, num_classes)
        self.num_classes = num_classes

    def backbone_parameters(self):
        return list(self.backbone.parameters())

    def set_backbone_frozen(self, frozen=True):
        for p in self.backbone.parameters():
            p.requires_grad = not frozen

    def forward(self, x):
        feat = self.backbone(x)
        N, M, C, T, V = feat.shape
        x = feat.reshape(N * M, C, T, V)
        x = self.pool(x).reshape(N, M, C).mean(dim=1)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.fight_head(x)


def build_model(num_classes=2, freeze_backbone=True):
    model = ViglanceFightModel(num_classes=num_classes)
    model.set_backbone_frozen(freeze_backbone)
    return model
