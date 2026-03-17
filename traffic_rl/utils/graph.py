# =============================================================================
# utils/graph.py  ——  距离矩阵与位置张量的构建
# =============================================================================
# ConFormer 的三个数据集（METRLA / PEMSBAY / TKY）没有附带经纬度坐标文件，
# 因此用合成的二维网格布局来近似地理位置，计算节点间欧氏距离。
#
# 如果你有真实的 lat/lon 文件，可以在此替换 make_distance_matrix 的实现。
# =============================================================================

import math
import numpy as np
import torch


def make_distance_matrix(num_nodes):
    """
    构建合成距离矩阵（节点排布在二维整数网格上）。

    把 num_nodes 个节点按行优先排在 side × side 的网格上，
    用节点间的欧氏距离作为调度成本。单位：网格格子数（1 格 ≈ 1 km）。

    Args:
        num_nodes : 节点总数 N

    Returns:
        dist : [N, N] numpy float32 数组
    """
    side   = int(math.ceil(math.sqrt(num_nodes)))
    coords = np.array(
        [(i // side, i % side) for i in range(num_nodes)],
        dtype=np.float32,
    )
    # 广播计算两两欧氏距离
    diff = coords[:, None, :] - coords[None, :, :]   # [N, N, 2]
    dist = np.sqrt((diff ** 2).sum(-1)).astype(np.float32)  # [N, N]
    return dist


def make_location_tensor(num_nodes, device):
    """
    构建归一化 (x, y) 位置张量，用作 DQN 状态的一部分。

    坐标归一化到 [0, 1]，给智能体提供地理位置信息，
    使其在调度时可以隐式感知节点间的相对距离。

    Args:
        num_nodes : 节点总数 N
        device    : "cpu" 或 "cuda"

    Returns:
        location : [N, 2] float tensor
    """
    side   = int(math.ceil(math.sqrt(num_nodes)))
    coords = np.array(
        [(i // side, i % side) for i in range(num_nodes)],
        dtype=np.float32,
    )
    coords = coords / side   # 归一化到 [0, 1]
    return torch.FloatTensor(coords).to(device)   # [N, 2]
