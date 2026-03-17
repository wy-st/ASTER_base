# =============================================================================
# utils/env.py  ——  资源分配环境（来自 ASTER）
# =============================================================================
# ResourceEnv 追踪每个节点的资源占有情况和冷却时间。
# 当资源被调度到某节点时，该节点进入 k_steps 步的冷却期，
# 冷却期内不能再次接受调度。
# 用匈牙利算法计算最优调度分配，最小化总调度距离。
# =============================================================================

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


class ResourceEnv:
    """
    资源调度环境。

    状态：
        resources[N] : 每个节点当前持有的资源数（0 或 1）
        cooldowns[N] : 每个节点剩余的冷却步数

    可调度：resources[n]==1 且 cooldowns[n]==0 的节点
    """

    def __init__(self, num_nodes, total_resources, k_steps):
        """
        num_nodes       : 节点总数 N
        total_resources : 总资源数量
        k_steps         : 调度后冷却步数
        """
        self.N               = num_nodes
        self.total_resources = total_resources
        self.k_steps         = k_steps
        self.resources       = np.zeros(num_nodes, dtype=np.int32)
        self.cooldowns       = np.zeros(num_nodes, dtype=np.int32)
        self.reset()

    # ------------------------------------------------------------------
    def reset(self):
        """随机重置资源分布，清空冷却。"""
        self.resources = np.zeros(self.N, dtype=np.int32)
        self.cooldowns = np.zeros(self.N, dtype=np.int32)
        chosen = np.random.choice(self.N, self.total_resources, replace=False)
        self.resources[chosen] = 1

    # ------------------------------------------------------------------
    def step(self, actions, distance_matrix):
        """
        推进一步：
          1. 所有冷却时间 -1
          2. 用匈牙利算法把资源从空闲节点移到请求节点
          3. 被调度的目标节点冷却计时器置为 k_steps

        Args:
            actions        : list 或 [N] array，值在 {0, 1}
            distance_matrix: [N, N] numpy 数组，节点间距离

        Returns:
            new_available : int，调度后可用资源数
            total_cost    : float，本次调度总距离
        """
        self.cooldowns = np.maximum(0, self.cooldowns - 1)

        total_cost, (src_nodes, tgt_nodes) = self._hungarian_assign(
            actions, distance_matrix
        )

        for s, t in zip(src_nodes, tgt_nodes):
            self.resources[s] -= 1
            self.resources[t] += 1
            self.cooldowns[t]  = self.k_steps

        new_available = int(
            np.sum((self.resources == 1) & (self.cooldowns == 0))
        )
        return new_available, total_cost

    # ------------------------------------------------------------------
    def get_state(self):
        """返回 [N, 2] tensor：每行是 (resources_n, cooldowns_n)。"""
        return torch.tensor(
            np.stack([self.resources, self.cooldowns], axis=-1),
            dtype=torch.float32,
        )

    # ------------------------------------------------------------------
    def _hungarian_assign(self, actions, distance_matrix):
        """
        匈牙利算法：把空闲资源节点分配给请求调度的节点。

        供给节点（supply）：resources==1 且 cooldowns==0
        需求节点（demand）：actions==1

        Returns: (total_cost, (src_indices, tgt_indices))
        """
        supply = np.where((self.resources == 1) & (self.cooldowns == 0))[0]
        demand = np.where(np.array(actions) == 1)[0]

        if len(supply) == 0 or len(demand) == 0:
            return 0.0, (np.array([], dtype=int), np.array([], dtype=int))

        cost   = distance_matrix[np.ix_(supply, demand)]
        r, c   = linear_sum_assignment(cost)
        return float(cost[r, c].sum()), (supply[r], demand[c])
