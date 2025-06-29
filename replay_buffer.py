import numpy as np
import random
import torch

class PrioritizedReplayBuffer:
    def __init__(self, capacity=100000, alpha=0.6, beta_start=0.4, beta_frames=100000):
        """
        初始化优先经验池。

        参数:
        - capacity (int): 经验池的最大容量。
        - alpha (float): 优先级控制参数 (即您提到的温度参数)。控制采样的随机性，
                         alpha=0表示均匀采样, alpha=1表示完全按优先级采样。
        - beta_start (float): 重要性采样权重的初始值。
        - beta_frames (int): beta值从beta_start线性增长到1.0所需的训练步数。
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1  # 跟踪训练步数以更新beta

        # 为了高效查找，我们使用一个基于“和树”的数据结构
        # 它允许我们在O(log n)时间内完成加权采样
        self.priorities = np.zeros((capacity,), dtype=np.float64)
        self.buffer = [None] * capacity
        
        self.pos = 0
        self.full = False

        # 和树相关的参数
        self.tree_start = 1
        while self.tree_start < self.capacity:
            self.tree_start *= 2
        self.sum_tree = np.zeros((self.tree_start * 2,), dtype=np.float64)

        self.max_priority = 1.0

    def _propagate(self, idx, change):
        """在和树上传播优先级的变化"""
        parent = idx // 2
        self.sum_tree[parent] += change
        if parent != 1:
            self._propagate(parent, change)

    def _retrieve(self, value, node_idx=1):
        """根据一个值在和树中找到对应的叶子节点索引"""
        left_child = node_idx * 2
        right_child = left_child + 1

        if left_child >= len(self.sum_tree):
            return node_idx

        if value <= self.sum_tree[left_child]:
            return self._retrieve(value, left_child)
        else:
            return self._retrieve(value - self.sum_tree[left_child], right_child)

    def add(self, state, policy, value):
        """向经验池中添加一个新的经验，并赋予其最大优先级以确保它至少被训练一次"""
        idx = self.pos
        self.buffer[idx] = (state, policy, value)
        
        # 使用当前记录的最大优先级
        priority = self.max_priority
        tree_idx = idx + self.tree_start
        
        # 更新和树
        change = priority - self.sum_tree[tree_idx]
        self.sum_tree[tree_idx] = priority
        self._propagate(tree_idx, change)
        
        self.pos = (self.pos + 1) % self.capacity
        if self.pos == 0:
            self.full = True

    def sample(self, batch_size):
        """
        从经验池中采样一批数据，同时返回数据、其在缓冲区中的索引以及重要性采样权重。
        """
        indices = []
        weights = np.zeros((batch_size,), dtype=np.float32)
        
        total_priority = self.sum_tree[1]
        segment = total_priority / batch_size

        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        self.frame += 1
        
        current_size = self.capacity if self.full else self.pos

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            
            tree_idx = self._retrieve(s)
            data_idx = tree_idx - self.tree_start
            
            indices.append(data_idx)
            
            # 计算重要性采样权重
            sampling_prob = self.sum_tree[tree_idx] / total_priority
            weights[i] = (current_size * sampling_prob) ** -beta

        # 归一化权重
        weights /= weights.max()
        
        batch = [self.buffer[idx] for idx in indices]
        states, policies, values = zip(*batch)

        return (
            list(states), 
            list(policies), 
            list(values), 
            indices, 
            torch.tensor(weights, dtype=torch.float32)
        )

    def update_priorities(self, batch_indices, td_errors, epsilon=1e-6):
        """
        在一次训练后，根据计算出的TD-Error更新对应样本的优先级。

        参数:
        - batch_indices (list): 采样出的一批数据在缓冲区中的原始索引。
        - td_errors (np.array): 对应每个样本的TD-Error绝对值。
        - epsilon (float): 一个很小的正数，防止优先级为0。
        """
        priorities = (np.abs(td_errors) + epsilon) ** self.alpha
        
        for idx, priority in zip(batch_indices, priorities):
            tree_idx = idx + self.tree_start
            
            # 更新和树
            change = priority - self.sum_tree[tree_idx]
            self.sum_tree[tree_idx] = priority
            self._propagate(tree_idx, change)

        # 更新全局最大优先级
        self.max_priority = max(self.max_priority, priorities.max())
        
    def __len__(self):
        return self.capacity if self.full else self.pos