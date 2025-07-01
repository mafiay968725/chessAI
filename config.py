# file: config.py

args = {
    'C': 2,
    'num_searches': 800,
    'num_iterations': 20,
    'num_selfPlay_episodes': 200,
    'num_cpu_threads': 18,
    'avg_new_sample_passes': 4.0,
    'batch_size': 256,
    'learning_rate': 0.001,
    'data_max_size': 100000,
    'num_res_blocks': 20,
    'num_hidden': 256,
    'dirichlet_alpha': 0.5, #暂时从0.3调到0.5
    'dirichlet_epsilon': 0.25,
    'value_loss_weight': 1,
    'elo_k_factor': 32,
    'num_candidates_to_train': 3, # 每轮自对弈后，尝试训练3个候选模型

    # 温度采样参数python
    'temperature_start': 1.0,
    'temperature_end': 0.1,
    'temperature_decay_moves': 15, #暂时从10拉到15

    # 评估体系参数
    'num_eval_games': 30,
    'board_size': 9,
    'num_rounds': 25, # 每局游戏的回合数，总步数为 num_rounds * 2
    'mcts_batch_size': 256,

    # =======================================================
    # --- 请确保您的配置包含以下所有参数 ---

    # 数据筛选开关
    'filter_zero_policy_data': True,  # True 表示开启筛选，False 表示关闭

    # 开局偏置参数
    'enable_opening_bias': True,  # True 表示启用开局偏置
    'opening_bias_strength': 0.1,  # 偏置的总体强度

    # 无效连接惩罚参数
    'enable_ineffective_connection_penalty': True,      # True 表示启用该惩罚
    'ineffective_connection_penalty_factor': 0.1,       # 惩罚系数(将无收益三连的策略概率乘以该系数)

    # 领地价值启发参数
    'enable_territory_heuristic': True,  # [之前缺失] True 表示启用领地启发
    'territory_heuristic_weight': 0.6,  # 领地启发所占的权重

    # 领地维持惩罚参数 (新功能)
    'enable_territory_penalty': True,      # True 表示启用该惩罚
    'territory_penalty_strength': 0.5,   # 惩罚强度(一个负向的偏置)

    # 防守奖励参数
    'enable_opponent_territory_threat_bonus': True, # 新的功能开关
    'opponent_territory_threat_bonus_strength': 0.8, # 奖励强度。1.5代表防守收益最高的点，其策略概率最多可提升150%
    
     # 优先经验池回放 (PER) 相关参数
    'enable_per': True,                   # True 表示启用PER, False则退化为标准经验池
    'per_alpha': 0.6,                     # 优先级指数，0=纯随机, 1=纯优先级
    'per_beta_start': 0.4,                # 重要性采样权重的初始值，会随训练线性增长到1.0
    'per_beta_frames': 1000000,           # beta增长到1.0所需的总训练steps
    'per_epsilon': 1e-6,                  # 一个小常数，防止任何经验的优先级为0
    
    # =======================================================
    'history_steps': 3, # 记录T-1, T-2, T-3三步历史
}
