# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 48       # 每环境48步，收集更多时序信息（1024×48=49152 样本/iter）
    max_iterations = 1500        # 充分训练：1500 iter × 49152 ≈ 7370万样本
    save_interval = 100          # 每100轮保存，便于挑选最优模型
    experiment_name = "excavator_ppo"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 64],   # 三层网络：更强的表达能力处理253维观测
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,      # 适度探索，避免过早收敛
        num_learning_epochs=8,   # 更多梯度更新轮次，充分利用每批样本
        num_mini_batches=4,      # batch_size=12288，梯度估计更稳定
        learning_rate=3.0e-4,    # 标准PPO学习率，配合adaptive schedule自动调整
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )