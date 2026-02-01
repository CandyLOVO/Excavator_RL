# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # num_steps_per_env = 24
    num_steps_per_env = 32 # 增加每个环境的步数以提高样本多样性
    # max_iterations = 300
    max_iterations = 500 # 增加最大迭代次数以提升训练效果
    save_interval = 50
    experiment_name = "excavator_ppo"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        # actor_hidden_dims=[64, 64],
        # critic_hidden_dims=[64, 64],
        actor_hidden_dims=[128, 128],
        critic_hidden_dims=[128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,  # 增加探索
        num_learning_epochs=5,
        # num_mini_batches=4,
        num_mini_batches=8, # 增加小批量数量以提高样本利用率
        learning_rate=1.0e-4,  # 降低学习率以提高稳定性
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )