from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from .excavator_ppo_env_cfg import ExcavatorPpoEnvCfg


class ExcavatorPpoEnv(DirectRLEnv):
    cfg: ExcavatorPpoEnvCfg

    #初始化，接收自身配置参数
    def __init__(self, cfg: ExcavatorPpoEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._body_dof_idx, _ = self.robot.find_joints(self.cfg.body_dof_name) #关节索引
        self._wheel_dof_idx, _ = self.robot.find_joints(self.cfg.wheel_dof_name)
        self.joint_pos = self.robot.data.joint_pos
        self.dof_pos_lower_limits = self.robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.dof_pos_upper_limits = self.robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        self.pos_actions = torch.zeros((self.num_envs, len(self._body_dof_idx)), device=self.device)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg) #机器人为Articulation类型，传入配置参数
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    #更新动作，得到动作张量的副本
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone() #避免修改原始动作张量，将获取数据与正在训练的张量分离
        body_vel_actions = torch.clamp(self.actions[:, :len(self._body_dof_idx)], -1.0, 1.0) #将actions解释为速度
        body_pos_actions = self.pos_actions + self.dt * body_vel_actions * self.cfg.position_action_scale #将actions解释为位置
        self.pos_actions = torch.clamp(body_pos_actions, self.dof_pos_lower_limits[self._body_dof_idx], self.dof_pos_upper_limits[self._body_dof_idx])
        self.vel_actions = self.actions[:, len(self._body_dof_idx):]

    #应用动作，更新的数据应用于物理模拟，为指定关节设置期望目标值
    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self.pos_actions, joint_ids=self._body_dof_idx) #设置目标位置
        self.robot.set_joint_velocity_target(self.vel_actions, joint_ids=self._wheel_dof_idx) #设置目标速度

    #获取观测
    def _get_observations(self) -> dict:
        self.robot_lin_vel = self.robot.data.root_com_lin_vel_b
        obs = torch.hstack((self.joint_pos[:, :len(self._body_dof_idx)], self.robot_lin_vel))
        observations = {"policy": obs}
        return observations

    #获取奖励，计算函数compute_rewards见最后
    def _get_rewards(self) -> torch.Tensor:
        total_reward = torch.linalg.norm(self.robot.data.root_com_lin_vel_b, dim=1)
        return total_reward

    #获取终止状态，返回是否越界和是否超时
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = torch.zeros_like(time_out, dtype=torch.bool)
        truncated = time_out.to(torch.bool)
        return terminated, truncated

    #重置环境
    def _reset_idx(self, env_ids: Sequence[int] | None): #env_ids表示要重置的环境索引
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids) #调用父类的重置方法

        #重置流程：获取默认初始状态 -> 调整位置到环境原点 -> 写入模拟器
        joint_pos = self.robot.data.default_joint_pos[env_ids] #获取默认关节位置
        self.joint_pos[env_ids] = joint_pos
        self.robot.write_joint_position_to_sim(joint_pos, None, env_ids)

        default_root_state = self.robot.data.default_root_state[env_ids] #获取默认根状态
        default_root_state[:, :3] += self.scene.env_origins[env_ids] #将根位置调整到各自环境的原点
        self.robot.write_root_state_to_sim(default_root_state, env_ids) #写入关节位置和速度

