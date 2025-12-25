from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import isaaclab.utils.math as math_utils

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

        #################创建指令向量（目标值）##################
        self.commands = torch.randn((self.cfg.scene.num_envs, 3)).to(device=self.device) #初始随机指令——世界坐标系
        self.commands[:, -1] = 0.0
        cmd_norm = torch.linalg.norm(self.commands, dim=1, keepdim=True).clamp_min(1e-6)
        self.commands = self.commands / cmd_norm
        #####################################################

        #####################创建可视化标记#####################
        self.visualization_markers = define_markers()
        self.marker_locations = torch.zeros((self.cfg.scene.num_envs, 3)).to(device=self.device) #标记位置
        self.marker_offset = torch.zeros((self.cfg.scene.num_envs, 3)).to(device=self.device) #标记偏移量
        self.marker_offset[:, -1] = 3.0 #标记在Z轴上方2米
        self.forward_marker_orientations = torch.zeros((self.cfg.scene.num_envs, 4)).to(device=self.device) #底盘朝向标记四元数
        self.command_marker_orientations = torch.zeros((self.cfg.scene.num_envs, 4)).to(device=self.device) #指令朝向标记四元数
        ######################################################

        self.yaws = torch.atan2(self.commands[:, 1], self.commands[:, 0]).unsqueeze(1) #command的偏航角，(-pi, pi]
        self.up_dir = torch.tensor([0.0, 0.0, 1.0]).to(device=self.device) #向量Z轴
   
    def _visualize_markers(self):
        # get marker locations and orientations
        self.marker_locations = self.robot.data.root_pos_w #机器人位置——世界坐标系
        self.forward_marker_orientations = self.robot.data.root_quat_w
        self.command_marker_orientations = math_utils.quat_from_angle_axis(self.yaws, self.up_dir).squeeze()

        # offset markers so they are above the jetbot
        loc = self.marker_locations + self.marker_offset
        loc = torch.vstack((loc, loc)) #两个标记的位置
        rots = torch.vstack((self.forward_marker_orientations, self.command_marker_orientations)) #两个标记的朝向

        # render the markers
        all_envs = torch.arange(self.cfg.scene.num_envs)
        indices = torch.hstack((torch.zeros_like(all_envs), torch.ones_like(all_envs))) #标记索引：0-前进方向，1-指令方向
        self.visualization_markers.visualize(loc, rots, marker_indices=indices)

    #更新动作，得到动作张量的副本
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone() #避免修改原始动作张量，将获取数据与正在训练的张量分离
        body_vel_actions = torch.clamp(self.actions[:, self._body_dof_idx], -1.0, 1.0) #将actions解释为速度
        body_pos_actions = self.pos_actions + self.dt * body_vel_actions * self.cfg.position_action_scale #将actions解释为位置
        self.pos_actions = torch.clamp(body_pos_actions, self.dof_pos_lower_limits[self._body_dof_idx], self.dof_pos_upper_limits[self._body_dof_idx])
        self.vel_actions = self.actions[:, self._wheel_dof_idx]

        self._visualize_markers()

    #应用动作，更新的数据应用于物理模拟，为指定关节设置期望目标值
    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self.pos_actions, joint_ids=self._body_dof_idx) #设置目标位置
        self.robot.set_joint_velocity_target(self.vel_actions, joint_ids=self._wheel_dof_idx) #设置目标速度

    #获取观测
    def _get_observations(self) -> dict:
        self.robot_lin_vel = self.robot.data.root_com_lin_vel_w

        self.forwards = math_utils.quat_apply(self.robot.data.root_quat_w, self.robot.data.FORWARD_VEC_B) #将机器人本体前进方向向量（单位向量，仅方向）转换到世界坐标系
        dot = torch.sum(self.forwards * self.commands, dim=-1, keepdim=True)
        cross = torch.cross(self.forwards, self.commands, dim=-1)[:,-1].reshape(-1,1)

        obs = torch.hstack((self.robot_lin_vel, dot, cross))
        observations = {"policy": obs}
        return observations

    #获取奖励，计算函数compute_rewards见最后
    def _get_rewards(self) -> torch.Tensor:
        # forward_velocity = torch.sum(self.robot.data.root_lin_vel_w[:, :2] * self.commands[:, :2], dim=-1) #在命令方向的速度投影
        # velocity_reward = torch.clamp(forward_velocity, 0, 1.0)
        
        dot = torch.sum(self.forwards * self.commands, dim=-1, keepdim=True)
        cross = torch.cross(self.forwards, self.commands, dim=-1)[:,-1].reshape(-1,1) 
        yaw_error = torch.atan2(cross, dot) #使用反正切函数计算偏航误差，范围[-π, π]
        yaw_reward = torch.exp(-3*torch.abs(yaw_error)).squeeze(-1)

        total_reward = yaw_reward
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

        #重置指令向量和可视化标记
        self.commands = torch.randn((self.cfg.scene.num_envs, 3)).to(device=self.device) #初始随机指令——世界坐标系
        self.commands[:,-1] = 0.0
        self.commands = self.commands/torch.linalg.norm(self.commands, dim=1, keepdim=True) #归一化
        self.yaws = torch.atan2(self.commands[:, 1], self.commands[:, 0]).unsqueeze(1)
        self._visualize_markers()

        #重置环境参数流程：获取默认初始状态 -> 调整位置到环境原点 -> 写入模拟器
        joint_pos = self.robot.data.default_joint_pos[env_ids] #获取默认关节位置
        self.joint_pos[env_ids] = joint_pos
        self.robot.write_joint_position_to_sim(joint_pos, None, env_ids)

        default_root_state = self.robot.data.default_root_state[env_ids] #获取默认根状态
        default_root_state[:, :3] += self.scene.env_origins[env_ids] #将根位置调整到各自环境的原点
        self.robot.write_root_state_to_sim(default_root_state, env_ids) #写入关节位置和速度

def define_markers() -> VisualizationMarkers:
    """Define markers with various different shapes."""
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/myMarkers",
        markers={
                "forward": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(1, 1, 2),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
                ),
                "command": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(1, 1, 2),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
        },
    )
    return VisualizationMarkers(cfg=marker_cfg)