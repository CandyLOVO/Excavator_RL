from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import RayCaster
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

        self.num_body_dof = len(self.cfg.body_dof_name)
        self.num_wheel_dof = len(self.cfg.wheel_dof_name)
        self._body_dof_idx, _ = self.robot.find_joints(self.cfg.body_dof_name) #关节索引 0、7、8、9
        self._wheel_dof_idx, _ = self.robot.find_joints(self.cfg.wheel_dof_name) #1、2、3、4、5、6
 
        self.joint_pos = self.robot.data.joint_pos
        self.dof_pos_lower_limits = self.robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.dof_pos_upper_limits = self.robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        self.pos_actions = self.robot.data.default_joint_pos[:, self._body_dof_idx].clone()
        self.default_joint_pos = self.robot.data.default_joint_pos.clone()  # 默认关节位置（观测用）

        # # 机械臂“收起”目标位置，设为关节行程中点
        # arm_idx = self._body_dof_idx[1:]  # boom, forearm, bucket
        # arm_lower = self.dof_pos_lower_limits[arm_idx]
        # arm_upper = self.dof_pos_upper_limits[arm_idx]
        # self.arm_stowed_pos = ((arm_lower + arm_upper) / 2.0).unsqueeze(0)  # (1, 3)

        self.dt = self.cfg.sim.dt * self.cfg.decimation #动作更新频率 = 模拟时间步长 * decimation

        # 动作缓冲区（观测中包含当前动作）
        self.actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.heading_error = torch.zeros(self.num_envs, device=self.device) #储存heading误差供观测和奖励使用
        self.base_link_lin_vel_b = torch.zeros(self.num_envs, 3, device=self.device)  # 底盘 link origin 线速度（体坐标系）

        # 命令缩放向量（用于观测归一化）
        self.commands_scale = torch.tensor(
            [self.cfg.lin_vel_scale, self.cfg.lin_vel_scale, self.cfg.ang_vel_scale],
            device=self.device,
        )

        #### 测试语句 ####
        # print(f"DEBUG: Body DOF Indices: {self._body_dof_idx}")
        # print(f"DEBUG: Wheel DOF Indices: {self._wheel_dof_idx}")
        # print(f"DEBUG: Default joint pos shape: {self.robot.data.default_joint_pos.shape}")
        # print(f"DEBUG: Body default positions: {self.robot.data.default_joint_pos[0, self._body_dof_idx]}")
        # print(f"DEBUG: Initialized pos_actions: {self.pos_actions[0]}")
        # print(f"DEBUG: Body DOF names: {self.cfg.body_dof_name}")
        # print(f"DEBUG: DOF lower limits (body): {self.dof_pos_lower_limits[self._body_dof_idx]}")
        # print(f"DEBUG: DOF upper limits (body): {self.dof_pos_upper_limits[self._body_dof_idx]}")

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot

        # 高度扫描传感器
        self._height_scanner = RayCaster(self.cfg.height_scanner)
        self.scene.sensors["height_scanner"] = self._height_scanner

        # 灯光
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # 地形
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # 克隆环境（必须在所有场景对象注册之后）
        self.scene.clone_environments(copy_from_source=False) #每个环境地形、初始朝向等独立随机

        # 设置环境原点和目标位置
        num_stages = self.cfg.track_num_stages
        tw         = self.cfg.track_width
        sec_l      = self.cfg.track_section_length #每段地形沿 y 方向长度

        y_start = 0.5 * sec_l - num_stages * sec_l / 2.0 # col 0 起点位置
        y_goal  = (num_stages - 0.5) * sec_l - num_stages * sec_l / 2.0 # col 5 中心

        self._terrain.env_origins = torch.zeros((self.num_envs, 3), device=self.device) #环境原点位置
        self._terrain.env_origins[:, 1] = y_start 

        self.goal_y = y_goal
        self.start_y = y_start
        self.track_length = y_goal - y_start
        self.half_track_width = tw / 2.0

        # 命令向量 [lin_vel_x, lin_vel_y, ang_vel_yaw, heading]
        self.commands = torch.zeros((self.num_envs, self.cfg.num_commands), device=self.device)
        self.raw_lin_vel_cmd = torch.zeros(self.num_envs, device=self.device)  # 原始前进速度指令
        self._resample_commands(torch.arange(self.num_envs, device=self.device)) #初始命令随机

        # 可视化标记
        self.visualization_markers = define_markers()
        self.marker_locations = torch.zeros((self.num_envs, 3)).to(device=self.device)
        self.marker_offset = torch.zeros((self.num_envs, 3)).to(device=self.device)
        self.marker_offset[:, -1] = 3.0
        self.forward_marker_orientations = torch.zeros((self.num_envs, 4)).to(device=self.device)
        self.command_marker_orientations = torch.zeros((self.num_envs, 4)).to(device=self.device)

        self.yaws = self.commands[:, 3:4].clone()  # heading 角度用于可视化
        self.up_dir = torch.tensor([0.0, 0.0, 1.0]).to(device=self.device)
   
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

    # 命令重采样
    def _resample_commands(self, env_ids: torch.Tensor):
        """随机生成 lin_vel_x, lin_vel_y, heading
        在 heading 模式下 ang_vel_yaw 由 _update_heading_command 重算
        """
        n = len(env_ids)
        cfg = self.cfg
        self.commands[env_ids, 0] = torch.empty(n, device=self.device).uniform_(
            cfg.lin_vel_x_range[0], cfg.lin_vel_x_range[1]
        )
        self.commands[env_ids, 1] = torch.empty(n, device=self.device).uniform_(
            cfg.lin_vel_y_range[0], cfg.lin_vel_y_range[1]
        )
        if cfg.heading_command:
            self.commands[env_ids, 3] = torch.empty(n, device=self.device).uniform_( #随机 heading 目标
                cfg.heading_range[0], cfg.heading_range[1]
            )
        else:
            self.commands[env_ids, 2] = torch.empty(n, device=self.device).uniform_( #随机 ang_vel_yaw 目标
                cfg.ang_vel_yaw_range[0], cfg.ang_vel_yaw_range[1]
            )
        # 小速度命令置零（避免微小指令干扰）
        self.commands[env_ids, :2] *= (
            torch.norm(self.commands[env_ids, :2], dim=1) > 0.2
        ).unsqueeze(1)
        # 保存原始前进速度指令
        self.raw_lin_vel_cmd[env_ids] = self.commands[env_ids, 0].clone()
        # 更新可视化 heading 角度
        self.yaws = self.commands[:, 3:4].clone()

    def _update_heading_command(self):
        """heading 模式：根据 heading 误差重新计算 ang_vel_yaw 命令。
        智能体始终尝试调整朝向以对齐目标 heading 而不是直接控制角速度。
        """
        if not self.cfg.heading_command:
            return
        forward = math_utils.quat_apply(self.robot.data.root_quat_w, self.robot.data.FORWARD_VEC_B)
        heading = torch.atan2(forward[:, 1], forward[:, 0])  # 当前机体朝向角(-π, π)
        heading_error = self.commands[:, 3] - heading
        heading_error = torch.atan2(torch.sin(heading_error), torch.cos(heading_error)) #归一化偏航误差到(-π, π)
        self.heading_error = heading_error  # 存储供观测和奖励使用
        self.commands[:, 2] = torch.clamp(
            self.cfg.heading_kp * heading_error,
            -self.cfg.max_ang_vel, self.cfg.max_ang_vel,
        )  # 比例控制，增益与截断均由 cfg 配置

        # error_sign = torch.sign(heading_error) #会使挖掘机超过目标方向后再前进
        # abs_error = torch.abs(heading_error)
        # enhanced_error = error_sign * torch.pow(abs_error, 0.8)       
        # self.commands[:, 2] = torch.clamp(
        #     self.cfg.heading_kp * enhanced_error,
        #     -self.cfg.max_ang_vel, self.cfg.max_ang_vel,
        # )  # 比例控制，增益与截断均由 cfg 配置

        # # 当航向误差大时降低前进速度指令（软门控）#只留下heading_gate
        # heading_alignment = torch.clamp(torch.cos(heading_error), min=0.0)
        # self.commands[:, 0] = self.raw_lin_vel_cmd * heading_alignment #cos门控：偏差90°->0，偏差0°->1

    #更新动作，得到动作张量的副本
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # 保存动作用于观测
        self.last_actions[:] = self.actions[:]
        self.actions[:] = actions[:]

        vel_actions = actions[:, :2].clone() * self.cfg.action_scale
        left_wheel_vel = vel_actions[:, 0]   # 左侧履带速度
        right_wheel_vel = vel_actions[:, 1]  # 右侧履带速度

        self.vel_actions = torch.zeros((self.num_envs, self.num_wheel_dof), device=self.device)
        self.vel_actions[:, 0:3] = left_wheel_vel.unsqueeze(1)
        self.vel_actions[:, 3:6] = right_wheel_vel.unsqueeze(1)

        arm_actions = actions[:, 2:5].clone()  # 提取机械臂动作
        arm_dof_idx = self._body_dof_idx[1:]  # 跳过body_yaw_joint，只控制boom, forearm, bucket
        current_arm_pos = self.robot.data.joint_pos[:, arm_dof_idx] #获取当前机械臂关节位置  
        arm_pos_delta = arm_actions * self.dt * self.cfg.position_action_scale
        new_arm_pos = current_arm_pos + arm_pos_delta
        new_arm_pos = torch.clamp(
            new_arm_pos,
            self.dof_pos_lower_limits[arm_dof_idx],
            self.dof_pos_upper_limits[arm_dof_idx]
        )

        body_actions = actions[:, 5].clone()  # 提取车体偏航动作
        body_dof_idx = self._body_dof_idx[0]  # 仅body_yaw_joint索引
        current_body_pos = self.robot.data.joint_pos[:, body_dof_idx] #获取当前车体偏航位置
        body_pos_delta = body_actions * self.dt * self.cfg.body_yaw_scale
        new_body_pos = current_body_pos + body_pos_delta
        new_body_pos = torch.clamp(
            new_body_pos,
            self.dof_pos_lower_limits[body_dof_idx],
            self.dof_pos_upper_limits[body_dof_idx]
        )
        
        # 更新完整的body位置目标
        self.pos_actions = self.robot.data.default_joint_pos[:, self._body_dof_idx].clone() # 重置为默认位置
        self.pos_actions[:, 1:] = new_arm_pos
        self.pos_actions[:, 0] = new_body_pos

        self._visualize_markers()

    #应用动作，更新的数据应用于物理模拟，为指定关节设置期望目标值
    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self.pos_actions, joint_ids=self._body_dof_idx) #设置目标位置
        self.robot.set_joint_velocity_target(self.vel_actions, joint_ids=self._wheel_dof_idx) #设置目标速度

    #获取观测
    def _get_observations(self) -> dict:
        cfg = self.cfg

        # 本体线速度 / 角速度（body frame）
        base_lin_vel = self.base_link_lin_vel_b  # (N,3) link origin 速度，非质心速度，避免转向时质心偏移产生寄生分量
        base_ang_vel = self.robot.data.root_com_ang_vel_b  # (N,3) 角速度对刚体上所有点一致，无需修正

        # 重力投影（body frame）
        gravity_world = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, -1)
        self.gravity_body = math_utils.quat_apply_inverse(self.robot.data.root_quat_w, gravity_world)

        # 机械臂关节状态（4 DOF: body_yaw, boom, forearm, bucket），使用相对默认位置的偏移量
        arm_joint_pos = self.robot.data.joint_pos[:, self._body_dof_idx] - self.default_joint_pos[:, self._body_dof_idx]  # (N,4)
        arm_joint_vel = self.robot.data.joint_vel[:, self._body_dof_idx]  # (N,4)

        # 轮子速度
        wheel_vel = self.robot.data.joint_vel[:, self._wheel_dof_idx]  # (N,6)

        # 地形高度测量（RayCaster）使用机器人根节点 z 坐标而非传感器 z 坐标
        height_data = (
            self.robot.data.root_pos_w[:, 2].unsqueeze(1) 
            - self._height_scanner.data.ray_hits_w[..., 2]
            - cfg.base_height_offset 
        )
        height_data = torch.nan_to_num(height_data, nan=0.0, posinf=1.0, neginf=-1.0)  # 射线未命中时ray_hits为inf，防止NaN传播
        height_data = height_data.clip(-1.0, 1.0)

        obs = torch.cat((
            base_lin_vel * cfg.lin_vel_scale,
            base_ang_vel * cfg.ang_vel_scale,
            self.gravity_body,
            self.commands[:, 0:1] * cfg.lin_vel_scale,    # 目标前进速度（经航向门控）
            torch.sin(self.heading_error).unsqueeze(1),   # 航向误差 sin 分量（策略直接感知方向偏差）
            torch.cos(self.heading_error).unsqueeze(1),   # 航向误差 cos 分量
            arm_joint_pos * cfg.dof_pos_scale,
            arm_joint_vel * cfg.dof_vel_scale,
            wheel_vel * cfg.wheel_vel_scale,
            self.actions,
            height_data * cfg.height_scale,
        ), dim=-1)

        # 整体观测NaN/Inf保护，防止物理引擎异常污染网络权重
        obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        obs = torch.clamp(obs, -100.0, 100.0)

        return {"policy": obs}

    #获取奖励
    def _get_rewards(self) -> torch.Tensor:
        cfg = self.cfg

        # 线速度跟踪 [0, 1]
        lin_vel_error = torch.sum(torch.square(
            self.commands[:, :2] - self.base_link_lin_vel_b[:, :2]
        ), dim=1)
        tracking_lin_vel = torch.exp(-lin_vel_error / cfg.tracking_lin_vel_sigma)

        # 角速度跟踪奖励 [0, 1]
        ang_vel_error = torch.square(self.commands[:, 2] - self.robot.data.root_com_ang_vel_b[:, 2])
        tracking_ang_vel = torch.exp(-ang_vel_error / cfg.tracking_ang_vel_sigma)

        # 航向误差（由_update_heading_command在_get_dones中计算并存储）[0, 1]
        heading_error = self.heading_error
        heading_error_abs = torch.abs(self.heading_error)
        reward_far = torch.exp(-heading_error_abs / cfg.heading_sigma_far)  # 远区引导 (宽)
        reward_near = torch.exp(-heading_error_abs / cfg.heading_sigma_near) # 近区微调 (窄，梯度极大)
        heading_reward = 0.4 * reward_far + 0.6 * reward_near

        # heading_gate = torch.clamp(torch.cos(heading_error), min=0.0) #航向门控，偏差90°->0，偏差0°->1
        # heading_gate = -2.0/math.pi * torch.abs(heading_error) + 1.0 #线性门控，偏差90°->0，偏差0°->1
        heading_gate = torch.exp(-torch.square(heading_error) / (2 * 0.5**2)) #高斯门控，偏差0.5 rad (~28°)处约为0.61，偏差90°处约为0.0019

        # # 前进奖励，仅当朝向对齐时给予奖励 [0, +) #和tracking_lin_vel冲突
        # body_lin_vel = self.base_link_lin_vel_b[:, 0] #前进速度（body x 方向）
        # forward_reward = torch.clamp(body_lin_vel, min=0.0)

        # # 转向速度奖励 [0, +)，奖励正确方向的角速度大小 #和tracking_ang_vel职责冲突
        # yaw_rate = self.robot.data.root_com_ang_vel_b[:, 2]
        # correct_dir = torch.sign(heading_error) * torch.sign(yaw_rate) #方向一致为1,反向为-1
        # turning_reward = torch.clamp(correct_dir, min=0.0) * torch.abs(yaw_rate) #方向正确时奖励角速度大小

        # 后退惩罚 (-, 0]
        body_forward_vel = self.base_link_lin_vel_b[:, 0]
        forward_vel = self.robot.data.root_lin_vel_w[:, 1] #前进速度（world y 方向）
        backward_body_penalty = -torch.clamp(-body_forward_vel, min=0.0)
        backward_world_penalty = -torch.clamp(-forward_vel, min=0.0)

        # 倾覆惩罚 (-, 0]
        orientation_penalty = -torch.sum(torch.square(self.gravity_body[:, :2]), dim=1)

        # 垂直线速度惩罚 (-, 0]
        lin_vel_z_penalty = -torch.square(self.base_link_lin_vel_b[:, 2])

        # 角速度 xy 惩罚 (-, 0]
        ang_vel_xy_penalty = -torch.sum(torch.square(self.robot.data.root_com_ang_vel_b[:, :2]), dim=1)

        # 底盘高度奖励 (-, 0]
        base_height = self.robot.data.root_pos_w[:, 2]
        terrain_below = self._height_scanner.data.ray_hits_w[..., 2]
        terrain_below = torch.nan_to_num(terrain_below, nan=-10.0, posinf=-10.0, neginf=-10.0) 
        terrain_height_est = torch.max(terrain_below, dim=1).values  # 机器人正下方附近最高地形点
        clearance = base_height - terrain_height_est  # 实际离地高度
        base_height_reward = -torch.square(clearance - cfg.base_height_target) #(期望离地高度-实际离地高度)^2

        # 动作平滑度 (-, 0]
        action_rate = -torch.sum(torch.square(self.actions - self.last_actions), dim=1)

        # # 机械臂收纳奖励 [0, 1]
        # arm_joint_pos = self.robot.data.joint_pos[:, self._body_dof_idx[1:]]
        # arm_safe_reward = torch.exp(-torch.sum(torch.square(arm_joint_pos - self.arm_stowed_pos), dim=1)) #偏差越小奖励越大

        # 机械臂能耗惩罚 (-, 0]，弱惩罚
        arm_actions = self.actions[:, 2:5]
        arm_effort = -torch.sum(torch.square(arm_actions), dim=1)

        # 车体偏航居中 [0, 1]
        body_yaw = self.robot.data.joint_pos[:, self._body_dof_idx[0]]
        centering_reward = torch.exp(-torch.abs(body_yaw))

        # 存活奖励 cfg.alive_reward_scale/步
        alive_reward = torch.ones(self.num_envs, device=self.device)

        total_reward = (
            + 1.0 * tracking_lin_vel * heading_gate # 线速度跟踪，航向对齐时才计入 [0, 1]
            + 0.5 * tracking_ang_vel # 角速度跟踪 [0, 1]
            + 3.0 * heading_reward # 朝向对齐 [0, 1]
            + 1.0 * backward_world_penalty # 后退惩罚，world (-, 0]
            + 0.5 * backward_body_penalty # 后退惩罚，body (-, 0]
            + 0.5 * orientation_penalty # 倾覆惩罚 (-, 0]
            # + 0.2 * lin_vel_z_penalty # 减垂直速度成分，减少弹跳 (-, 0]
            # + 0.05 * ang_vel_xy_penalty # xy角速度惩罚 (-, 0]
            # + 0.15 * base_height_reward # 底盘高度保持 (-, 0]
            # + 0.01 * action_rate # 动作平滑度 (-, 0]
            # + 0.01 * arm_effort # 机械臂能耗 (-, 0]
            # + 0.3 * centering_reward # 车体偏航居中 [0, 1]
            # + 0.5 * alive_reward  # 存活奖励 cfg.alive_reward_scale/步
        )

        total_reward = torch.nan_to_num(total_reward, nan=0.0, posinf=0.0, neginf=0.0) #奖励NaN/Inf保护，当物理异常导致奖励计算出NaN/Inf时置零，防止污染网络权重

        return total_reward

    #获取终止状态，返回是否越界和是否超时
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        self.forwards = math_utils.quat_apply(self.robot.data.root_quat_w, self.robot.data.FORWARD_VEC_B) #计算前进方向向量（世界坐标系，_get_rewards使用）
        self._update_heading_command()

        # 计算底盘几何中心（link origin）线速度，避免整机质心偏移导致转向时的寄生横向速度
        # body_link_vel_w 使用 v_link = v_com + ω × (-r_com) 从质心速度反推 link origin 速度
        root_link_vel_w = self.robot.data.body_link_vel_w[:, 0, :3]  # (N, 3) root link origin 线速度（世界系）
        self.base_link_lin_vel_b = math_utils.quat_apply_inverse(
            self.robot.data.root_quat_w, root_link_vel_w
        )  # (N, 3) 转换到体坐标系

        # 翻车检测
        gravity_world = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, -1)
        self.gravity_body = math_utils.quat_apply_inverse(self.robot.data.root_quat_w, gravity_world) #重力投影（_get_rewards也用）
        flipped = self.gravity_body[:, 2] > -0.3

        # 虚空坠落检测
        robot_pos = self.robot.data.root_pos_w
        fallen_into_void = robot_pos[:, 2] < -5.0

        # 出界检测
        too_far_back = robot_pos[:, 1] < (self.start_y - 5.0) #当前进方向坐标小于起点坐标-5m时，认为越界
        too_far_side = torch.abs(robot_pos[:, 0]) > self.half_track_width #当前横向坐标绝对值大于半轨道宽度时，认为越界

        terminated = flipped | too_far_back | too_far_side | fallen_into_void
        truncated = time_out.to(torch.bool)
        return terminated, truncated

    #重置环境
    def _reset_idx(self, env_ids: Sequence[int] | None): #env_ids表示要重置的环境索引
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        # 重置关节内部状态（清除残留外力/力矩/速度缓冲，必须在写入新状态之前调用）
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids) #调用父类的重置方法

        # 重置动作缓冲区
        self.actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0

        # 重采样命令
        self._resample_commands(env_ids)
        self._visualize_markers()

        #重置环境参数流程：获取默认初始状态 -> 调整位置到环境原点 -> 写入模拟器
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]  # 默认为零速度
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)  # 同时写入位置和速度

        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        default_root_state[:, 2] += 0.05  # 仅略微抬高，避免重心靠前导致前倾

        # 在起点平地内添加随机位置偏移，增加训练多样性
        n = len(env_ids)
        random_x = (torch.rand(n, device=self.device) - 0.5) * self.cfg.track_width * 0.8
        random_y = (torch.rand(n, device=self.device) - 0.5) * self.cfg.track_section_length * 0.6
        default_root_state[:, 0] += random_x
        default_root_state[:, 1] += random_y

        # 初始朝向随机，以目标航向为中心 ±π，确保最大偏差不超过 90°
        random_yaw = self.commands[env_ids, 3] + (torch.rand(n, device=self.device)) * math.pi 
        default_root_state[:, 3:7] = math_utils.quat_from_angle_axis(random_yaw.unsqueeze(-1), self.up_dir).reshape(-1, 4) #根据随机朝向计算初始四元数

        self.robot.write_root_state_to_sim(default_root_state, env_ids)

        # 根据已知的 random_yaw 立即更新航向角速度指令，避免观测中残留上一 episode 结束时的旧 commands[:,2]
        if self.cfg.heading_command:
            heading_error = self.commands[env_ids, 3] - random_yaw
            heading_error = torch.atan2(torch.sin(heading_error), torch.cos(heading_error))
            self.heading_error[env_ids] = heading_error  # 存储供重置后首帧观测使用
            self.commands[env_ids, 2] = torch.clamp( #根据初始随机朝向计算初始ang_vel_yaw
                self.cfg.heading_kp * heading_error,
                -self.cfg.max_ang_vel, self.cfg.max_ang_vel,
            )

            # error_sign = torch.sign(heading_error)
            # abs_error = torch.abs(heading_error)
            # enhanced_error = error_sign * torch.pow(abs_error, 0.8)
            # self.commands[env_ids, 2] = torch.clamp(
            #     self.cfg.heading_kp * enhanced_error,
            #     -self.cfg.max_ang_vel, self.cfg.max_ang_vel,
            # )  # 比例控制，增益与截断均由 cfg 配置

            # # 重置时也按航向对齐度缩放前进速度指令
            # heading_alignment = torch.clamp(torch.cos(heading_error), min=0.0)
            # self.commands[env_ids, 0] = self.raw_lin_vel_cmd[env_ids] * heading_alignment

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