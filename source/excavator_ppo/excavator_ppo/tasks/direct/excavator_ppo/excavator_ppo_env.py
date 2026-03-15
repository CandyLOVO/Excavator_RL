from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import RayCaster, ContactSensor
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

        self.dt = self.cfg.sim.dt * self.cfg.decimation #动作更新频率 = 模拟时间步长 * decimation

        # 动作缓冲区（观测中包含当前动作）
        self.actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.heading_error = torch.zeros(self.num_envs, device=self.device) #储存heading误差供观测和奖励使用
        self.base_link_lin_vel_b = torch.zeros(self.num_envs, 3, device=self.device)  # 底盘 link origin 线速度（体坐标系）
        self.root_link_vel_w = torch.zeros(self.num_envs, 3, device=self.device)  # 底盘 link origin 线速度（世界坐标系）
        self.velocity_deficit = torch.zeros(self.num_envs, device=self.device)  # 速度缺额（衡量受困程度）
        self.stuck_counter = torch.zeros(self.num_envs, device=self.device)  # 持续受困计数器（防止出生瞬间误触发）
        self.struggling_intensity = torch.zeros(self.num_envs, device=self.device)  # 连续受困强度（供奖励/调试）
        self.last_body_forward_vel = torch.zeros(self.num_envs, device=self.device)  # 上一步前进速度（用于支撑加速度奖励）

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

        # 铲斗接触传感器（检测机械臂与地面的接触力，为支撑行为提供反馈）
        self._bucket_contact = ContactSensor(self.cfg.bucket_contact_sensor)
        self.scene.sensors["bucket_contact"] = self._bucket_contact

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

        # 当航向误差大时降低前进速度指令（软门控）#只留下heading_gate
        heading_alignment = torch.clamp(torch.cos(heading_error), min=0.0)
        self.commands[:, 0] = self.raw_lin_vel_cmd * heading_alignment #cos门控：偏差90°->0，偏差0°->1

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
        current_arm_pos = self.pos_actions[:, 1:] #基于当前目标位置进行累加，而不是实际位置
        arm_pos_delta = arm_actions * self.dt * self.cfg.position_action_scale
        new_arm_pos = current_arm_pos + arm_pos_delta
        new_arm_pos = torch.clamp(
            new_arm_pos,
            self.dof_pos_lower_limits[arm_dof_idx],
            self.dof_pos_upper_limits[arm_dof_idx]
        )

        body_actions = actions[:, 5].clone()  # 提取车体偏航动作
        body_dof_idx = self._body_dof_idx[0]  # 仅body_yaw_joint索引
        current_body_pos = self.pos_actions[:, 0] #基于当前目标偏航位置进行累加
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

        # 刷新线速度和航向（确保观测数据为当前步，与 _get_rewards 调用顺序无关）
        root_link_vel_w = self.robot.data.body_link_vel_w[:, 0, :3]
        self.root_link_vel_w = root_link_vel_w
        self.base_link_lin_vel_b = math_utils.quat_apply_inverse(
            self.robot.data.root_quat_w, root_link_vel_w
        )
        self._update_heading_command()

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

        # 机械臂关节力矩（boom, forearm, bucket）——通过 PD 控制器位置误差估算，作为机械臂受力反馈
        arm_dof_idx = self._body_dof_idx[1:]  # boom, forearm, bucket
        arm_pos_target = self.pos_actions[:, 1:]  # 当前位置目标
        arm_pos_current = self.robot.data.joint_pos[:, arm_dof_idx]
        arm_torque_proxy = (arm_pos_target - arm_pos_current)  # 与实际 PD 力矩成正比，当铲斗触地受阻时误差增大

        # 铲斗地面接触力（z 方向反力，正值表示铲斗正在压地面）
        bucket_contact_z = self._bucket_contact.data.net_forces_w[:, 0, 2].clamp(min=0.0) #铲斗接触传感器测量的世界坐标系下z方向接触力，仅正值（压地面）
        bucket_contact_obs = (bucket_contact_z * cfg.contact_force_scale).unsqueeze(1)  # (N, 1)

        # 速度缺额：实际/指令速度比，衡量受困程度 (0=跟踪良好, 1=完全受困)
        cmd_vel = self.commands[:, 0].clamp(min=0.3)
        actual_vel = self.base_link_lin_vel_b[:, 0]
        self.velocity_deficit = (1.0 - (actual_vel / cmd_vel).clamp(0.0, 1.0))
        velocity_deficit_obs = self.velocity_deficit.unsqueeze(1)  # (N, 1)

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
            arm_torque_proxy * cfg.arm_torque_scale,      # (N, 3) 机械臂力矩反馈
            bucket_contact_obs,                            # (N, 1) 铲斗地面接触力
            velocity_deficit_obs,                          # (N, 1) 受困指示器
            height_data * cfg.height_scale,
        ), dim=-1)

        # 整体观测NaN/Inf保护，防止物理引擎异常污染网络权重
        obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        obs = torch.clamp(obs, -100.0, 100.0)

        return {"policy": obs}

    def _compute_struggling_intensity(
        self,
        actual_forward_vel: torch.Tensor,
        cmd_forward_vel: torch.Tensor,
        wheel_vel_abs: torch.Tensor,
    ) -> torch.Tensor:
        """计算连续受困强度。
        """
        wants_to_move = cmd_forward_vel > 0.3
        cmd_safe = torch.clamp(cmd_forward_vel, min=0.3)
        progress_ratio = torch.clamp(actual_forward_vel / cmd_safe, 0.0, 1.0) #实际速度与指令速度的比率，衡量跟踪程度
        velocity_deficit = 1.0 - progress_ratio #速度缺额，0=跟踪良好, 1=完全受困

        # 关键：只有"明显慢到接近被挡住"才触发受困，避免正常慢速巡航被误判。
        slow_score = torch.clamp((0.45 - actual_forward_vel) / 0.45, 0.0, 1.0)
        severe_deficit = torch.clamp((velocity_deficit - 0.35) / 0.65, 0.0, 1.0)
        wheel_slip_score = torch.clamp((wheel_vel_abs - 2.0) / 3.0, 0.0, 1.0) * (actual_forward_vel < 0.35).float()

        heading_alignment = torch.clamp(torch.cos(self.heading_error), min=0.0)
        heading_factor = 0.2 + 0.8 * heading_alignment #航向不是硬门槛，转向阶段保留弱感知

        raw_struggling = (
            (0.65 * severe_deficit + 0.35 * wheel_slip_score)
            * slow_score
            * wants_to_move.float()
            * heading_factor
        ) #计算受困程度，考虑速度缺额、轮速和航向对齐，只有在有前进指令时才评估受困，并且航向偏差较大时放宽受困判断

        warmup_steps = int(0.5 / self.dt) #缩短落地忽略期至0.5秒，遇到障碍可以更快反应
        raw_struggling = raw_struggling * (self.episode_length_buf > warmup_steps).float()

        self.stuck_counter = torch.where(
            raw_struggling > 0.12,
            self.stuck_counter + raw_struggling,
            torch.clamp(self.stuck_counter - 0.5, min=0.0),
        ) #受困时计数器增加，非受困时更快衰减

        activation_steps = int(0.3 / self.dt) #受困计数超过0.3秒即激活受困强度
        ramp_steps = int(0.5 / self.dt) #在0.5秒内爬升到满值
        struggling_intensity = torch.clamp(
            (self.stuck_counter - activation_steps).float() / max(ramp_steps, 1),
            0.0,
            1.0,
        ) #受困强度根据计数器线性增加，超过ramp_steps后达到最大值1.0
        return struggling_intensity

    #获取奖励
    def _get_rewards(self) -> torch.Tensor:
        cfg = self.cfg

        #刷新缓存的身体状态
        self.forwards = math_utils.quat_apply(self.robot.data.root_quat_w, self.robot.data.FORWARD_VEC_B)
        self._update_heading_command()
        root_link_vel_w = self.robot.data.body_link_vel_w[:, 0, :3]  # (N, 3) root link origin 线速度（世界系）
        self.root_link_vel_w = root_link_vel_w
        self.base_link_lin_vel_b = math_utils.quat_apply_inverse(
            self.robot.data.root_quat_w, root_link_vel_w
        )  # (N, 3) 转换到体坐标系
        gravity_world = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, -1)
        self.gravity_body = math_utils.quat_apply_inverse(self.robot.data.root_quat_w, gravity_world)

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
        heading_error_abs = torch.abs(heading_error)
        reward_far = torch.exp(-heading_error_abs / cfg.heading_sigma_far)  # 远区引导 (宽)
        reward_near = torch.exp(-heading_error_abs / cfg.heading_sigma_near) # 近区微调 (窄，梯度极大)
        heading_reward = 0.4 * reward_far + 0.6 * reward_near

        heading_gate = torch.clamp(torch.cos(self.heading_error), min=0.0) #航向门控，偏差90°->0，偏差0°->1
        # heading_gate = -2.0/math.pi * torch.abs(self.heading_error) + 1.0 #线性门控，偏差90°->0，偏差0°->1
        # heading_gate = torch.exp(-torch.square(self.heading_error) / (2 * 0.5**2)) #高斯门控，偏差0.5 rad (~28°)处约为0.61，偏差90°处约为0.0019

        # # 前进奖励，仅当朝向对齐时给予奖励 [0, +) #和tracking_lin_vel冲突
        # body_lin_vel = self.base_link_lin_vel_b[:, 0] #前进速度（body x 方向）
        # forward_reward = torch.clamp(body_lin_vel, min=0.0)

        # # 转向速度奖励 [0, +)，奖励正确方向的角速度大小 #和tracking_ang_vel职责冲突
        # yaw_rate = self.robot.data.root_com_ang_vel_b[:, 2]
        # correct_dir = torch.sign(heading_error) * torch.sign(yaw_rate) #方向一致为1,反向为-1
        # turning_reward = torch.clamp(correct_dir, min=0.0) * torch.abs(yaw_rate) #方向正确时奖励角速度大小

        # 后退惩罚 (-, 0]，按航向对齐度调节严厉程度（背对目标时放松，避免转向死锁）
        body_forward_vel = self.base_link_lin_vel_b[:, 0]
        forward_vel = self.root_link_vel_w[:, 1] #前进速度（world y 方向，link origin 速度）
        # heading_alignment = torch.clamp(torch.cos(heading_error), min=0.0)  # 对齐=1, 背对=0
        # backward_body_penalty = -torch.clamp(-body_forward_vel, min=0.0) * heading_alignment
        # backward_world_penalty = -torch.clamp(-forward_vel, min=0.0) * heading_alignment
        backward_body_penalty = -torch.clamp(-body_forward_vel, min=0.0)
        backward_world_penalty = -torch.clamp(-forward_vel, min=0.0)

        # 倾覆惩罚 (-, 0]
        orientation_penalty = -torch.sum(torch.square(self.gravity_body[:, :2]), dim=1)

        # 动作平滑度 (-, 0]
        action_rate = -torch.sum(torch.square(self.actions - self.last_actions), dim=1)

        # 垂直速度和角速度惩罚，增设死区以容忍正常行驶时的路面颠簸，但严惩暴力跳跃
        lin_vel_z_abs = torch.abs(self.base_link_lin_vel_b[:, 2])
        lin_vel_z_penalty = -torch.square(torch.clamp(lin_vel_z_abs - 0.2, min=0.0))
        ang_vel_xy_abs = torch.abs(self.robot.data.root_com_ang_vel_b[:, :2])
        ang_vel_xy_penalty = -torch.sum(torch.square(torch.clamp(ang_vel_xy_abs - 0.5, min=0.0)), dim=1)

        # 受困检测
        actual_forward_vel = self.base_link_lin_vel_b[:, 0]
        cmd_forward_vel = self.commands[:, 0]
        wheel_vel_abs = torch.mean(torch.abs(self.robot.data.joint_vel[:, self._wheel_dof_idx]), dim=1)
        struggling_intensity = self._compute_struggling_intensity(
            actual_forward_vel,
            cmd_forward_vel,
            wheel_vel_abs,
        )
        self.struggling_intensity = struggling_intensity

        # 获取铲斗触地强度
        bucket_contact_z = self._bucket_contact.data.net_forces_w[:, 0, 2].clamp(min=0.0)
        contact_strength = 1.0 - torch.exp(-bucket_contact_z / 1500.0)
        
        # 拖地惩罚 (-, 0]：非受困时强烈惩罚触地（防止未遇障把机械臂当拐杖）。受困时惩罚降为0
        drag_penalty = -contact_strength * torch.square(1.0 - struggling_intensity)

        # 上装能耗惩罚 (-, 0]，鼓励高效利用上半身（包含旋转座舱和机械臂），抑制无用的“划船”、左右乱晃
        upper_actions = self.actions[:, 2:6]
        upper_effort = -torch.sum(torch.square(upper_actions), dim=1)

        # 核心进度奖励 (Progress Reward)
        # 鼓励挖掘机在世界坐标系下向着目标方向产生实际的位移，使用 heading_gate 避免倒位获利
        target_yaw = self.commands[:, 3]
        target_dir_x = torch.cos(target_yaw)
        target_dir_y = torch.sin(target_yaw)
        actual_vel_x = self.root_link_vel_w[:, 0]
        actual_vel_y = self.root_link_vel_w[:, 1]
        progress_vel = actual_vel_x * target_dir_x + actual_vel_y * target_dir_y #目标方向上的实际速度分量

        body_forward_vel = self.base_link_lin_vel_b[:, 0] #底盘前进速度分量

        # 联合进度：只有既在向目标移动，又是底盘正向移动时才给予高额奖励。这就自然且非强制地抵制了“打转/侧滑/倒车”向目标移动的行为。
        effective_progress = torch.min(progress_vel, body_forward_vel)
        progress_reward = torch.clamp(effective_progress, min=0.0, max=cfg.lin_vel_x_range[1]) * heading_gate
        
        # 受困触地支持奖励：当陷入困境时，首先鼓励机械臂主动触底提供支撑力（基础分），如果进一步产生了向前位移，则给予乘数加成。
        # 不再和前进速度直接相乘（防止受困完全停滞时得不到触地探索的奖励闭环）。
        movement_bonus = 1.0 + 2.0 * torch.clamp(effective_progress, min=0.0, max=cfg.lin_vel_x_range[1])
        support_reward = struggling_intensity * contact_strength * movement_bonus * heading_gate

        total_reward = (
            + 5.0 * progress_reward                     # 极大地奖励车头朝前且向目标行进的综合有效位移
            + 4.0 * support_reward                      # 受困时用机械臂触地起支撑作用给予高额加分，无需立即产生位移即有收益
            + 1.5 * heading_reward                      # 辅助航向对齐奖励
            + 1.0 * backward_body_penalty               # 适度惩罚倒车，配合 gating 已能切断倒车获利，允许短后退调姿
            + 2.0 * drag_penalty                        # 强烈惩罚在非受困时机械臂拖地，逼迫其平地收起
            + 1.0 * orientation_penalty                 # 惩罚过度翘起或翻车（直接抑制前倾后仰）
            + 1.0 * lin_vel_z_penalty                   # 严惩底盘质心突然垂直剧烈跳动（但带死区容忍爬坡）
            + 0.5 * ang_vel_xy_penalty                  # 严惩底盘剧烈翻滚/俯仰
            + 0.05 * upper_effort                       # 上半身能耗惩罚，抑制车体无意义地持续乱转
            + 0.5 * action_rate                         # 平滑惩罚
        )

        total_reward = torch.nan_to_num(total_reward, nan=0.0, posinf=0.0, neginf=0.0)

        # 更新上一步前进速度缓存（用于下一步动态拉拽奖励）
        self.last_body_forward_vel = body_forward_vel.clone()

        return total_reward

    #获取终止状态，返回是否越界和是否超时
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # 翻车检测
        gravity_world = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, -1)
        current_gravity_body = math_utils.quat_apply_inverse(self.robot.data.root_quat_w, gravity_world)
        flipped = current_gravity_body[:, 2] > -0.3 # 当机器人翻转超过约 70°（即 up 向量 z 分量 > -0.3）时，认为翻车终止

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

        # 重置目标位置动作缓冲区，防止跨回合残留
        if hasattr(self, 'pos_actions'):
            self.pos_actions[env_ids] = self.robot.data.default_joint_pos[env_ids][:, self._body_dof_idx].clone()

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

        # 初始朝向随机，以目标航向为中心 ±π
        random_yaw = self.commands[env_ids, 3] + (torch.rand(n, device=self.device) - 0.5) * 2.0 * math.pi #在目标航向基础上添加±180°的随机偏航
        default_root_state[:, 3:7] = math_utils.quat_from_angle_axis(random_yaw.unsqueeze(-1), self.up_dir).reshape(-1, 4) #根据随机朝向计算初始四元数

        self.robot.write_root_state_to_sim(default_root_state, env_ids)

        # 重置速度缓存，防止 _get_observations 读到重置前的旧速度
        self.base_link_lin_vel_b[env_ids] = 0.0
        self.root_link_vel_w[env_ids] = 0.0
        self.stuck_counter[env_ids] = 0.0  # 重置受困计数器
        self.struggling_intensity[env_ids] = 0.0
        self.last_body_forward_vel[env_ids] = 0.0  # 重置前进速度缓存

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

            # 重置时也按航向对齐度缩放前进速度指令
            heading_alignment = torch.clamp(torch.cos(heading_error), min=0.0)
            self.commands[env_ids, 0] = self.raw_lin_vel_cmd[env_ids] * heading_alignment

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