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

        self.num_body_dof = len(self.cfg.body_dof_name)
        self.num_wheel_dof = len(self.cfg.wheel_dof_name)
        self._body_dof_idx, _ = self.robot.find_joints(self.cfg.body_dof_name) #关节索引 0、7、8、9
        self._wheel_dof_idx, _ = self.robot.find_joints(self.cfg.wheel_dof_name) #1、2、3、4、5、6
 
        self.joint_pos = self.robot.data.joint_pos
        self.dof_pos_lower_limits = self.robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.dof_pos_upper_limits = self.robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        self.pos_actions = self.robot.data.default_joint_pos[:, self._body_dof_idx].clone()

        self.dt = self.cfg.sim.dt * self.cfg.decimation

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
        self.robot = Articulation(self.cfg.robot_cfg) #机器人为Articulation类型，传入配置参数
        # add ground plane
        # spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        ##################### 创建指地形 ######################
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        ######################################################
        
        ################# 创建指令向量（目标值）##################
        self.commands = torch.randn((self.cfg.scene.num_envs, 3)).to(device=self.device) #初始随机指令——世界坐标系
        self.commands[:, -1] = 0.0
        cmd_norm = torch.linalg.norm(self.commands, dim=1, keepdim=True).clamp_min(1e-6)
        self.commands = self.commands / cmd_norm
        ######################################################

        #####################创建可视化标记#####################
        self.visualization_markers = define_markers()
        self.marker_locations = torch.zeros((self.cfg.scene.num_envs, 3)).to(device=self.device) #标记位置
        self.marker_offset = torch.zeros((self.cfg.scene.num_envs, 3)).to(device=self.device) #标记偏移量
        self.marker_offset[:, -1] = 3.0 #标记在Z轴上方3米
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
        # self.actions = actions.clone() #避免修改原始动作张量，将获取数据与正在训练的张量分离
        # body_vel_actions = torch.clamp(self.actions[:, self._body_dof_idx], -1.0, 1.0) #将actions解释为速度
        # body_pos_actions = self.pos_actions + self.dt * body_vel_actions * self.cfg.position_action_scale #将actions解释为位置
        # self.pos_actions = torch.clamp(body_pos_actions, self.dof_pos_lower_limits[self._body_dof_idx], self.dof_pos_upper_limits[self._body_dof_idx])
        # self.vel_actions = self.actions[:, self._wheel_dof_idx]

        vel_actions = actions[:, :2].clone() * self.cfg.action_scale
        left_wheel_vel = vel_actions[:, 0]   # 左侧履带速度
        right_wheel_vel = vel_actions[:, 1]  # 右侧履带速度

        self.vel_actions = torch.zeros((self.num_envs, self.num_wheel_dof), device=self.device)
        self.vel_actions[:, 0:3] = left_wheel_vel.unsqueeze(1)
        self.vel_actions[:, 3:6] = right_wheel_vel.unsqueeze(1)

        # body_arm_actions = actions[:, 2:6].clone()
        # current_pos = self.robot.data.joint_pos[:, self._body_dof_idx]
        # scales = torch.tensor([self.cfg.body_yaw_scale, self.cfg.position_action_scale, self.cfg.position_action_scale, self.cfg.position_action_scale], device=self.device)
        # pos_delta = body_arm_actions * self.dt * scales
        # new_pos = current_pos + pos_delta
        # self.pos_actions = torch.clamp(
        #     new_pos,
        #     self.dof_pos_lower_limits[self._body_dof_idx],
        #     self.dof_pos_upper_limits[self._body_dof_idx]
        # )

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
        
        # 更新完整的body位置目标（包括body_yaw保持默认位置）
        self.pos_actions = self.robot.data.default_joint_pos[:, self._body_dof_idx].clone() # 重置为默认位置
        self.pos_actions[:, 1:] = new_arm_pos  # 更新数据
        self.pos_actions[:, 0] = new_body_pos  # 更新数据

        self._visualize_markers()

    #应用动作，更新的数据应用于物理模拟，为指定关节设置期望目标值
    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self.pos_actions, joint_ids=self._body_dof_idx) #设置目标位置
        self.robot.set_joint_velocity_target(self.vel_actions, joint_ids=self._wheel_dof_idx) #设置目标速度

    #获取观测
    def _get_observations(self) -> dict:
        # self.robot_lin_vel = self.robot.data.root_com_lin_vel_b
        # self.robot_ang_vel = self.robot.data.root_com_ang_vel_b  # 角速度
        
        # self.forwards = math_utils.quat_apply(self.robot.data.root_quat_w, self.robot.data.FORWARD_VEC_B)
        # dot = torch.sum(self.forwards * self.commands, dim=-1, keepdim=True)
        # cross = torch.cross(self.forwards, self.commands, dim=-1)[:,-1].reshape(-1,1)
        
        # # 重力向量在本体坐标系下的投影（用于检测底盘倾斜）
        # gravity_world = torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(self.num_envs, 1)
        # self.gravity_body = math_utils.quat_apply_inverse(self.robot.data.root_quat_w, gravity_world)
        
        # body_pos = self.robot.data.joint_pos[:, self._body_dof_idx]

        # obs = torch.hstack((
        #     self.robot_lin_vel[:, :2],   # xy平面线速度 [2维]
        #     dot,                         # 朝向与目标的点积 [1维]
        #     cross,                       # 朝向与目标的叉积 [1维]
        #     self.gravity_body[:, :2],    # 重力在本体坐标系xy平面的投影 [2维] - 反映倾斜程度
        #     body_pos,                    # 机体4个关节位置 [4维]
        # ))

        self.robot_lin_vel = self.robot.data.root_com_lin_vel_b
        self.robot_ang_vel = self.robot.data.root_com_ang_vel_b  # 角速度
        
        self.forwards = math_utils.quat_apply(self.robot.data.root_quat_w, self.robot.data.FORWARD_VEC_B)
        dot = torch.sum(self.forwards * self.commands, dim=-1, keepdim=True)
        cross = torch.cross(self.forwards, self.commands, dim=-1)[:,-1].reshape(-1,1)
        
        # 重力向量在本体坐标系下的投影（用于检测底盘倾斜）
        gravity_world = torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(self.num_envs, 1)
        self.gravity_body = math_utils.quat_apply_inverse(self.robot.data.root_quat_w, gravity_world)

        body_pos = self.robot.data.joint_pos[:, self._body_dof_idx]
        
        obs = torch.hstack((
            self.robot_lin_vel[:, :2],  # xy平面线速度 [2维]
            dot,                        # 朝向与目标的点积 [1维]
            cross,                      # 朝向与目标的叉积 [1维]
            self.gravity_body[:, :2],   # 重力在本体坐标系xy平面的投影 [2维] - 反映倾斜程度
            body_pos,                   # [4维]
        ))
        
        observations = {"policy": obs}
        return observations

    #获取奖励，计算函数compute_rewards见最后
    def _get_rewards(self) -> torch.Tensor:    
        # dot = torch.sum(self.forwards * self.commands, dim=-1, keepdim=True)
        # cross = torch.cross(self.forwards, self.commands, dim=-1)[:,-1].reshape(-1,1) 
        # yaw_error = torch.atan2(cross, dot)  # 偏航误差，范围[-π, π]
        # yaw_reward = torch.exp(-3.0 * torch.abs(yaw_error)).squeeze(-1)

        # forward_velocity = torch.sum(self.robot.data.root_lin_vel_b[:, :2] * self.commands[:, :2], dim=-1)
        # velocity_reward = torch.tanh(forward_velocity) * torch.clamp(dot.squeeze(), min=0.0)

        # # robot_lin_vel_b = self.robot.data.root_com_lin_vel_b[:, 0]
        # # backward_penalty = -torch.tanh(torch.clamp(-robot_lin_vel_b, min=0.0)) #后退惩罚
        # robot_lin_vel_b = self.robot.data.root_com_lin_vel_b[:, 0]
        # backward_penalty = -torch.clamp(robot_lin_vel_b, max=0.0).abs() #后退惩罚

        # pitch_tilt = torch.abs(self.gravity_body[:, 0])  # pitch方向倾斜
        # roll_tilt = torch.abs(self.gravity_body[:, 1])   # roll方向倾斜
        # pitch_penalty = -pitch_tilt  # 惩罚前后倾
        # roll_penalty = -roll_tilt    # 惩罚左右倾

        # # body_yaw = self.robot.data.joint_pos[:, self._body_dof_idx[0]]
        # # centering_penalty = -torch.abs(body_yaw) #惩罚身体不朝向指令方向
        # body_yaw = self.robot.data.joint_pos[:, self._body_dof_idx[0]]
        # centering_penalty = -torch.square(body_yaw)
        # # body_yaw = self.robot.data.joint_pos[:, self._body_dof_idx[0]]
        # # centering_reward = torch.exp(-5.0 * torch.abs(body_yaw)) #鼓励身体朝向指令方向

        # total_reward = (
        #     1.0 * yaw_reward * (3.0 * velocity_reward + 1.0)
        #     + 0.3 * backward_penalty
        #     + 0.5 * pitch_penalty
        #     + 0.5 * roll_penalty
        #     + 1.0 * centering_penalty
        # )

        dot = torch.sum(self.forwards * self.commands, dim=-1, keepdim=True)
        cross = torch.cross(self.forwards, self.commands, dim=-1)[:,-1].reshape(-1,1) 
        yaw_error = torch.atan2(cross, dot)  # 偏航误差，范围[-π, π]
        # yaw_reward = torch.exp(-1.0 * torch.abs(yaw_error)).squeeze(-1) #在误差接近180度时，停在原地，该附近的值极小且斜率极其平缓，不知道往哪边转
        
        abs_yaw_error = torch.abs(yaw_error).squeeze(-1)
        yaw_reward = 1.0 - (abs_yaw_error / math.pi) #线性斜率

        # yaw_reward_lin = 1.0 - (abs_yaw_error / math.pi)
        # yaw_reward_exp = torch.exp(-10.0 * abs_yaw_error)
        # yaw_reward = 0.7 * yaw_reward_lin + 0.3 * yaw_reward_exp


        ang_vel_z = self.robot.data.root_ang_vel_b[:, 2]
        turning_reward = torch.abs(ang_vel_z) * (1.0 - yaw_reward) # 越不对齐，转动奖励越高

        yaw_penalty = (-torch.exp(torch.abs(yaw_error)) + 1.0).squeeze(-1)

        forward_velocity = torch.sum(self.robot.data.root_lin_vel_b[:, :2] * self.commands[:, :2], dim=-1)
        heading_alignment = torch.sum(self.forwards[:, :2] * self.commands[:, :2], dim=-1)
        velocity_reward = torch.clamp(forward_velocity, 0, 1.0) * torch.clamp(heading_alignment, 0, 1.0)

        robot_lin_vel_b = self.robot.data.root_com_lin_vel_b[:, 0]
        backward_penalty = -torch.clamp(robot_lin_vel_b, max=0.0).abs() #后退惩罚

        pitch_tilt = torch.abs(self.gravity_body[:, 0])  # pitch方向倾斜 [0, 1.57]
        roll_tilt = torch.abs(self.gravity_body[:, 1])   # roll方向倾斜
        pitch_penalty = -pitch_tilt  # 惩罚前后倾
        roll_penalty = -roll_tilt    # 惩罚左右倾

        body_yaw = self.robot.data.joint_pos[:, self._body_dof_idx[0]] #[0, 3.14]
        centering_penalty = -torch.exp(torch.abs(body_yaw)) + 1.0
        centering_reward = torch.exp(-1.0 * torch.abs(body_yaw))


        total_reward = (
            # 1.0 * yaw_reward * (4.0*velocity_reward + 1.0)
            3.0 * yaw_reward
            # + 0.2 * yaw_penalty
            + 0.9 * velocity_reward
            + 0.3 * backward_penalty
            + 0.5 * pitch_penalty
            + 0.5 * roll_penalty
            + 0.4 * centering_penalty
            + 0.2 * centering_reward
        )
        
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

        #重置指令向量和可视化标记（只重置需要重置的环境）
        new_commands = torch.randn((len(env_ids), 3), device=self.device) #为需要重置的环境生成新指令
        new_commands[:, -1] = 0.0
        new_commands = new_commands / torch.linalg.norm(new_commands, dim=1, keepdim=True) #归一化
        self.commands[env_ids] = new_commands
        self.yaws[env_ids] = torch.atan2(new_commands[:, 1], new_commands[:, 0]).unsqueeze(1)
        self._visualize_markers()

        #重置环境参数流程：获取默认初始状态 -> 调整位置到环境原点 -> 写入模拟器
        joint_pos = self.robot.data.default_joint_pos[env_ids] #获取默认关节位置
        self.robot.write_joint_position_to_sim(joint_pos, None, env_ids)

        default_root_state = self.robot.data.default_root_state[env_ids].clone() #获取默认根状态
        # default_root_state[:, :3] += self.scene.env_origins[env_ids] #重置在环境生成的原点
        default_root_state[:, :3] += self._terrain.env_origins[env_ids] #重置在地形生成的原点
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