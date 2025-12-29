from excavator_ppo.robots.excavator import EXCAVATOR_CFG  # 挖掘机机器人配置

from isaaclab.assets import ArticulationCfg #机器人配置
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg #交互场景配置，环境数量、环境间距、物理复制
from isaaclab.sim import SimulationCfg #模拟配置，时间步长、渲染间隔
from isaaclab.utils import configclass

@configclass
class ExcavatorPpoEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 20.0

    # - spaces definition
    action_space = 2
    observation_space = 16
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot(s)
    robot_cfg: ArticulationCfg = EXCAVATOR_CFG.replace(prim_path="/World/envs/env_.*/Robot") #替换所有副本路径

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=2048, env_spacing=15.0, replicate_physics=True)
    body_dof_name = ["body_yaw_joint", "boom_pitch_joint", "forearm_pitch_joint", "bucket_pitch_joint"] #无所谓顺序，只是提供查询字典
    wheel_dof_name = ["left_wheel_joint", "left_front_wheel_joint", "left_behind_wheel_joint", "right_wheel_joint", "right_front_wheel_joint", "right_behind_wheel_joint"]
    left_wheel_dof_name = ["left_wheel_joint", "left_front_wheel_joint", "left_behind_wheel_joint"] #1、2、3
    right_wheel_dof_name = ["right_wheel_joint", "right_front_wheel_joint", "right_behind_wheel_joint"] #4、5、6
    
    # initial_angle_range = [0.0, 0.25]
    position_action_scale = 2.3
    action_scale = 7.5  # 增加动力，确保挖掘机能够移动和转向