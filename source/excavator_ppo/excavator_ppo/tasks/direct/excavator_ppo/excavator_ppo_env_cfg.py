from excavator_ppo.robots.excavator import EXCAVATOR_JCB_CFG  # 挖掘机机器人配置

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
    action_space = 5
    observation_space = 6
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot(s)
    robot_cfg: ArticulationCfg = EXCAVATOR_JCB_CFG.replace(prim_path="/World/envs/env_.*/Robot") #替换所有副本路径

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=500, env_spacing=6.0, replicate_physics=True)
    dof_name = ["body_yaw_joint", "boom_pitch_joint", "forearm_pitch_joint", "bucket_pitch_joint"]

    # initial_angle_range = [0.0, 0.25]