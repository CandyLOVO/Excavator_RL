from excavator_ppo.robots.excavator import EXCAVATOR_CFG  # 挖掘机机器人配置
# from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG

from isaaclab.assets import ArticulationCfg #机器人配置
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg #交互场景配置，环境数量、环境间距、物理复制
from isaaclab.sim import SimulationCfg #模拟配置，时间步长、渲染间隔
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
import isaaclab.terrains as terrain_gen
import isaaclab.sim as sim_utils

@configclass
class ExcavatorPpoEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 20.0

    # - spaces definition
    action_space = 6
    observation_space = 10
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
        size=(100.0, 100.0),
        border_width=0.0,
        num_rows=1,
        num_cols=1,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        sub_terrains={
            "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                proportion=1.0, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0
            ),
        },
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground", #USD stage中创建地形的根路径
        terrain_type="generator", #“程序生成器”生成多个子地形网络
        terrain_generator=ROUGH_TERRAINS_CFG, #每个子地形(size)8*8m，子地形行列数(num_rows, num_cols)10*20，子地形边界外延(border_width)20m
        max_init_terrain_level=5, #课程难度层级，为None，默认max_init_level=num_rows-1；设置为5意味着初始难度层级会从0～5之间随机抽取（若 num_rows > 6）
        collision_group=-1, #被设置为“与环境实例发生碰撞”的全局路径（例如 ground 要与所有 env 中的机器人发生碰撞，因此常设为 -1）
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply", #两个接触体有不同摩擦系数时，合成策略采用乘法（常见选项：average, min, max, multiply 等）。multiply 会把两个摩擦值相乘，结果通常更小/更大取决于值
            restitution_combine_mode="multiply", #弹性系数合成策略采用乘法
            static_friction=1.0, #静摩擦系数
            dynamic_friction=1.0, #动摩擦系数
        ),
        debug_vis=False, #是否创建并显示 terrain origins（子地形原点 / env spawn 点）等调试标记
    )

    # robot(s)
    robot_cfg: ArticulationCfg = EXCAVATOR_CFG.replace(prim_path="/World/envs/env_.*/Robot") #替换所有副本路径

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=2048, env_spacing=30.0, replicate_physics=True)
    body_dof_name = ["body_yaw_joint", "boom_pitch_joint", "forearm_pitch_joint", "bucket_pitch_joint"] #无所谓顺序，只是提供查询字典
    wheel_dof_name = ["left_wheel_joint", "left_front_wheel_joint", "left_behind_wheel_joint", "right_wheel_joint", "right_front_wheel_joint", "right_behind_wheel_joint"]
    left_wheel_dof_name = ["left_wheel_joint", "left_front_wheel_joint", "left_behind_wheel_joint"] #1、2、3
    right_wheel_dof_name = ["right_wheel_joint", "right_front_wheel_joint", "right_behind_wheel_joint"] #4、5、6
    
    # initial_angle_range = [0.0, 0.25]
    position_action_scale = 1.5  # 机械臂位置控制缩放
    body_yaw_scale = 1.0 # 身体旋转控制缩放
    action_scale = 1.5  # 履带速度控制缩放