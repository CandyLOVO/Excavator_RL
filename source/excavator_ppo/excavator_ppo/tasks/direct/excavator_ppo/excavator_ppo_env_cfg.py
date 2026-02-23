from excavator_ppo.robots.excavator import EXCAVATOR_CFG  # 挖掘机机器人配置
# from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG

from isaaclab.assets import ArticulationCfg, RigidObjectCfg #机器人配置、刚体物体配置
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg #交互场景配置，环境数量、环境间距、物理复制
from isaaclab.sim import SimulationCfg #模拟配置，时间步长、渲染间隔
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
import isaaclab.terrains as terrain_gen
import isaaclab.sim as sim_utils
import math

@configclass
class ExcavatorPpoEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 30.0

    # - spaces definition
    action_space = 6
    observation_space = 10
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot(s)
    robot_cfg: ArticulationCfg = EXCAVATOR_CFG.replace(prim_path="/World/envs/env_.*/Robot") #替换所有副本路径

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024, 
        env_spacing=10.0, 
        replicate_physics=True, 
    )

    _num_envs = scene.num_envs if hasattr(scene, "num_envs") else 1
    _env_spacing = scene.env_spacing if hasattr(scene, "env_spacing") else 10.0

    _rows = int(math.ceil(math.sqrt(_num_envs)))
    _cols = int(math.ceil(_num_envs / _rows))

    _tile_size = max(8.0, float(_env_spacing) * 1.5)
    _border_width = 50.0  # 在地形网格外围添加平坦边界（米），防止挖掘机跑出后掉落
    
    ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
        size=(_tile_size, _tile_size),
        border_width=_border_width,  # 平坦边界宽度
        num_rows=_rows,
        num_cols=_cols,
        horizontal_scale=0.2,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        sub_terrains={
            "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                proportion=1.0, noise_range=(0.02, 0.08), noise_step=0.02, border_width=0
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
            friction_combine_mode="max", #两个接触体有不同摩擦系数时，合成策略采用乘法（常见选项：average, min, max, multiply 等）。multiply 会把两个摩擦值相乘，结果通常更小/更大取决于值
            restitution_combine_mode="min", #弹性系数合成模式
            static_friction=1.5, #静摩擦系数
            dynamic_friction=1.2, #动摩擦系数
            restitution=0.0, #弹性系数
        ),
        debug_vis=False, #是否创建并显示 terrain origins（子地形原点 / env spawn 点）等调试标记
    )

    body_dof_name = ["body_yaw_joint", "boom_pitch_joint", "forearm_pitch_joint", "bucket_pitch_joint"] #无所谓顺序，只是提供查询字典
    wheel_dof_name = ["left_wheel_joint", "left_front_wheel_joint", "left_behind_wheel_joint", "right_wheel_joint", "right_front_wheel_joint", "right_behind_wheel_joint"]
    left_wheel_dof_name = ["left_wheel_joint", "left_front_wheel_joint", "left_behind_wheel_joint"] #1、2、3
    right_wheel_dof_name = ["right_wheel_joint", "right_front_wheel_joint", "right_behind_wheel_joint"] #4、5、6
    
    # initial_angle_range = [0.0, 0.25]
    position_action_scale = 1.5  # 机械臂位置控制缩放
    body_yaw_scale = 1.0 # 身体旋转控制缩放
    action_scale = 1.5  # 履带速度控制缩放

    ##################### 静态平台配置 ######################
    platform_offset = (0.0, -4, 0.2)  # 平台相对于环境原点的偏移
    platform_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Platform",
        spawn=sim_utils.CuboidCfg(
            size=(5.0, 6.0, 1.5),  # 尺寸：长5m x 宽6m x 高1.5m
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,  # 设置为运动学物体（静态，不受物理影响）
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),  # 启用碰撞
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.5, 0.5, 0.5),  # 灰色外观
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=platform_offset,  # 初始位置将在环境中根据env原点调整
        ),
    )
    ########################################################