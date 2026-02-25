from excavator_ppo.robots.excavator import EXCAVATOR_CFG  # 挖掘机机器人配置
# from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG

from isaaclab.assets import ArticulationCfg, RigidObjectCfg #机器人配置、刚体物体配置
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg #交互场景配置，环境数量、环境间距、物理复制
from isaaclab.sim import SimulationCfg #模拟配置，时间步长、渲染间隔
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.sensors import RayCasterCfg, patterns
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
    # obs = base_lin_vel(3) + base_ang_vel(3) + gravity(3) + commands(3)
    #     + arm_joint_pos(4) + arm_joint_vel(4) + wheel_vel(6)
    #     + actions(6) + heights(221)
    observation_space = 253
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

    # ────── 高度扫描传感器（RayCaster）──────
    # 挂载在 base_link，向前偏移 1m，从 20m 高处向下射线测量地形高度
    # 覆盖 8m(纵向) × 6m(横向)，分辨率 0.5m → 17×13 = 221 个采样点
    # 前方可见距离 ~5m，后方 ~3m，给策略足够前瞻规划时间
    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(1.0, 0.0, 20.0)),  # 向前偏移 1m
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.5, size=[8.0, 6.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # ────── 命令配置（4 维: lin_vel_x, lin_vel_y, ang_vel_yaw, heading）──────
    num_commands = 4
    heading_command = True      # heading 模式：从 heading 误差重新计算 ang_vel_yaw
    command_resampling_time = 10.0  # 命令重采样间隔 (s)
    lin_vel_x_range = [0.3, 1.5]    # 前进速度 (m/s)，挖掘机较慢
    lin_vel_y_range = [0.0, 0.0]    # 侧向速度（挖掘机不侧移）
    ang_vel_yaw_range = [0.0, 0.0]  # 偏航角速度范围（heading 模式下由误差重算）
    heading_range = [math.pi / 2, math.pi / 2]  # 始终指向 +y（π/2 rad）

    # 所有挖掘机沿 +y 方向在同一条宽赛道上依次穿越地形
    track_num_stages = 6          # 地形阶段数
    track_width = 80.0            # 赛道宽度 (m)，所有挖掘机共享
    track_section_length = 40.0   # 每段地形沿 y 方向长度 (m)，6段共240m长条道路
    track_difficulty = 0.5        # 地形难度（0.0~1.0），控制随机粗糙度、障碍密度/高度、波浪振幅、台阶高度等参数
    _border_width = 0.0          # 地形边界平坦区 (m)

    TRACK_TERRAINS_CFG = TerrainGeneratorCfg(
        size=(track_width, track_section_length),  # 每个 tile: 80m(x宽) × 40m(y长)
        border_width=_border_width,
        num_rows=1,                    # 赛道数量
        num_cols=track_num_stages,     # 地形段数
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        curriculum=True,
        difficulty_range=(track_difficulty, track_difficulty),
        sub_terrains={
            # 字典顺序决定列映射，勿调换 !!
            "flat_start": terrain_gen.MeshPlaneTerrainCfg(
                proportion=1.0 / 6,          # col 0 — 起点平地
            ),
            "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                proportion=1.0 / 6,          # col 1 — 随机粗糙地形
                noise_range=(0.02, 0.10),
                noise_step=0.02,
                border_width=0.25, 
            ),
            "discrete_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
                proportion=1.0 / 6,          # col 2 — 离散障碍物（纯凸起，需配合机械臂翻越）
                obstacle_height_mode="fixed", # 只生成凸起障碍物，无凹坑
                obstacle_height_range=(0.25, 0.50), # 障碍高度
                obstacle_width_range=(3.0, 5.0),     # 障碍宽度
                num_obstacles=50, 
                platform_width=3.0,
                border_width=0.25,
            ),
            "wave": terrain_gen.HfWaveTerrainCfg(
                proportion=1.0 / 6,          # col 3 — 波浪起伏（模拟工地土堆/缓坡山丘）
                amplitude_range=(0.10, 1.0), # 振幅
                num_waves=2,                  # 波周期
                border_width=0.25,
            ),
            "pyramid_stairs": terrain_gen.HfPyramidStairsTerrainCfg(
                proportion=1.0 / 6,          # col 4 — 金字塔台阶（模拟采矿阶梯/土方边坡）
                step_height_range=(0.08, 0.20),  # 台阶高度 0.08~0.20m，需要谨慎驾驶
                step_width=0.8,               # 台阶宽度 0.8m，提供足够轮距着陆面
                platform_width=2.5,           # 顶部平台 2.5m，给挖掘机足够转向空间
                border_width=0.25,
            ),
            "flat_end": terrain_gen.MeshPlaneTerrainCfg(
                proportion=1.0 / 6,          # col 5 — 终点平地
            ),
        },
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TRACK_TERRAINS_CFG,
        max_init_terrain_level=0,   # 不使用课程升降级
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="max",
            restitution_combine_mode="min",
            static_friction=1.5,
            dynamic_friction=1.2,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    body_dof_name = ["body_yaw_joint", "boom_pitch_joint", "forearm_pitch_joint", "bucket_pitch_joint"] #无所谓顺序，只是提供查询字典
    wheel_dof_name = ["left_wheel_joint", "left_front_wheel_joint", "left_behind_wheel_joint", "right_wheel_joint", "right_front_wheel_joint", "right_behind_wheel_joint"]
    left_wheel_dof_name = ["left_wheel_joint", "left_front_wheel_joint", "left_behind_wheel_joint"] #1、2、3
    right_wheel_dof_name = ["right_wheel_joint", "right_front_wheel_joint", "right_behind_wheel_joint"] #4、5、6
    
    # initial_angle_range = [0.0, 0.25]
    position_action_scale = 1.5  # 机械臂位置控制缩放
    body_yaw_scale = 1.0 # 身体旋转控制缩放
    action_scale = 1.5  # 履带速度控制缩放
    action_rate_scale = 0.01  # 动作平滑度惩罚系数
    arm_effort_scale = 0.01   # 机械臂能耗惩罚系数（弱惩罚，不阻止必要使用）

    # ────── 观测缩放因子（参考 Go2 四足机器人）──────
    lin_vel_scale = 2.0        # 线速度缩放
    ang_vel_scale = 0.25       # 角速度缩放
    dof_pos_scale = 1.0        # 关节位置缩放
    dof_vel_scale = 0.05       # 关节速度缩放
    height_scale = 5.0         # 高度测量缩放
    base_height_offset = 0.5   # 底盘高度偏移 (m)，用于计算相对地形高度

    ######### 静态平台配置（已禁用 — 复杂地形下平台可能与地形特征穿插）########
    # platform_offset = (0.0, -4, 0.2)
    # platform_cfg: RigidObjectCfg = RigidObjectCfg(
    #     prim_path="/World/envs/env_.*/Platform",
    #     spawn=sim_utils.CuboidCfg(
    #         size=(5.0, 6.0, 1.5),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=platform_offset),
    # )
    ########################################################