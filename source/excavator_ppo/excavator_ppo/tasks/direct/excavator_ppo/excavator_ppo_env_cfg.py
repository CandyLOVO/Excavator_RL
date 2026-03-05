from excavator_ppo.robots.excavator import EXCAVATOR_CFG  # 挖掘机机器人配置
# from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG

from isaaclab.assets import ArticulationCfg, RigidObjectCfg #机器人配置、刚体物体配置
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg #交互场景配置，环境数量、环境间距、物理复制
from isaaclab.sim import SimulationCfg, PhysxCfg  #模拟配置，时间步长、渲染间隔、物理引擎配置
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
    decimation = 2 #2个模拟时间步更新一次动作
    episode_length_s = 40.0  # 更长episode给挖掘机更多时间完成复杂地形

    # - spaces definition
    action_space = 6
    observation_space = 253
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physx=PhysxCfg(
            gpu_collision_stack_size=2**27,  # 128MB，修复 collisionStackSize overflow（默认2^26不够）
        ),
    )

    # robot(s)
    robot_cfg: ArticulationCfg = EXCAVATOR_CFG.replace(prim_path="/World/envs/env_.*/Robot") #替换所有副本路径

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024,  # RTX 4080 Laptop 12GB: 1024 环境平衡显存与吞吐
        env_spacing=10.0, 
        replicate_physics=True, 
    )

    # 高度扫描传感器（RayCaster）
    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(1.0, 0.0, 20.0)),  # 向前偏移 1m，高度 20m
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.5, size=[8.0, 6.0]), #分辨率0.5m，范围8m×6m，17*13个采样点
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )

    # command配置（4 维: lin_vel_x, lin_vel_y, ang_vel_yaw, heading）
    num_commands = 4
    heading_command = True      # heading 模式：从 heading 误差重新计算 ang_vel_yaw
    command_resampling_time = 10.0  # 命令重采样间隔 (s)
    lin_vel_x_range = [0.5, 1.5]    # 前进速度 (m/s)，与轮子最大线速~1.47m/s匹配
    lin_vel_y_range = [0.0, 0.0]    # 侧向速度（m/s)
    ang_vel_yaw_range = [0.0, 0.0]  # 偏航角速度范围（heading 模式下由误差重算）
    heading_range = [math.pi / 2, math.pi / 2]  # 目标航向 +y（π/2 rad）    
    heading_kp = 0.5            # 期望角速度的比例增益
    max_ang_vel = 1.0           # 期望角速度截断上限 (rad/s)

    # 地形配置
    track_num_stages = 6          # 地形阶段数
    track_width = 80.0            # 赛道宽度 (m)
    track_section_length = 40.0   # 每段地形沿 y 方向长度 (m)
    track_difficulty = 0.5        # 地形难度（0.0~1.0），控制随机粗糙度、障碍密度/高度、波浪振幅、台阶高度等参数
    _border_width = 0.0           # 地形边界平坦区 (m)

    TRACK_TERRAINS_CFG = TerrainGeneratorCfg(
        size=(track_width, track_section_length),  # 每段地形尺寸
        border_width=_border_width,
        num_rows=1,                    # 赛道数量
        num_cols=track_num_stages,     # 地形段数
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False, #关闭地形缓存，每次训练都生成新地形
        curriculum=True, #启用课程学习，随着训练进展逐渐增加地形难度
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
        terrain_type="generator", #使用地形生成器而非导入预制地形
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
    action_scale = 6.0  # 履带速度控制缩放，v_max = ω_limit × r_wheel = 6 rad/s × 0.245 m ≈ 1.47 m/s
    position_action_scale = 1.5  # 机械臂位置控制缩放，每步最大位置增量 = 1.0 × dt（1/120*2） × 1.5 = 0.025 rad -> 1.5 rad/s
    body_yaw_scale = 1.0  # 车体偏航控制缩放，每步最大偏航增量 = 1.0 × dt × 1.0 = 0.0167 rad -> 1.0 rad/s
    action_rate_scale = 0.01  # 动作平滑度惩罚系数
    arm_effort_scale = 0.01   # 机械臂能耗惩罚系数（弱惩罚，不阻止必要使用）

    # 观测缩放因子 scale ≈ 1/典型最大值，让观测大致归一化到 [-1, 1] 范围
    lin_vel_scale = 0.5        # 线速度缩放：v_max ≈ 1.47 m/s -> 0.74（可接受）
    ang_vel_scale = 0.25       # 角速度缩放：典型 ±4 rad/s -> ±1.0
    dof_pos_scale = 1.0        # 关节位置缩放
    dof_vel_scale = 0.5        # 机械臂关节速度缩放：max ≈ 2.0 -> 1.0
    wheel_vel_scale = 0.167    # 轮子速度缩放：velocity_limit = 6.0 rad/s -> 1.0（≈1/6）
    height_scale = 1.0         # 高度测量缩放（数据已 clip 到 [-1, 1]）
    base_height_offset = 0.5   # excavator.py 中的 init_state pos z 定义