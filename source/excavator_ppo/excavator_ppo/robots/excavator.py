import isaaclab.sim as sim_utils #接受USD资产，生成所需的SpawnCfg（用于指定仿真中定义机器人的USD资产）
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg #隐式执行器配置，PD控制

EXCAVATOR_CFG = ArticulationCfg(
    spawn = sim_utils.UsdFileCfg(
        usd_path="./excavator_ppo/source/excavator_ppo/excavator_ppo/robots/USD/excavator_JCB/excavator_JCB.usd",
        #USDFileCfg对刚性体和机器人等具有特殊参数
        #刚体属性
        rigid_props = sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False, #启用重力
            linear_damping=0, #线性阻尼，无空气阻力
            angular_damping=0, #角阻尼
            max_linear_velocity=1000.0, #最大线速度
            max_angular_velocity=1000.0, #最大角速度
            max_depenetration_velocity=1.0, #最大穿透速度：1m/s的速度来修正穿透
        ),
        #关节求解器属性
        Articulation_props = sim_utils.ArticulationRootPropertiesCfg(
            enable_self_collisions = True, #启用自碰撞
            solver_position_iteration_count = 8, #位置求解迭代次数，迭代调整物体位置，解决物体之间的穿透问题
            solver_velocity_iteration_count = 0, #速度求解迭代次数，解决约束相关的速度问题（如反弹、摩擦）：设置为0,物理引擎只解决穿透，不改变速度，速度完全由控制器决定
        ),
    ),

    init_state = ArticulationCfg.InitialStateCfg(
        pos = (0.0, 0.0, 0.5), #root_link初始位置（多环境）
        joint_pos = {
            "body_yaw_joint": 0.0,
            "boom_pitch_joint": 0.0,
            "forearm_pitch_joint": 0.0,
            "bucket_pitch_joint": 0.0,
        },
    ),
    
    #定义执行器（电机）
    actuators = {
        "body_joint": ImplicitActuatorCfg(
            joint_name="body_yaw_joint",
            stiffness=10000.0, #位置控制的刚度（N·m/rad）
            damping=1000.0, #阻尼（N·m·s/rad）
            effort_limit_sim=30000.0, #最大力矩/力（N·m）
            max_velocity=2.0, #rad/s
        ),
        "boom_joint": ImplicitActuatorCfg(
            joint_name="boom_pitch_joint",
            stiffness=80000.0,
            damping=8000.0,
            effort_limit_sim=120000.0,
            max_velocity=1.5,
        ),
        "forearm_joint": ImplicitActuatorCfg(
            joint_name="forearm_pitch_joint",
            stiffness=60000.0,
            damping=6000.0,
            effort_limit_sim=90000.0,
            max_velocity=1.8,
        ),
        "bucket_joint": ImplicitActuatorCfg(
            joint_name="bucket_pitch_joint",
            stiffness=40000.0,
            damping=4000.0,
            effort_limit_sim=60000.0,
            max_velocity=2.0,
        ),
    },
)
