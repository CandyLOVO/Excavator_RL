import isaaclab.sim as sim_utils #接受USD资产，生成所需的SpawnCfg（用于指定仿真中定义机器人的USD资产）
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg #隐式执行器配置，PD控制

EXCAVATOR_CFG = ArticulationCfg(
    spawn = sim_utils.UsdFileCfg(
        usd_path="source/excavator_ppo/excavator_ppo/robots/USD/excavator_six_wheels/excavator_six_wheels.usd",
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
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, 
            solver_position_iteration_count=4, 
            solver_velocity_iteration_count=4,
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
            joint_names_expr=["body_yaw_joint"],
            stiffness=10000.0, #位置控制的刚度（N·m/rad）
            damping=1000.0, #阻尼（N·m·s/rad）
            effort_limit_sim=30000.0, #最大力矩/力（N·m）
            velocity_limit_sim=2.0, #rad/s
        ),
        "boom_joint": ImplicitActuatorCfg(
            joint_names_expr=["boom_pitch_joint"],
            stiffness=80000.0,
            damping=8000.0,
            effort_limit_sim=120000.0,
            velocity_limit_sim=1.5,
        ),
        "forearm_joint": ImplicitActuatorCfg(
            joint_names_expr=["forearm_pitch_joint"],
            stiffness=60000.0,
            damping=6000.0,
            effort_limit_sim=90000.0,
            velocity_limit_sim=1.8,
        ),
        "bucket_joint": ImplicitActuatorCfg(
            joint_names_expr=["bucket_pitch_joint"],
            stiffness=40000.0,
            damping=4000.0,
            effort_limit_sim=60000.0,
            velocity_limit_sim=2.0,
        ),
        "wheel_joints": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_wheel_joint",
                "right_wheel_joint",
                "left_front_wheel_joint",
                "right_front_wheel_joint",
                "left_behind_wheel_joint",
                "right_behind_wheel_joint",
            ],
            stiffness=0.0,       # 必须为 0，代表不控制位置
            damping=10000.0,     # 设置较大的阻尼，它在速度模式下充当 P 增益
            effort_limit=500000.0,  # 给一个极大的力矩限制（50万 N·m），确保推得动重型底盘
            velocity_limit=5.0,
        ),
    },
)
