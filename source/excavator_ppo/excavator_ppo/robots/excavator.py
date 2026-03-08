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
            linear_damping=0.1, #线性阻尼，防止无阻力飞行
            angular_damping=0.05, #角阻尼，抑制旋转漂移
            max_linear_velocity=10.0, #最大线速度（挖掘机合理上限）
            max_angular_velocity=20.0, #最大角速度
            max_depenetration_velocity=10.0, #最大穿透修正速度：10m/s的速度来修正穿透
        ),
        #关节求解器属性
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, 
            solver_position_iteration_count=8, 
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
            stiffness=100000.0, #位置控制的刚度（N·m/rad）
            damping=25000.0, #阻尼（N·m·s/rad），阻尼比≈0.6，接近临界阻尼
            effort_limit_sim=50000.0, #最大力矩50kN·m，防止反作用力矩翻车
            velocity_limit_sim=2.0, #rad/s
        ),
        "boom_joint": ImplicitActuatorCfg(
            joint_names_expr=["boom_pitch_joint"],
            stiffness=500000.0,
            damping=80000.0, #提高阻尼比到~0.6，减少振荡
            effort_limit_sim=200000.0, #200kN·m≈5×重力矩，足够快速追踪
            velocity_limit_sim=1.5,
        ),
        "forearm_joint": ImplicitActuatorCfg(
            joint_names_expr=["forearm_pitch_joint"],
            stiffness=400000.0,
            damping=50000.0, #提高阻尼比到~0.7
            effort_limit_sim=150000.0, #150kN·m≈8×重力矩
            velocity_limit_sim=1.8,
        ),
        "bucket_joint": ImplicitActuatorCfg(
            joint_names_expr=["bucket_pitch_joint"],
            stiffness=300000.0,
            damping=30000.0,
            effort_limit_sim=80000.0, #80kN·m≈25×重力矩，bucket惯量小无需更高
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
            stiffness=0.0,       # 必须为0，代表不控制位置
            damping=1000.0,      # 速度模式P增益：1000×8.2rad/s=8200N·m/轮，6轮共~200kN推力
            effort_limit=10000.0, # 每轮最大力矩 10000N·m，为峰值力矩8200留余量
            velocity_limit=8.5,  # 限制轮子最大角速度，v_max=8.2×0.245≈2.0m/s（留余量8.5）
        ),
    },
)
