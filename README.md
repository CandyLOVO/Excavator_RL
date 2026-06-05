# 无人挖掘机强化学习协同越障

## 项目概述

本项目面向灾害救援、矿山开采等复杂地形，提出一种基于深度强化学习的无人挖掘机自主通行方法。通过将履带底盘与机械臂协同控制，挖掘机能自主完成铲斗支撑、机身抬升和越障动作，无需预设轨迹。实验证明，该方法显著提升了复杂地形的通行成功率和机身稳定性，为工程机械无人化提供了可行方案。

### 挖掘机模型

<p align="center">
  <img src="https://github.com/CandyLOVO/Excavator_RL/blob/main/excavator.png" alt="excavator" width="60%">
</p>

### 地形设计

由6段地形依次构成的复合赛道，总长120米，模拟野外泥土或碎石路面特性。

挖掘机在全流程中，依次通过起点平地、随机粗糙地形、金字塔台阶、正弦波浪地形、离散随机障碍地形，最终到达终点平地视为任务完成，并给予稀疏奖励。

<p align="center">
  <img src="https://github.com/CandyLOVO/Excavator_RL/blob/main/%E5%AE%8C%E6%95%B4%E5%9C%B0%E5%BD%A2%E5%9B%BE-%E6%96%9C%E5%90%91.png" alt="terrain" width="60%">
</p>

## 研究思路

本研究将挖掘机底盘与机械臂统一纳入强化学习框架，构建261维观测空间（含机身状态、铲斗接触力、局部地形高度图）与6维连续动作空间。通过设计受困识别门控与多条件乘积越障奖励，使策略网络仅在底盘真实受阻时激活机械臂辅助。模型根据实时地形感知，自主决策铲斗支撑点位与下压力度，协调履带差速与关节运动，实现“感知—判断—支撑—抬升—跨越”的完整越障闭环。

## 研究成果

在包含台阶、沟壑、随机障碍物的复合地形测试中，该方法的全流程通行成功率达到60.58%，相比无机械臂辅助方案（0%）与无门控方案（36.83%）均有显著提升。机身平均侧倾角由20.48°降至16.98°，峰值侧倾角由70.72°降至47.59%。挖掘机能够自主完成铲斗支撑、底盘抬升与履带跟进的连贯越障动作，验证了协同控制策略的有效性。

## 效果展示

### 越障视频

通过随机粗糙地形、金字塔台阶地形

<p align="center">
  <img src="https://github.com/CandyLOVO/Excavator_RL/blob/main/Demonstration%20Video1.gif" alt="Demonstration Video1">
</p>

---

通过波浪起伏地形、离散障碍物地形

<p align="center">
  <img src="https://github.com/CandyLOVO/Excavator_RL/blob/main/Demonstration%20Video2.gif" alt="Demonstration Video2">
</p>

### 奖励曲线

<p align="center">
  <img src="https://github.com/CandyLOVO/Excavator_RL/blob/main/all_terrains_mean_reward.png" alt="all_terrains_mean_reward" width="80%">
</p>

---

蓝线：去除机械臂有关奖励函数，仅保留底盘行驶相关奖励。

<p align="center">
  <img src="https://github.com/CandyLOVO/Excavator_RL/blob/main/%E6%97%A0%E6%9C%BA%E6%A2%B0%E8%87%82%E5%A5%96%E5%8A%B1.png" alt="without robotic arm rewards" width="80%">
</p>

---

绿线：使用6维联合控制，去掉受困识别门控，越障奖励始终激活。

<p align="center">
  <img src="https://github.com/CandyLOVO/Excavator_RL/blob/main/%E6%97%A0%E8%B6%8A%E9%9A%9C%E9%97%A8%E6%8E%A7.png" alt="without obstacle crossing rewards" width="80%">
</p>
