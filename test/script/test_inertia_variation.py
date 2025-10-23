#!/usr/bin/env python3
"""
测试质量矩阵M(q)是否随关节位置变化

目的：验证转动惯量是否随qpos变化，从而确定应该给出范围还是单一值
方法：让摆杆在MuJoCo中实际旋转，实时记录转动惯量变化
"""

import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
import time

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建简单的单关节摆杆模型
xml_string = """
<mujoco model="test_inertia">
  <option timestep="0.001" gravity="0 0 -9.81"/>
  
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <light directional="true" pos="0 0 3" dir="0 0 -1"/>
    
    <body name="base" pos="0 0 0.03">
      <geom type="cylinder" size="0.075 0.03" rgba="0.3 0.3 0.3 1"/>
      
      <body name="link1" pos="0 0 0.07">
        <joint name="joint1" type="hinge" axis="0 0 1" limited="false" damping="0.1"/>
        <geom type="cylinder" size="0.015 0.04" rgba="0.6 0.3 0.1 1"/>
        <inertial pos="0 0 0" mass="0.5" diaginertia="0.001 0.001 0.0001"/>
        
        <body name="arm" pos="0.1 0 0.04">
          <geom type="cylinder" size="0.015 0.1" 
                quat="0.707107 0 0.707107 0" rgba="0.1 0.5 0.8 1"/>
          <inertial pos="0 0 0" mass="1.0" 
                    diaginertia="0.00167 0.00167 0.000113"/>
          
          <body name="load" pos="0.1 0 0">
            <geom type="sphere" size="0.04" rgba="0.8 0.1 0.1 1"/>
            <inertial pos="0 0 0" mass="5.0" 
                      diaginertia="0.0032 0.0032 0.0032"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  
  <actuator>
    <motor name="motor1" joint="joint1" gear="1" ctrllimited="true" ctrlrange="-100 100"/>
  </actuator>
</mujoco>
"""

# 创建模型
model = mujoco.MjModel.from_xml_string(xml_string)
data = mujoco.MjData(model)

print("="*70)
print("测试：质量矩阵M(q)是否随关节位置变化")
print("="*70)
print("方法：让摆杆在MuJoCo中实际旋转，实时记录转动惯量")
print("控制：使用正弦信号让摆杆来回摆动")
print("="*70)

# 记录数据
time_data = []
position_data = []
velocity_data = []
inertia_data = []

# 测试参数
test_duration = 10.0  # 测试10秒
amplitude = np.deg2rad(60)  # 摆动幅度 ±60度
frequency = 0.5  # 摆动频率 0.5Hz

print(f"\n测试参数:")
print(f"  持续时间: {test_duration} 秒")
print(f"  摆动幅度: ±60°")
print(f"  摆动频率: {frequency} Hz")
print(f"\n开始仿真...")

# 运行仿真
with mujoco.viewer.launch_passive(model, data) as viewer:
    # 设置相机视角
    viewer.cam.azimuth = 45
    viewer.cam.elevation = -20
    viewer.cam.distance = 0.8
    viewer.cam.lookat = np.array([0.0, 0.0, 0.2])
    
    while viewer.is_running() and data.time < test_duration:
        step_start = time.time()
        
        t = data.time
        
        # 正弦位置控制
        pos_des = amplitude * np.sin(2 * np.pi * frequency * t)
        
        # PD控制
        kp = 100.0
        kd = 10.0
        torque = kp * (pos_des - data.qpos[0]) - kd * data.qvel[0]
        torque = np.clip(torque, -100, 100)
        
        data.ctrl[0] = torque
        
        # 计算当前转动惯量
        mujoco.mj_forward(model, data)
        M = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, M, data.qM)
        inertia = M[0, 0]
        
        # 记录数据
        time_data.append(t)
        position_data.append(data.qpos[0])
        velocity_data.append(data.qvel[0])
        inertia_data.append(inertia)
        
        # 仿真步进
        mujoco.mj_step(model, data)
        viewer.sync()
        
        # 实时控制
        time_until_next = model.opt.timestep - (time.time() - step_start)
        if time_until_next > 0:
            time.sleep(time_until_next)

print("仿真完成！")

# 转换为numpy数组
time_data = np.array(time_data)
position_data = np.array(position_data)
velocity_data = np.array(velocity_data)
inertias = np.array(inertia_data)

# 统计分析
print("\n" + "="*70)
print("统计分析:")
print("="*70)
print(f"最小转动惯量: {np.min(inertias):.8f} kg·m²")
print(f"最大转动惯量: {np.max(inertias):.8f} kg·m²")
print(f"平均转动惯量: {np.mean(inertias):.8f} kg·m²")
print(f"变化范围: {np.max(inertias) - np.min(inertias):.8f} kg·m²")
print(f"相对变化率: {(np.max(inertias) - np.min(inertias)) / np.mean(inertias) * 100:.2f}%")

# 判断是否需要给出范围
if (np.max(inertias) - np.min(inertias)) / np.mean(inertias) > 0.01:  # 变化超过1%
    print("\n⚠️ 结论: 转动惯量随关节位置显著变化！")
    print("   建议给结构部门提供转动惯量范围: [{:.6f}, {:.6f}] kg·m²".format(
        np.min(inertias), np.max(inertias)))
else:
    print("\n✓ 结论: 转动惯量基本不变")
    print("   可以给结构部门提供单一值: {:.6f} kg·m²".format(np.mean(inertias)))

# 绘制结果
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# 子图1: 关节位置
axes[0].plot(time_data, np.rad2deg(position_data), 'b-', linewidth=1.5, label='关节位置')
axes[0].set_ylabel('位置 (度)', fontsize=11)
axes[0].set_title('关节运动轨迹', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# 子图2: 转动惯量随时间变化
axes[1].plot(time_data, inertias, 'r-', linewidth=1.5, label='转动惯量')
axes[1].axhline(y=np.mean(inertias), color='g', linestyle='--', linewidth=2, 
                label=f'平均值: {np.mean(inertias):.6f}')
axes[1].fill_between(time_data, np.min(inertias), np.max(inertias), 
                      alpha=0.2, color='orange', 
                      label=f'变化范围: [{np.min(inertias):.6f}, {np.max(inertias):.6f}]')
axes[1].set_ylabel('转动惯量 (kg·m²)', fontsize=11)
axes[1].set_title('转动惯量随时间变化', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].legend()

# 子图3: 转动惯量 vs 关节位置
axes[2].scatter(np.rad2deg(position_data), inertias, c=time_data, 
                cmap='viridis', s=1, alpha=0.5)
axes[2].set_xlabel('关节位置 (度)', fontsize=11)
axes[2].set_ylabel('转动惯量 (kg·m²)', fontsize=11)
axes[2].set_title('转动惯量 vs 关节位置', fontsize=12, fontweight='bold')
axes[2].grid(True, alpha=0.3)
cbar = plt.colorbar(axes[2].collections[0], ax=axes[2])
cbar.set_label('时间 (s)', fontsize=10)

plt.tight_layout()
plt.savefig('inertia_variation_analysis.png', dpi=150)
print(f"\n图表已保存: inertia_variation_analysis.png")
plt.show()

print("\n" + "="*70)
print("解释:")
print("="*70)
print("""
对于单关节旋转系统（绕Z轴旋转）：
- 如果所有物体都在旋转平面内（XY平面），且质量分布关于Z轴对称
  → 转动惯量不随qpos变化（因为始终绕Z轴旋转）

- 如果物体有偏心质量或不对称结构
  → 转动惯量会随qpos变化（因为质量分布相对旋转轴的距离改变）

本测试模型：
- 摆杆在XY平面内旋转，绕Z轴
- 负载在摆杆末端，形成偏心质量
- 但因为是绕Z轴旋转，所以转动惯量理论上应该不变

如果转动惯量确实变化，说明：
1. 模型中存在重力影响（但不影响转动惯量本身）
2. 数值计算误差
3. 或者模型结构导致的实际变化
""")

