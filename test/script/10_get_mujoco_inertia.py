#!/usr/bin/env python3
"""
获取MuJoCo计算的转动惯量 - 简单演示
"""

import mujoco
import numpy as np

def get_mujoco_inertia(model, data, joint_pos=0):
    """
    获取MuJoCo计算的转动惯量
    
    Parameters:
    - model: MuJoCo模型
    - data: MuJoCo数据
    - joint_pos: 关节位置（可选，默认0）
    
    Returns:
    - inertia: 系统绕关节轴的转动惯量 (kg·m²)
    """
    
    # 设置关节位置
    data.qpos[0] = joint_pos
    data.qvel[0] = 0
    data.qacc[0] = 0
    
    # 前向动力学计算
    mujoco.mj_forward(model, data)
    
    # 计算质量矩阵M(q)
    M = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, M, data.qM)
    
    # 对于单关节系统，M[0,0]就是转动惯量
    inertia = M[0, 0]
    
    return inertia

def demo_get_inertia():
    """演示如何获取转动惯量"""
    
    print("="*60)
    print("获取MuJoCo计算的转动惯量")
    print("="*60)
    
    # 创建简单模型
    xml = """
<mujoco model="inertia_demo">
  <option timestep="0.001" gravity="0 0 -9.81"/>
  
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    
    <body name="base" pos="0 0 0.03">
      <geom type="cylinder" size="0.075 0.03"/>
      
      <body name="link1" pos="0 0 0.07">
        <joint name="joint1" type="hinge" axis="0 0 1" limited="false"/>
        <geom type="cylinder" size="0.015 0.04"/>
        <inertial pos="0 0 0" mass="0.5" diaginertia="0.001 0.001 0.0001"/>
        
        <body name="arm" pos="0.2 0 0.04">
          <geom type="cylinder" size="0.015 0.2" quat="0.707107 0 0.707107 0"/>
          <inertial pos="0 0 0" mass="0.2" diaginertia="0.0027 0.0027 0.0001"/>
          
          <body name="weight" pos="0.2 0 0">
            <geom type="sphere" size="0.04"/>
            <inertial pos="0 0 0" mass="2.0" diaginertia="0.0013 0.0013 0.0013"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  
  <actuator>
    <motor name="motor1" joint="joint1" gear="1"/>
  </actuator>
</mujoco>
"""
    
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    
    print("\n方法1：直接获取转动惯量")
    print("-" * 40)
    
    # 获取转动惯量
    inertia = get_mujoco_inertia(model, data)
    print(f"系统转动惯量: {inertia:.8f} kg·m²")
    
    print("\n方法2：详细步骤")
    print("-" * 40)
    
    # 详细步骤
    data.qpos[0] = 0
    data.qvel[0] = 0
    data.qacc[0] = 0
    
    print("1. 设置关节状态...")
    print(f"   关节位置: {data.qpos[0]:.3f} rad")
    print(f"   关节速度: {data.qvel[0]:.3f} rad/s")
    print(f"   关节加速度: {data.qacc[0]:.3f} rad/s²")
    
    print("\n2. 执行前向动力学...")
    mujoco.mj_forward(model, data)
    print("   ✓ 前向动力学完成")
    
    print("\n3. 计算质量矩阵M(q)...")
    M = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, M, data.qM)
    print(f"   质量矩阵: M = [{M[0,0]:.8f}]")
    
    print("\n4. 提取转动惯量...")
    inertia_detailed = M[0, 0]
    print(f"   转动惯量: I = M[0,0] = {inertia_detailed:.8f} kg·m²")
    
    print("\n方法3：验证物理关系")
    print("-" * 40)
    
    # 验证：施加角加速度，测量所需扭矩
    target_acc = 1.0  # rad/s²
    data.qacc[0] = target_acc
    
    print(f"1. 施加角加速度: α = {target_acc} rad/s²")
    
    # 使用逆动力学计算所需扭矩
    mujoco.mj_inverse(model, data)
    required_torque = data.qfrc_inverse[0]
    
    print(f"2. 计算所需扭矩: τ = {required_torque:.8f} Nm")
    print(f"3. 验证关系: τ = I × α")
    print(f"   {required_torque:.8f} = {inertia:.8f} × {target_acc}")
    
    theoretical_torque = inertia * target_acc
    error = abs(required_torque - theoretical_torque)
    print(f"4. 误差: {error:.10f} Nm")
    
    if error < 1e-6:
        print("   ✓ 验证通过！")
    else:
        print("   ⚠️ 存在误差")
    
    print("\n" + "="*60)
    print("总结")
    print("="*60)
    print("获取MuJoCo转动惯量的步骤：")
    print("1. 设置关节状态: data.qpos[0] = position")
    print("2. 前向动力学: mujoco.mj_forward(model, data)")
    print("3. 计算质量矩阵: mujoco.mj_fullM(model, M, data.qM)")
    print("4. 提取转动惯量: inertia = M[0, 0]")
    print("\n关键点：")
    print("- M(q)[0,0] = 系统绕关节轴的转动惯量")
    print("- 这是MuJoCo自动计算的结果")
    print("- 考虑了所有刚体的贡献")
    print("- 与手动平行轴定理计算应该一致")
    print("="*60)

if __name__ == "__main__":
    demo_get_inertia()
