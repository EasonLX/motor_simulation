#!/usr/bin/env python3
"""
MuJoCo质量矩阵M(q)详解
解释为什么可以直接从M(q)提取转动惯量
"""

import mujoco
import numpy as np
import matplotlib.pyplot as plt

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def create_simple_model():
    """创建简单的单关节模型"""
    xml = """
<mujoco model="mass_matrix_demo">
  <option timestep="0.001" gravity="0 0 -9.81"/>
  
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    
    <body name="base" pos="0 0 0.03">
      <geom type="cylinder" size="0.075 0.03"/>
      
      <body name="link1" pos="0 0 0.07">
        <joint name="joint1" type="hinge" axis="0 0 1" limited="false" damping="0"/>
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
    return mujoco.MjModel.from_xml_string(xml)

def demonstrate_mass_matrix():
    """演示质量矩阵的物理意义"""
    
    print("="*80)
    print("MuJoCo质量矩阵M(q)详解")
    print("="*80)
    
    model = create_simple_model()
    data = mujoco.MjData(model)
    
    print("\n1. 质量矩阵的物理意义")
    print("-" * 50)
    print("对于单关节系统，动力学方程为：")
    print("  M(q) * q̈ + C(q,q̇) + G(q) = τ")
    print("其中 M(q) 是 1×1 矩阵，M[0,0] = 系统绕关节轴的转动惯量")
    
    # 在不同关节位置计算质量矩阵
    positions = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
    
    print(f"\n2. 不同关节位置下的质量矩阵M(q)")
    print("-" * 50)
    print(f"{'关节位置(rad)':<15} {'关节位置(度)':<15} {'M[0,0] (kg·m²)':<20} {'说明'}")
    print("-" * 70)
    
    for pos in positions:
        data.qpos[0] = pos
        data.qvel[0] = 0
        data.qacc[0] = 0
        
        # 前向动力学计算
        mujoco.mj_forward(model, data)
        
        # 计算质量矩阵
        M = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, M, data.qM)
        
        inertia = M[0, 0]
        print(f"{pos:<15.3f} {np.rad2deg(pos):<15.1f} {inertia:<20.8f} {'转动惯量'}")
    
    print(f"\n3. 质量矩阵与转动惯量的关系")
    print("-" * 50)
    print("M(q)包含了整个系统在当前配置下的有效转动惯量！")
    print("这意味着：")
    print("  - M[0,0] = 系统绕关节轴的转动惯量")
    print("  - 包括所有刚体的贡献")
    print("  - 考虑了当前关节位置的影响")
    print("  - 可以直接用于动力学计算")

def demonstrate_physics():
    """演示物理意义"""
    
    print(f"\n4. 物理验证：扭矩 = 转动惯量 × 角加速度")
    print("-" * 50)
    
    model = create_simple_model()
    data = mujoco.MjData(model)
    
    # 设置水平位置（消除重力影响）
    data.qpos[0] = np.pi/2
    data.qvel[0] = 0
    
    # 计算质量矩阵
    mujoco.mj_forward(model, data)
    M = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, M, data.qM)
    inertia = M[0, 0]
    
    print(f"系统转动惯量（从M(q)提取）: {inertia:.8f} kg·m²")
    
    # 施加单位角加速度
    target_acc = 1.0  # rad/s²
    data.qacc[0] = target_acc
    
    # 使用逆动力学计算所需扭矩
    mujoco.mj_inverse(model, data)
    required_torque = data.qfrc_inverse[0]
    
    print(f"施加角加速度: {target_acc} rad/s²")
    print(f"所需扭矩: {required_torque:.8f} Nm")
    print(f"验证: τ = I × α = {inertia:.8f} × {target_acc} = {inertia * target_acc:.8f} Nm")
    print(f"误差: {abs(required_torque - inertia * target_acc):.10f} Nm")
    
    if abs(required_torque - inertia * target_acc) < 1e-6:
        print("✓ 验证通过！质量矩阵确实等于转动惯量")
    else:
        print("⚠️ 存在误差，可能由于数值精度或重力影响")

def demonstrate_configuration_dependency():
    """演示质量矩阵对配置的依赖性"""
    
    print(f"\n5. 质量矩阵的配置依赖性")
    print("-" * 50)
    print("对于多关节系统，M(q)会随关节位置变化")
    print("但对于我们的单关节系统，M(q)是常数")
    
    model = create_simple_model()
    data = mujoco.MjData(model)
    
    # 测试不同位置
    positions = np.linspace(0, 2*np.pi, 10)
    inertias = []
    
    for pos in positions:
        data.qpos[0] = pos
        data.qvel[0] = 0
        data.qacc[0] = 0
        
        mujoco.mj_forward(model, data)
        M = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, M, data.qM)
        inertias.append(M[0, 0])
    
    inertias = np.array(inertias)
    
    print(f"关节位置范围: 0 ~ 2π rad")
    print(f"转动惯量范围: {inertias.min():.8f} ~ {inertias.max():.8f} kg·m²")
    print(f"变化幅度: {(inertias.max() - inertias.min()):.10f} kg·m²")
    
    if np.allclose(inertias, inertias[0], rtol=1e-10):
        print("✓ 单关节系统：转动惯量为常数")
    else:
        print("⚠️ 转动惯量随位置变化（多关节系统特征）")

def compare_with_manual_calculation():
    """与手动计算对比"""
    
    print(f"\n6. 与手动计算对比")
    print("-" * 50)
    
    # 手动计算（平行轴定理）
    link1_mass = 0.5
    link1_inertia = 0.0001
    
    arm_mass = 0.2
    arm_length = 0.4
    arm_center = 0.2
    arm_inertia = (1/12) * arm_mass * arm_length**2 + arm_mass * arm_center**2
    
    weight_mass = 2.0
    weight_distance = 0.4
    weight_inertia = weight_mass * weight_distance**2
    
    manual_total = link1_inertia + arm_inertia + weight_inertia
    
    print(f"手动计算（平行轴定理）:")
    print(f"  Link1: {link1_inertia:.8f} kg·m²")
    print(f"  摆杆:  {arm_inertia:.8f} kg·m²")
    print(f"  重量块: {weight_inertia:.8f} kg·m²")
    print(f"  总计:  {manual_total:.8f} kg·m²")
    
    # MuJoCo计算
    model = create_simple_model()
    data = mujoco.MjData(model)
    
    data.qpos[0] = 0
    data.qvel[0] = 0
    data.qacc[0] = 0
    
    mujoco.mj_forward(model, data)
    M = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, M, data.qM)
    mujoco_inertia = M[0, 0]
    
    print(f"\nMuJoCo计算（质量矩阵）:")
    print(f"  M[0,0]: {mujoco_inertia:.8f} kg·m²")
    
    error = abs(manual_total - mujoco_inertia) / manual_total * 100
    print(f"\n对比结果:")
    print(f"  误差: {error:.6f}%")
    
    if error < 1.0:
        print("✓ 两种方法结果一致！")
    else:
        print("⚠️ 存在差异，需要检查计算")

def main():
    """主函数"""
    
    print("MuJoCo质量矩阵M(q)与转动惯量的关系")
    print("="*80)
    
    try:
        demonstrate_mass_matrix()
        demonstrate_physics()
        demonstrate_configuration_dependency()
        compare_with_manual_calculation()
        
        print(f"\n" + "="*80)
        print("总结")
        print("="*80)
        print("1. M(q)是系统的质量矩阵，包含有效转动惯量")
        print("2. 对于单关节系统，M[0,0] = 系统绕关节轴的转动惯量")
        print("3. 可以直接从M(q)提取转动惯量，无需手动计算")
        print("4. 这是MuJoCo自动计算的结果，考虑了所有刚体贡献")
        print("5. 与手动平行轴定理计算应该一致")
        print("="*80)
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
