#!/usr/bin/env python3
"""
可视化质量矩阵M(q)的物理意义
"""

import mujoco
import numpy as np
import matplotlib.pyplot as plt

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def create_model_with_variable_inertia():
    """创建可变惯量的模型"""
    xml = """
<mujoco model="variable_inertia">
  <option timestep="0.001" gravity="0 0 -9.81"/>
  
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    
    <body name="base" pos="0 0 0.03">
      <geom type="cylinder" size="0.075 0.03"/>
      
      <body name="link1" pos="0 0 0.07">
        <joint name="joint1" type="hinge" axis="0 0 1" limited="false" damping="0"/>
        <geom type="cylinder" size="0.015 0.04"/>
        <inertial pos="0 0 0" mass="0.5" diaginertia="0.001 0.001 0.0001"/>
        
        <body name="arm" pos="0.15 0 0.04">
          <geom type="cylinder" size="0.015 0.15" quat="0.707107 0 0.707107 0"/>
          <inertial pos="0 0 0" mass="0.15" diaginertia="0.0011 0.0011 0.0001"/>
          
          <body name="weight" pos="0.15 0 0">
            <geom type="sphere" size="0.04"/>
            <inertial pos="0 0 0" mass="1.0" diaginertia="0.0006 0.0006 0.0006"/>
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

def plot_mass_matrix_vs_position():
    """绘制质量矩阵随位置的变化"""
    
    model = create_model_with_variable_inertia()
    data = mujoco.MjData(model)
    
    # 测试不同关节位置
    positions = np.linspace(0, 2*np.pi, 50)
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
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 图1：质量矩阵随位置变化
    ax1.plot(positions, inertias, 'b-', linewidth=2, label='M[0,0] (转动惯量)')
    ax1.set_xlabel('关节位置 (rad)', fontsize=12)
    ax1.set_ylabel('转动惯量 (kg·m²)', fontsize=12)
    ax1.set_title('质量矩阵M(q)随关节位置的变化', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 添加关键位置标记
    key_positions = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
    for pos in key_positions:
        ax1.axvline(x=pos, color='r', linestyle='--', alpha=0.5)
        ax1.text(pos, inertias.max()*0.9, f'{pos:.1f}', rotation=90, ha='right')
    
    # 图2：扭矩-加速度关系验证
    test_positions = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
    test_accelerations = [0.5, 1.0, 1.5, 2.0]
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(test_positions)))
    
    for i, pos in enumerate(test_positions):
        data.qpos[0] = pos
        data.qvel[0] = 0
        
        # 计算该位置的质量矩阵
        mujoco.mj_forward(model, data)
        M = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, M, data.qM)
        inertia = M[0, 0]
        
        # 测试不同加速度下的扭矩
        torques = []
        for acc in test_accelerations:
            data.qacc[0] = acc
            mujoco.mj_inverse(model, data)
            torque = data.qfrc_inverse[0]
            torques.append(torque)
        
        # 绘制扭矩-加速度关系
        theoretical_torques = [inertia * acc for acc in test_accelerations]
        
        ax2.plot(test_accelerations, torques, 'o-', color=colors[i], 
                label=f'位置 {pos:.2f} rad', linewidth=2, markersize=6)
        ax2.plot(test_accelerations, theoretical_torques, '--', color=colors[i], alpha=0.7)
    
    ax2.set_xlabel('角加速度 (rad/s²)', fontsize=12)
    ax2.set_ylabel('所需扭矩 (Nm)', fontsize=12)
    ax2.set_title('扭矩 = 转动惯量 × 角加速度 验证', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('mass_matrix_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return inertias

def demonstrate_physical_meaning():
    """演示物理意义"""
    
    print("="*80)
    print("质量矩阵M(q)的物理意义演示")
    print("="*80)
    
    model = create_model_with_variable_inertia()
    data = mujoco.MjData(model)
    
    # 选择一个位置进行详细分析
    test_position = np.pi/4
    data.qpos[0] = test_position
    data.qvel[0] = 0
    data.qacc[0] = 0
    
    print(f"\n测试位置: {test_position:.3f} rad ({np.rad2deg(test_position):.1f}°)")
    
    # 计算质量矩阵
    mujoco.mj_forward(model, data)
    M = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, M, data.qM)
    inertia = M[0, 0]
    
    print(f"质量矩阵M(q):")
    print(f"  M = [{M[0,0]:.8f}]")
    print(f"  转动惯量 = {inertia:.8f} kg·m²")
    
    # 验证物理关系
    print(f"\n物理验证:")
    print(f"  施加角加速度 α = 1.0 rad/s²")
    
    data.qacc[0] = 1.0
    mujoco.mj_inverse(model, data)
    required_torque = data.qfrc_inverse[0]
    
    print(f"  所需扭矩 τ = {required_torque:.8f} Nm")
    print(f"  验证: τ = I × α = {inertia:.8f} × 1.0 = {inertia:.8f} Nm")
    print(f"  误差: {abs(required_torque - inertia):.10f} Nm")
    
    if abs(required_torque - inertia) < 1e-6:
        print("  ✓ 验证通过！质量矩阵确实等于转动惯量")
    else:
        print("  ⚠️ 存在误差")

def main():
    """主函数"""
    
    print("MuJoCo质量矩阵M(q)可视化分析")
    print("="*80)
    
    try:
        # 绘制分析图
        inertias = plot_mass_matrix_vs_position()
        
        # 演示物理意义
        demonstrate_physical_meaning()
        
        print(f"\n" + "="*80)
        print("关键结论")
        print("="*80)
        print("1. M(q)[0,0] = 系统绕关节轴的转动惯量")
        print("2. 对于单关节系统，M(q)通常是常数")
        print("3. 可以直接从M(q)提取转动惯量")
        print("4. 这是MuJoCo自动计算的结果，考虑了所有刚体")
        print("5. 与手动平行轴定理计算应该一致")
        print("="*80)
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
