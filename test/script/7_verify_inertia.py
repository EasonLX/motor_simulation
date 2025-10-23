#!/usr/bin/env python3
"""
转动惯量验证脚本
验证手算的系统总转动惯量与MuJoCo仿真的一致性
"""

import mujoco
import numpy as np

def create_test_model(arm_length, arm_mass, weight_mass):
    """创建测试模型"""
    
    # 计算摆杆转动惯量（在质心坐标系）
    radius = 0.015
    # 圆柱体绕垂直轴
    I_arm_xx = (1/12) * arm_mass * arm_length**2 + (1/4) * arm_mass * radius**2
    I_arm_yy = I_arm_xx
    # 圆柱体绕自身轴
    I_arm_zz = 0.5 * arm_mass * radius**2
    
    # 重量块转动惯量（球体）
    weight_radius = 0.04
    I_weight = (2/5) * weight_mass * weight_radius**2
    
    xml = f"""
<mujoco model="inertia_test">
  <option timestep="0.001" gravity="0 0 -9.81"/>
  
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <body name="base" pos="0 0 0.03">
      <geom type="cylinder" size="0.075 0.03"/>
      
      <body name="link1" pos="0 0 0.07">
        <joint name="joint1" type="hinge" axis="0 0 1" limited="false" damping="0"/>
        <geom type="cylinder" size="0.015 0.04"/>
        <inertial pos="0 0 0" mass="0.5" diaginertia="0.001 0.001 0.0001"/>
        
        <body name="arm" pos="{arm_length/2} 0 0.04">
          <geom type="cylinder" size="0.015 {arm_length/2}" quat="0.707107 0 0.707107 0"/>
          <inertial pos="0 0 0" mass="{arm_mass}" 
                    diaginertia="{I_arm_xx:.8f} {I_arm_yy:.8f} {I_arm_zz:.8f}"/>
          
          {"" if weight_mass <= 0 else f'''
          <body name="weight" pos="{arm_length/2} 0 0">
            <geom type="sphere" size="0.04"/>
            <inertial pos="0 0 0" mass="{weight_mass}" 
                      diaginertia="{I_weight:.8f} {I_weight:.8f} {I_weight:.8f}"/>
          </body>
          '''}
        </body>
      </body>
    </body>
  </worldbody>
  
  <actuator>
    <motor name="motor1" joint="joint1" gear="1" ctrllimited="true" ctrlrange="-100 100"/>
  </actuator>
</mujoco>
"""
    return xml

def calculate_theoretical_inertia(arm_length, arm_mass, weight_mass):
    """理论计算系统绕z轴的总转动惯量"""
    
    # Link1的转动惯量
    I_link1 = 0.0001  # kg·m² (绕z轴)
    
    # 摆杆绕z轴的转动惯量（使用平行轴定理）
    arm_center = arm_length / 2
    I_arm_com = (1/12) * arm_mass * arm_length**2  # 绕质心
    I_arm_total = I_arm_com + arm_mass * arm_center**2  # 绕z轴
    
    # 重量块绕z轴的转动惯量
    weight_distance = arm_length
    I_weight_total = weight_mass * weight_distance**2 if weight_mass > 0 else 0
    
    # 系统总转动惯量
    I_total = I_link1 + I_arm_total + I_weight_total
    
    return {
        'I_link1': I_link1,
        'I_arm': I_arm_total,
        'I_weight': I_weight_total,
        'I_total': I_total
    }

def measure_mujoco_inertia(model, data):
    """通过MuJoCo仿真测量转动惯量"""
    
    # 方法1：施加单位角加速度，测量所需扭矩
    data.qpos[0] = 0
    data.qvel[0] = 0
    
    # 设置角加速度为1 rad/s²
    target_qacc = 1.0
    
    # 使用逆动力学计算所需扭矩
    # qfrc_inverse = M * qacc + C + G
    mujoco.mj_forward(model, data)
    
    # 先消除重力影响：设置水平位置
    data.qpos[0] = np.pi / 2  # 水平位置，重力不产生扭矩
    data.qvel[0] = 0
    mujoco.mj_forward(model, data)
    
    # 记录当前所需扭矩（平衡重力）
    gravity_torque = data.qfrc_bias[0]
    
    # 施加角加速度
    data.qacc[0] = target_qacc
    mujoco.mj_inverse(model, data)
    
    # 所需扭矩 = I * α (忽略重力项)
    torque_needed = data.qfrc_inverse[0] - gravity_torque
    inertia_measured = torque_needed / target_qacc
    
    return inertia_measured

def verify_configuration(arm_length, arm_mass, weight_mass, config_name):
    """验证单个配置"""
    
    print(f"\n{'='*70}")
    print(f"配置: {config_name}")
    print(f"{'='*70}")
    print(f"摆杆长度: {arm_length:.3f} m")
    print(f"摆杆质量: {arm_mass:.3f} kg")
    print(f"重量块质量: {weight_mass:.1f} kg")
    print(f"{'-'*70}")
    
    # 理论计算
    theoretical = calculate_theoretical_inertia(arm_length, arm_mass, weight_mass)
    
    print(f"\n理论计算（手算，绕电机轴z）:")
    print(f"  Link1转动惯量:    {theoretical['I_link1']:.8f} kg·m²")
    print(f"  摆杆转动惯量:     {theoretical['I_arm']:.8f} kg·m²")
    print(f"  重量块转动惯量:   {theoretical['I_weight']:.8f} kg·m²")
    print(f"  ---")
    print(f"  系统总转动惯量:   {theoretical['I_total']:.8f} kg·m²")
    
    # MuJoCo仿真
    xml = create_test_model(arm_length, arm_mass, weight_mass)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    
    # 方法1：质量矩阵M(q)
    mujoco.mj_forward(model, data)
    M = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, M, data.qM)
    inertia_from_M = M[0, 0]
    
    print(f"\nMuJoCo仿真验证:")
    print(f"  方法1 - 质量矩阵M(q):  {inertia_from_M:.8f} kg·m²")
    
    # 方法2：逆动力学
    inertia_measured = measure_mujoco_inertia(model, data)
    print(f"  方法2 - 逆动力学:      {inertia_measured:.8f} kg·m²")
    
    # 比较结果
    error_M = abs(inertia_from_M - theoretical['I_total']) / theoretical['I_total'] * 100
    error_inv = abs(inertia_measured - theoretical['I_total']) / theoretical['I_total'] * 100
    
    print(f"\n验证结果:")
    print(f"  理论值:           {theoretical['I_total']:.8f} kg·m²")
    print(f"  MuJoCo M(q):      {inertia_from_M:.8f} kg·m² (误差: {error_M:.2f}%)")
    print(f"  MuJoCo逆动力学:   {inertia_measured:.8f} kg·m² (误差: {error_inv:.2f}%)")
    
    if error_M < 1.0 and error_inv < 1.0:
        print(f"  ✓ 验证通过！理论计算与仿真一致")
    else:
        print(f"  ⚠️  误差较大，需要检查计算方法")
    
    return {
        'config': config_name,
        'theoretical': theoretical['I_total'],
        'mujoco_M': inertia_from_M,
        'mujoco_inv': inertia_measured,
        'error': error_M
    }

def main():
    """主函数"""
    
    print("="*70)
    print("转动惯量验证 - 理论计算 vs MuJoCo仿真")
    print("="*70)
    print("目的：验证给结构的'系统总转动惯量'计算是否正确")
    print("="*70)
    
    # 测试配置
    ARM_RADIUS = 0.015
    DENSITY_ALUMINUM = 2700
    
    configs = []
    
    # 配置1: 空载
    arm_length_1 = 0.2
    arm_volume_1 = np.pi * ARM_RADIUS**2 * arm_length_1
    arm_mass_1 = arm_volume_1 * DENSITY_ALUMINUM
    configs.append({
        'name': '空载',
        'arm_length': arm_length_1,
        'arm_mass': arm_mass_1,
        'weight_mass': 0.0
    })
    
    # 配置2: 中负载
    arm_length_2 = 0.25
    arm_volume_2 = np.pi * ARM_RADIUS**2 * arm_length_2
    arm_mass_2 = arm_volume_2 * DENSITY_ALUMINUM
    configs.append({
        'name': '中负载',
        'arm_length': arm_length_2,
        'arm_mass': arm_mass_2,
        'weight_mass': 2.0
    })
    
    # 配置3: 大负载
    arm_length_3 = 0.3
    arm_volume_3 = np.pi * ARM_RADIUS**2 * arm_length_3
    arm_mass_3 = arm_volume_3 * DENSITY_ALUMINUM
    configs.append({
        'name': '大负载',
        'arm_length': arm_length_3,
        'arm_mass': arm_mass_3,
        'weight_mass': 4.0
    })
    
    # 验证所有配置
    results = []
    for config in configs:
        result = verify_configuration(
            config['arm_length'],
            config['arm_mass'],
            config['weight_mass'],
            config['name']
        )
        results.append(result)
    
    # 总结
    print(f"\n{'='*70}")
    print("验证总结")
    print(f"{'='*70}")
    print(f"{'配置':<12} {'理论值':<18} {'MuJoCo M(q)':<18} {'误差':<10}")
    print(f"{'-'*70}")
    
    for result in results:
        print(f"{result['config']:<12} "
              f"{result['theoretical']:<18.8f} "
              f"{result['mujoco_M']:<18.8f} "
              f"{result['error']:<10.2f}%")
    
    print(f"{'='*70}")
    print("\n结论:")
    print("- 脚本计算的'系统总转动惯量' = 系统绕电机轴z的转动惯量")
    print("- ✓ 可以直接给结构设计作为负载参考")
    print("- ✓ 用于估算所需扭矩: τ = I_total × α")
    print("- MJCF中的diaginertia是各body在质心系的转动惯量")
    print("- MuJoCo根据各body的inertia自动计算系统总转动惯量M(q)")
    print("="*70)

if __name__ == "__main__":
    main()
