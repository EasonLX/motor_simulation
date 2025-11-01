#!/usr/bin/env python3
"""
摆杆负载测试 - 验证电机驱动能力
核心目标：确认电机能否稳定驱动不同负载的摆杆完成位置跟踪任务
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
import matplotlib.pyplot as plt
from collections import deque
import yaml
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_config():
    """加载配置文件"""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'test_params.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# 加载配置
config = load_config()

# ==================== 配置参数 ====================
# 摆杆参数
ARM_LENGTH = config['arm']['length']
ARM_DIAMETER = config['arm']['diameter'] 
ARM_DENSITY = config['arm']['density']

# 负载范围
LOAD_MASSES = config['load_masses']

# 电机参数
MOTOR_MAX_TORQUE = config['motor']['max_torque']

# 控制参数
KP = config['control']['kp']
KD = config['control']['kd']

# 测试信号参数 (根据readme要求)
TEST_DURATION = config['test_signal']['duration']
CHIRP_F_START = config['test_signal']['f_start']
CHIRP_F_END = config['test_signal']['f_end']
CHIRP_AMPLITUDE = config['test_signal']['amplitude']
# =================================================


def calculate_arm_mass(length, diameter, density):
    """计算摆杆质量"""
    radius = diameter / 2
    volume = np.pi * radius**2 * length
    mass = volume * density
    return mass


def create_model_xml(arm_length, arm_mass, load_mass):
    """创建MuJoCo模型"""
    
    radius = ARM_DIAMETER / 2
    # 计算摆杆转动惯量（圆柱体，质心坐标系）
    I_xx = (1/12) * arm_mass * arm_length**2 + (1/4) * arm_mass * radius**2
    I_yy = I_xx
    I_zz = 0.5 * arm_mass * radius**2
    
    # 负载转动惯量（球体）
    load_radius = 0.04
    I_load = (2/5) * load_mass * load_radius**2
    
    xml = f"""
<mujoco model="load_test">
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
        
        <body name="arm" pos="{arm_length/2} 0 0.04">
          <geom type="cylinder" size="{radius} {arm_length/2}" 
                quat="0.707107 0 0.707107 0" rgba="0.1 0.5 0.8 1"/>
          <inertial pos="0 0 0" mass="{arm_mass}" 
                    diaginertia="{I_xx:.8f} {I_yy:.8f} {I_zz:.8f}"/>
          
          <body name="load" pos="{arm_length/2} 0 0">
            <geom type="sphere" size="{load_radius}" rgba="0.8 0.1 0.1 1"/>
            <inertial pos="0 0 0" mass="{load_mass}" 
                      diaginertia="{I_load:.8f} {I_load:.8f} {I_load:.8f}"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  
  <actuator>
    <motor name="motor1" joint="joint1" gear="1" ctrllimited="true" ctrlrange="-{MOTOR_MAX_TORQUE} {MOTOR_MAX_TORQUE}"/>
  </actuator>
</mujoco>
"""
    return xml


def chirp_signal(t, f_start, f_end, duration, amplitude_deg):
    """Chirp扫频信号"""
    k = (f_end - f_start) / duration
    phase = 2 * np.pi * (f_start * t + 0.5 * k * t**2)
    return np.deg2rad(amplitude_deg) * np.sin(phase)


def run_load_test(arm_mass, load_mass, show_plot=False):
    """运行单次负载测试"""
    
    print(f"\n{'='*70}")
    print(f"负载测试: {load_mass} kg")
    print(f"{'='*70}")
    
    # 创建模型
    xml_string = create_model_xml(ARM_LENGTH, arm_mass, load_mass)
    model = mujoco.MjModel.from_xml_string(xml_string)
    data = mujoco.MjData(model)
    
    # 获取MuJoCo计算的转动惯量
    data.qpos[0] = 0
    data.qvel[0] = 0
    data.qacc[0] = 0
    mujoco.mj_forward(model, data)
    
    M = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, M, data.qM)
    inertia = M[0, 0]
    
    print(f"摆杆质量: {arm_mass:.3f} kg")
    print(f"负载质量: {load_mass} kg")
    print(f"系统转动惯量: {inertia:.6f} kg·m²")
    
    # 记录数据
    time_data = []
    torque_data = []
    pos_error_data = []
    
    # 运行仿真
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 45
        viewer.cam.elevation = -20
        viewer.cam.distance = 1.5
        viewer.cam.lookat = np.array([0.0, 0.0, 0.2])
        
        prev_pos_des = 0.0
        
        while viewer.is_running() and data.time < TEST_DURATION:
            step_start = time.time()
            
            t = data.time
            
            # Chirp信号
            pos_des = chirp_signal(t, CHIRP_F_START, CHIRP_F_END, TEST_DURATION, CHIRP_AMPLITUDE)
            vel_des = (pos_des - prev_pos_des) / model.opt.timestep
            prev_pos_des = pos_des
            
            # PD控制
            pos_err = pos_des - data.qpos[0]
            vel_err = vel_des - data.qvel[0]
            torque = KP * pos_err + KD * vel_err
            torque = np.clip(torque, -MOTOR_MAX_TORQUE, MOTOR_MAX_TORQUE)
            
            data.ctrl[0] = torque
            
            # 记录数据
            time_data.append(t)
            torque_data.append(torque)
            pos_error_data.append(pos_err)
            
            mujoco.mj_step(model, data)
            viewer.sync()
            
            time_until_next = model.opt.timestep - (time.time() - step_start)
            if time_until_next > 0:
                time.sleep(time_until_next)
    
    # 分析结果
    torque_data = np.array(torque_data)
    pos_error_data = np.array(pos_error_data)
    
    max_torque = np.max(np.abs(torque_data))
    rms_torque = np.sqrt(np.mean(torque_data**2))
    max_pos_error = np.max(np.abs(pos_error_data))
    rms_pos_error = np.sqrt(np.mean(pos_error_data**2))
    
    print(f"\n测试结果:")
    print(f"  最大扭矩: {max_torque:.2f} Nm")
    print(f"  RMS扭矩: {rms_torque:.2f} Nm")
    print(f"  扭矩利用率: {max_torque/MOTOR_MAX_TORQUE*100:.1f}%")
    print(f"  最大位置误差: {np.rad2deg(max_pos_error):.3f}°")
    print(f"  RMS位置误差: {np.rad2deg(rms_pos_error):.3f}°")
    
    # 判断是否能稳定驱动
    if max_torque < MOTOR_MAX_TORQUE * 0.9:
        print(f"  ✓ 电机能稳定驱动 (扭矩裕度: {(1-max_torque/MOTOR_MAX_TORQUE)*100:.1f}%)")
        status = "OK"
    elif max_torque < MOTOR_MAX_TORQUE:
        print(f"  ⚠️ 接近极限 (扭矩裕度: {(1-max_torque/MOTOR_MAX_TORQUE)*100:.1f}%)")
        status = "MARGINAL"
    else:
        print(f"  ✗ 超出电机能力！")
        status = "FAIL"
    
    return {
        'load_mass': load_mass,
        'inertia': inertia,
        'max_torque': max_torque,
        'rms_torque': rms_torque,
        'max_pos_error': max_pos_error,
        'rms_pos_error': rms_pos_error,
        'status': status
    }


def main():
    """主函数"""
    
    print("="*70)
    print("摆杆负载测试 - 验证电机驱动能力")
    print("="*70)
    print(f"摆杆参数:")
    print(f"  长度: {ARM_LENGTH} m")
    print(f"  直径: {ARM_DIAMETER} m (半径: {ARM_DIAMETER/2} m)")
    print(f"  材质: 钢 (密度: {ARM_DENSITY} kg/m³)")
    
    # 计算摆杆质量
    arm_mass = calculate_arm_mass(ARM_LENGTH, ARM_DIAMETER, ARM_DENSITY)
    print(f"  质量: {arm_mass:.3f} kg")
    
    print(f"\n电机参数:")
    print(f"  最大扭矩: {MOTOR_MAX_TORQUE} Nm")
    
    print(f"\n控制参数:")
    print(f"  Kp: {KP}, Kd: {KD}")
    
    print(f"\n测试信号 (按readme要求):")
    print(f"  Chirp {CHIRP_F_START}~{CHIRP_F_END} Hz, 幅值±{CHIRP_AMPLITUDE}°")
    print(f"  持续时间: {TEST_DURATION} 秒")
    print(f"  信号类型: 扫频信号 (0.1Hz→10Hz, 20秒)")
    
    print(f"\n负载范围: {LOAD_MASSES[0]}~{LOAD_MASSES[-1]} kg")
    print("="*70)
    
    # 运行测试
    results = []
    for load_mass in LOAD_MASSES:
        input(f"\n按Enter开始测试 {load_mass}kg 负载...")
        result = run_load_test(arm_mass, load_mass, show_plot=False)
        results.append(result)
    
    # 生成汇总报告
    print(f"\n{'='*90}")
    print("测试汇总报告 - 给结构设计部门参考")
    print(f"{'='*90}")
    print(f"{'负载(kg)':<10} {'转动惯量(kg·m²)':<18} {'最大扭矩(Nm)':<15} {'扭矩利用率':<12} {'状态':<10}")
    print(f"{'-'*90}")
    
    for result in results:
        status_symbol = {
            'OK': '✓ 正常',
            'MARGINAL': '⚠️ 接近极限',
            'FAIL': '✗ 超限'
        }[result['status']]
        
        print(f"{result['load_mass']:<10} "
              f"{result['inertia']:<18.6f} "
              f"{result['max_torque']:<15.2f} "
              f"{result['max_torque']/MOTOR_MAX_TORQUE*100:<12.1f}% "
              f"{status_symbol:<10}")
    
    # 确定最大可用负载
    ok_results = [r for r in results if r['status'] == 'OK']
    if ok_results:
        max_safe_load = max([r['load_mass'] for r in ok_results])
        print(f"\n{'='*90}")
        print(f"结论:")
        print(f"  电机最大稳定驱动负载: {max_safe_load} kg")
        print(f"  对应转动惯量: {[r['inertia'] for r in results if r['load_mass']==max_safe_load][0]:.6f} kg·m²")
        print(f"  对应最大扭矩: {[r['max_torque'] for r in results if r['load_mass']==max_safe_load][0]:.2f} Nm")
        print(f"\n  推荐工装设计负载范围: 1~{max_safe_load} kg")
        print(f"  对应转动惯量范围: {results[0]['inertia']:.6f} ~ {[r['inertia'] for r in results if r['load_mass']==max_safe_load][0]:.6f} kg·m²")
        print(f"{'='*90}")
    else:
        print(f"\n⚠️ 警告: 所有负载都超出电机能力！")
    
    # 保存结果
    import csv
    with open('load_test_results.csv', 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['load_mass', 'inertia', 'max_torque', 
                                               'rms_torque', 'max_pos_error', 'rms_pos_error', 'status'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n详细结果已保存至: load_test_results.csv")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n测试已中断")

