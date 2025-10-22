import mujoco
import mujoco.viewer
import numpy as np
import time

"""
仿真验证不同摆杆配置
测试不同惯量下电机的动态响应
"""

def create_model_xml(arm_length, arm_mass, weight_mass, weight_position):
    """
    动态生成MJCF模型
    
    Parameters:
    - arm_length: 摆杆长度 (m)
    - arm_mass: 摆杆质量 (kg)
    - weight_mass: 重量块质量 (kg)
    - weight_position: 重量块位置 (m，沿摆杆方向)
    """
    
    # 计算link2的转动惯量（圆柱体）
    radius = 0.015
    # 圆柱体绕中心轴: I_zz = 0.5 * m * r^2
    # 圆柱体绕垂直轴: I_xx = I_yy = (1/12) * m * L^2 + (1/4) * m * r^2
    I_xx = (1/12) * arm_mass * arm_length**2 + (1/4) * arm_mass * radius**2
    I_yy = I_xx
    I_zz = 0.5 * arm_mass * radius**2
    
    xml = f"""
<mujoco model="single_arm_test">
  <option timestep="0.001" gravity="0 0 -9.81"/>
  
  <asset>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1=".1 .2 .3" rgb2=".2 .3 .4"/>
    <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance=".2"/>
    <material name="base_mat" rgba="0.3 0.3 0.3 1"/>
    <material name="link1_mat" rgba="0.6 0.3 0.1 1"/>
    <material name="link2_mat" rgba="0.1 0.5 0.8 1"/>
    <material name="weight_mat" rgba="0.8 0.1 0.1 1"/>
  </asset>
  
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1" material="grid"/>
    <light directional="true" pos="0 0 3" dir="0 0 -1"/>
    
    <!-- 基座 -->
    <body name="base_link" pos="0 0 0.03">
      <geom name="base_geom" type="cylinder" size="0.075 0.03" material="base_mat"/>
      
      <!-- link1: 竖直连接杆 -->
      <body name="link1" pos="0 0 0.07">
        <joint name="joint1" type="hinge" axis="0 0 1" limited="false" damping="0.1"/>
        <geom name="link1_geom" type="cylinder" size="0.015 0.04" material="link1_mat"/>
        <inertial pos="0 0 0" mass="0.5" diaginertia="0.001 0.001 0.0001"/>
        
        <!-- link2: 水平摆杆 -->
        <body name="link2" pos="{arm_length/2} 0 0.04">
          <geom name="link2_geom" type="cylinder" size="0.015 {arm_length/2}" 
                quat="0.707107 0 0.707107 0" material="link2_mat"/>
          <inertial pos="0 0 0" mass="{arm_mass}" diaginertia="{I_xx:.6f} {I_yy:.6f} {I_zz:.6f}"/>
          
          <!-- 重量块（如果有）-->
          {"" if weight_mass <= 0 else f'''
          <body name="weight" pos="{weight_position - arm_length/2} 0 0">
            <geom name="weight_geom" type="sphere" size="0.03" material="weight_mat"/>
            <inertial pos="0 0 0" mass="{weight_mass}" diaginertia="0.0001 0.0001 0.0001"/>
          </body>
          '''}
        </body>
      </body>
    </body>
  </worldbody>
  
  <actuator>
    <motor name="motor1" joint="joint1" gear="1" ctrllimited="true" ctrlrange="-10 10"/>
  </actuator>
  
</mujoco>
"""
    return xml

def chirp_signal(t, f_start, f_end, duration, amplitude):
    """生成Chirp扫频信号"""
    k = (f_end - f_start) / duration
    phase = 2 * np.pi * (f_start * t + 0.5 * k * t**2)
    return amplitude * np.sin(phase)

def run_test(arm_length, arm_mass, weight_mass, weight_position, test_name):
    """运行单次仿真测试"""
    print(f"\n{'='*60}")
    print(f"测试配置: {test_name}")
    print(f"{'='*60}")
    print(f"摆杆长度: {arm_length:.3f} m")
    print(f"摆杆质量: {arm_mass:.3f} kg")
    print(f"重量块质量: {weight_mass:.1f} kg")
    print(f"重量块位置: {weight_position:.3f} m")
    
    # 计算理论转动惯量
    arm_center = arm_length / 2
    I_link1 = 0.0001
    I_arm = (1/12) * arm_mass * arm_length**2 + arm_mass * arm_center**2
    I_weight = weight_mass * weight_position**2 if weight_mass > 0 else 0
    I_total = I_link1 + I_arm + I_weight
    print(f"理论转动惯量: {I_total:.6f} kg·m²")
    print(f"{'='*60}\n")
    
    # 创建模型
    xml_string = create_model_xml(arm_length, arm_mass, weight_mass, weight_position)
    model = mujoco.MjModel.from_xml_string(xml_string)
    data = mujoco.MjData(model)
    
    # 记录数据
    time_data = []
    pos_data = []
    vel_data = []
    torque_data = []
    
    # 运行仿真
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 45
        viewer.cam.elevation = -20
        viewer.cam.distance = 1.5
        viewer.cam.lookat = np.array([0.0, 0.0, 0.2])
        
        test_duration = 10.0  # 秒
        
        while viewer.is_running() and data.time < test_duration:
            step_start = time.time()
            
            # Chirp信号: 0.1Hz ~ 5Hz
            current_time = data.time
            desired_pos = chirp_signal(current_time, 0.1, 5.0, test_duration, np.deg2rad(10))
            
            # PD控制
            kp = 50.0
            kd = 5.0
            pos_error = desired_pos - data.qpos[0]
            vel_error = 0 - data.qvel[0]
            torque = kp * pos_error + kd * vel_error
            torque = np.clip(torque, -10, 10)
            
            data.ctrl[0] = torque
            
            # 记录数据
            time_data.append(current_time)
            pos_data.append(data.qpos[0])
            vel_data.append(data.qvel[0])
            torque_data.append(torque)
            
            mujoco.mj_step(model, data)
            viewer.sync()
            
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
    
    print(f"测试完成: {test_name}")
    print(f"最大角速度: {np.max(np.abs(vel_data)):.3f} rad/s")
    print(f"最大扭矩: {np.max(np.abs(torque_data)):.3f} Nm")
    
    return {
        'time': np.array(time_data),
        'position': np.array(pos_data),
        'velocity': np.array(vel_data),
        'torque': np.array(torque_data),
        'inertia': I_total
    }

def main():
    """主函数：测试三种配置"""
    
    ARM_RADIUS = 0.015
    DENSITY_ALUMINUM = 2700
    
    # 配置1: 空载
    arm_length_1 = 0.3
    arm_volume_1 = np.pi * ARM_RADIUS**2 * arm_length_1
    arm_mass_1 = arm_volume_1 * DENSITY_ALUMINUM
    config_1 = {
        'arm_length': arm_length_1,
        'arm_mass': arm_mass_1,
        'weight_mass': 0.0,
        'weight_position': arm_length_1,
        'name': '空载配置'
    }
    
    # 配置2: 中负载
    arm_length_2 = 0.35
    arm_volume_2 = np.pi * ARM_RADIUS**2 * arm_length_2
    arm_mass_2 = arm_volume_2 * DENSITY_ALUMINUM
    config_2 = {
        'arm_length': arm_length_2,
        'arm_mass': arm_mass_2,
        'weight_mass': 1.0,
        'weight_position': arm_length_2,
        'name': '中负载配置'
    }
    
    # 配置3: 大负载
    arm_length_3 = 0.4
    arm_volume_3 = np.pi * ARM_RADIUS**2 * arm_length_3
    arm_mass_3 = arm_volume_3 * DENSITY_ALUMINUM
    config_3 = {
        'arm_length': arm_length_3,
        'arm_mass': arm_mass_3,
        'weight_mass': 2.0,
        'weight_position': arm_length_3,
        'name': '大负载配置'
    }
    
    print("\n" + "="*60)
    print("MuJoCo仿真测试 - 不同惯量配置验证")
    print("="*60)
    print("将依次测试三种配置，每次测试10秒")
    print("控制信号: Chirp 0.1~5Hz, 幅值±10°")
    print("PD控制: Kp=50, Kd=5")
    print("="*60)
    
    # 依次测试
    for config in [config_1, config_2, config_3]:
        input(f"\n按Enter开始测试: {config['name']}...")
        run_test(
            config['arm_length'],
            config['arm_mass'],
            config['weight_mass'],
            config['weight_position'],
            config['name']
        )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n测试已中断")

