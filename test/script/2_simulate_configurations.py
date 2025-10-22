import mujoco
import mujoco.viewer
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import threading

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

"""
仿真验证不同摆杆配置
测试不同惯量下电机的动态响应
实时显示力矩/位置/速度跟踪波形
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
    <motor name="motor1" joint="joint1" gear="1" ctrllimited="true" ctrlrange="-100 100"/>
  </actuator>
  
</mujoco>
"""
    return xml

def chirp_signal(t, f_start, f_end, duration, amplitude):
    """生成Chirp扫频信号"""
    k = (f_end - f_start) / duration
    phase = 2 * np.pi * (f_start * t + 0.5 * k * t**2)
    return amplitude * np.sin(phase)


class RealtimePlotter:
    """实时绘制跟踪曲线"""
    
    def __init__(self, window_size=500):
        """
        初始化实时绘图器
        
        Parameters:
        - window_size: 显示的数据点数量
        """
        self.window_size = window_size
        
        # 数据缓冲区
        self.time_buffer = deque(maxlen=window_size)
        self.pos_des_buffer = deque(maxlen=window_size)
        self.pos_act_buffer = deque(maxlen=window_size)
        self.vel_des_buffer = deque(maxlen=window_size)
        self.vel_act_buffer = deque(maxlen=window_size)
        self.torque_cmd_buffer = deque(maxlen=window_size)
        
        # 创建图形
        plt.ion()
        self.fig, self.axes = plt.subplots(3, 1, figsize=(12, 9))
        self.fig.suptitle('实时跟踪曲线', fontsize=14, fontweight='bold')
        
        # 位置跟踪子图
        self.ax_pos = self.axes[0]
        self.line_pos_des, = self.ax_pos.plot([], [], 'r-', label='期望位置', linewidth=2)
        self.line_pos_act, = self.ax_pos.plot([], [], 'b-', label='实际位置', linewidth=1.5)
        self.ax_pos.set_ylabel('位置 (rad)', fontsize=11)
        self.ax_pos.legend(loc='upper right')
        self.ax_pos.grid(True, alpha=0.3)
        self.ax_pos.set_title('位置跟踪')
        
        # 速度跟踪子图
        self.ax_vel = self.axes[1]
        self.line_vel_des, = self.ax_vel.plot([], [], 'r-', label='期望速度', linewidth=2)
        self.line_vel_act, = self.ax_vel.plot([], [], 'g-', label='实际速度', linewidth=1.5)
        self.ax_vel.set_ylabel('速度 (rad/s)', fontsize=11)
        self.ax_vel.legend(loc='upper right')
        self.ax_vel.grid(True, alpha=0.3)
        self.ax_vel.set_title('速度跟踪')
        
        # 力矩子图
        self.ax_torque = self.axes[2]
        self.line_torque, = self.ax_torque.plot([], [], 'm-', label='控制力矩', linewidth=1.5)
        self.ax_torque.set_xlabel('时间 (s)', fontsize=11)
        self.ax_torque.set_ylabel('力矩 (Nm)', fontsize=11)
        self.ax_torque.legend(loc='upper right')
        self.ax_torque.grid(True, alpha=0.3)
        self.ax_torque.set_title('控制力矩')
        
        plt.tight_layout()
        plt.show(block=False)
        
        self.running = True
    
    def update_data(self, time_val, pos_des, pos_act, vel_des, vel_act, torque_cmd):
        """更新数据缓冲区"""
        self.time_buffer.append(time_val)
        self.pos_des_buffer.append(pos_des)
        self.pos_act_buffer.append(pos_act)
        self.vel_des_buffer.append(vel_des)
        self.vel_act_buffer.append(vel_act)
        self.torque_cmd_buffer.append(torque_cmd)
    
    def update_plot(self):
        """更新图形显示"""
        if len(self.time_buffer) < 2:
            return
        
        time_array = np.array(self.time_buffer)
        
        # 更新位置曲线
        self.line_pos_des.set_data(time_array, np.array(self.pos_des_buffer))
        self.line_pos_act.set_data(time_array, np.array(self.pos_act_buffer))
        self.ax_pos.relim()
        self.ax_pos.autoscale_view()
        
        # 更新速度曲线
        self.line_vel_des.set_data(time_array, np.array(self.vel_des_buffer))
        self.line_vel_act.set_data(time_array, np.array(self.vel_act_buffer))
        self.ax_vel.relim()
        self.ax_vel.autoscale_view()
        
        # 更新力矩曲线
        self.line_torque.set_data(time_array, np.array(self.torque_cmd_buffer))
        self.ax_torque.relim()
        self.ax_torque.autoscale_view()
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def close(self):
        """关闭绘图窗口"""
        self.running = False
        plt.ioff()
        plt.close(self.fig)

def run_test(arm_length, arm_mass, weight_mass, weight_position, test_name):
    """运行单次仿真测试（带实时波形显示）"""
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
    
    # 创建实时绘图器
    plotter = RealtimePlotter(window_size=1000)
    
    # 记录数据
    time_data = []
    pos_des_data = []
    pos_data = []
    vel_des_data = []
    vel_data = []
    torque_data = []
    
    # 运行仿真
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 45
        viewer.cam.elevation = -20
        viewer.cam.distance = 1.5
        viewer.cam.lookat = np.array([0.0, 0.0, 0.2])
        
        test_duration = 10.0  # 秒
        plot_update_counter = 0
        plot_update_interval = 50  # 每50步更新一次图形（减少绘图频率）
        
        while viewer.is_running() and data.time < test_duration and plotter.running:
            step_start = time.time()
            
            # Chirp信号: 0.1Hz ~ 5Hz
            current_time = data.time
            desired_pos = chirp_signal(current_time, 0.1, 5.0, test_duration, np.deg2rad(10))
            
            # 计算期望速度（数值微分）
            dt = model.opt.timestep
            if len(time_data) > 0:
                desired_vel = (desired_pos - pos_des_data[-1]) / dt
            else:
                desired_vel = 0.0
            
            # PD控制
            kp = 80.0
            kd = 5.0
            pos_error = desired_pos - data.qpos[0]
            vel_error = desired_vel - data.qvel[0]
            torque = kp * pos_error + kd * vel_error
            torque = np.clip(torque, -100, 100)
            
            data.ctrl[0] = torque
            
            # 记录数据
            time_data.append(current_time)
            pos_des_data.append(desired_pos)
            pos_data.append(data.qpos[0])
            vel_des_data.append(desired_vel)
            vel_data.append(data.qvel[0])
            torque_data.append(torque)
            
            # 更新实时绘图
            plotter.update_data(current_time, desired_pos, data.qpos[0], 
                              desired_vel, data.qvel[0], torque)
            
            plot_update_counter += 1
            if plot_update_counter >= plot_update_interval:
                plotter.update_plot()
                plot_update_counter = 0
            
            mujoco.mj_step(model, data)
            viewer.sync()
            
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
    
    # 最后更新一次图形
    plotter.update_plot()
    
    # 计算跟踪误差统计
    pos_error_array = np.array(pos_des_data) - np.array(pos_data)
    vel_error_array = np.array(vel_des_data) - np.array(vel_data)
    
    print(f"\n测试完成: {test_name}")
    print(f"{'='*60}")
    print(f"位置跟踪:")
    print(f"  - RMS误差: {np.sqrt(np.mean(pos_error_array**2)):.6f} rad ({np.rad2deg(np.sqrt(np.mean(pos_error_array**2))):.3f}°)")
    print(f"  - 最大误差: {np.max(np.abs(pos_error_array)):.6f} rad ({np.rad2deg(np.max(np.abs(pos_error_array))):.3f}°)")
    print(f"\n速度跟踪:")
    print(f"  - RMS误差: {np.sqrt(np.mean(vel_error_array**2)):.6f} rad/s")
    print(f"  - 最大速度: {np.max(np.abs(vel_data)):.3f} rad/s")
    print(f"\n控制力矩:")
    print(f"  - 最大力矩: {np.max(np.abs(torque_data)):.3f} Nm")
    print(f"  - 平均力矩: {np.mean(np.abs(torque_data)):.3f} Nm")
    print(f"{'='*60}\n")
    
    # 等待用户查看图形
    input("按Enter键关闭波形图并继续...")
    plotter.close()
    
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
    
    # 配置1: 空载 (0.2m摆杆)
    arm_length_1 = 0.2
    arm_volume_1 = np.pi * ARM_RADIUS**2 * arm_length_1
    arm_mass_1 = arm_volume_1 * DENSITY_ALUMINUM
    config_1 = {
        'arm_length': arm_length_1,
        'arm_mass': arm_mass_1,
        'weight_mass': 0.0,
        'weight_position': arm_length_1,
        'name': '空载配置'
    }
    
    # 配置2: 中负载 (0.25m摆杆 + 2kg重量)
    arm_length_2 = 0.25
    arm_volume_2 = np.pi * ARM_RADIUS**2 * arm_length_2
    arm_mass_2 = arm_volume_2 * DENSITY_ALUMINUM
    config_2 = {
        'arm_length': arm_length_2,
        'arm_mass': arm_mass_2,
        'weight_mass': 2.0,
        'weight_position': arm_length_2,
        'name': '中负载配置'
    }
    
    # 配置3: 大负载 (0.3m摆杆 + 4kg重量)
    arm_length_3 = 0.3
    arm_volume_3 = np.pi * ARM_RADIUS**2 * arm_length_3
    arm_mass_3 = arm_volume_3 * DENSITY_ALUMINUM
    config_3 = {
        'arm_length': arm_length_3,
        'arm_mass': arm_mass_3,
        'weight_mass': 4.0,
        'weight_position': arm_length_3,
        'name': '大负载配置'
    }
    
    print("\n" + "="*60)
    print("MuJoCo仿真测试 - 不同惯量配置验证（带实时波形显示）")
    print("="*60)
    print("将依次测试三种配置，每次测试10秒")
    print("控制信号: Chirp 0.1~5Hz, 幅值±10°")
    print("PD控制: Kp=50, Kd=5")
    print("\n实时显示:")
    print("  - 位置跟踪曲线（期望 vs 实际）")
    print("  - 速度跟踪曲线（期望 vs 实际）")
    print("  - 控制力矩曲线")
    print("="*60)
    
    # 依次测试
    for i, config in enumerate([config_1, config_2, config_3], 1):
        print(f"\n[{i}/3] 准备测试: {config['name']}")
        input("按Enter开始...")
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

