import mujoco
import mujoco.viewer
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import threading
import yaml
import os

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

"""
仿真验证不同摆杆配置
测试不同惯量下电机的动态响应
实时显示力矩/位置/速度跟踪波形
"""

def load_config(config_path="config.yaml"):
    """加载YAML配置文件"""
    # 获取脚本的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    if not os.path.exists(config_path):
        # 如果当前目录没有，尝试项目根目录的config子目录
        config_path = os.path.join(project_root, "config", "config.yaml")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def create_model_xml(arm_length, arm_mass, weight_mass, weight_position, config):
    """
    动态生成MJCF模型
    
    Parameters:
    - arm_length: 摆杆长度 (m)
    - arm_mass: 摆杆质量 (kg)
    - weight_mass: 重量块质量 (kg)
    - weight_position: 重量块位置 (m，沿摆杆方向)
    - config: 配置字典
    """
    
    # 从配置获取参数
    radius = config['material']['arm_radius']
    timestep = config['simulation']['timestep']
    gravity = config['simulation']['gravity']
    torque_limit = config['control']['torque_limit']
    
    # 计算link2的转动惯量（圆柱体）
    # 圆柱体绕中心轴: I_zz = 0.5 * m * r^2
    # 圆柱体绕垂直轴: I_xx = I_yy = (1/12) * m * L^2 + (1/4) * m * r^2
    I_xx = (1/12) * arm_mass * arm_length**2 + (1/4) * arm_mass * radius**2
    I_yy = I_xx
    I_zz = 0.5 * arm_mass * radius**2
    
    xml = f"""
<mujoco model="single_arm_test">
  <option timestep="{timestep}" gravity="{gravity[0]} {gravity[1]} {gravity[2]}"/>
  
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
    <motor name="motor1" joint="joint1" gear="1" ctrllimited="true" ctrlrange="-{torque_limit} {torque_limit}"/>
  </actuator>
  
</mujoco>
"""
    return xml

def chirp_signal(t, f_start, f_end, duration, amplitude):
    """生成Chirp扫频信号"""
    k = (f_end - f_start) / duration
    phase = 2 * np.pi * (f_start * t + 0.5 * k * t**2)
    return np.deg2rad(amplitude) * np.sin(phase)  # 转换为弧度


class RealtimePlotter:
    """实时绘制跟踪曲线"""
    
    def __init__(self, config):
        """
        初始化实时绘图器
        
        Parameters:
        - config: 配置字典
        """
        self.window_size = config['display']['window_size']
        self.figure_size = config['display']['figure_size']
        
        # 数据缓冲区
        self.time_buffer = deque(maxlen=self.window_size)
        self.pos_des_buffer = deque(maxlen=self.window_size)
        self.pos_act_buffer = deque(maxlen=self.window_size)
        self.vel_des_buffer = deque(maxlen=self.window_size)
        self.vel_act_buffer = deque(maxlen=self.window_size)
        self.torque_cmd_buffer = deque(maxlen=self.window_size)
        
        # 创建图形
        plt.ion()
        self.fig, self.axes = plt.subplots(3, 1, figsize=self.figure_size)
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

def run_test(arm_length, arm_mass, weight_mass, weight_position, test_name, config):
    """运行单次仿真测试（带实时波形显示）"""
    print(f"\n{'='*60}")
    print(f"测试配置: {test_name}")
    print(f"{'='*60}")
    print(f"摆杆长度: {arm_length:.3f} m")
    print(f"摆杆质量: {arm_mass:.3f} kg")
    print(f"重量块质量: {weight_mass:.1f} kg")
    print(f"重量块位置: {weight_position:.3f} m")
    
    # 从配置获取控制参数
    kp = config['control']['kp']
    kd = config['control']['kd']
    torque_limit = config['control']['torque_limit']
    
    print(f"控制参数: Kp={kp}, Kd={kd}")
    print(f"力矩限制: ±{torque_limit} Nm")
    
    # 计算理论转动惯量
    arm_center = arm_length / 2
    I_link1 = 0.0001
    I_arm = (1/12) * arm_mass * arm_length**2 + arm_mass * arm_center**2
    I_weight = weight_mass * weight_position**2 if weight_mass > 0 else 0
    I_total_theoretical = I_link1 + I_arm + I_weight
    print(f"理论转动惯量: {I_total_theoretical:.6f} kg·m²")
    print(f"{'='*60}\n")
    
    # 创建模型
    xml_string = create_model_xml(arm_length, arm_mass, weight_mass, weight_position, config)
    model = mujoco.MjModel.from_xml_string(xml_string)
    data = mujoco.MjData(model)
    
    # 获取MuJoCo计算的转动惯量
    data.qpos[0] = 0  # 设置初始位置
    data.qvel[0] = 0
    data.qacc[0] = 0
    mujoco.mj_forward(model, data)
    
    # 计算质量矩阵M(q)
    M = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, M, data.qM)
    I_total_mujoco = M[0, 0]
    
    print(f"MuJoCo转动惯量: {I_total_mujoco:.6f} kg·m²")
    print(f"计算误差: {abs(I_total_mujoco - I_total_theoretical)/I_total_theoretical*100:.2f}%")
    print(f"{'='*60}\n")
    
    # 创建实时绘图器
    plotter = RealtimePlotter(config)
    
    # 记录数据
    time_data = []
    pos_des_data = []
    pos_data = []
    vel_des_data = []
    vel_data = []
    torque_data = []
    
    # 从配置获取测试参数
    test_duration = config['test_signal']['duration']
    f_start = config['test_signal']['f_start']
    f_end = config['test_signal']['f_end']
    amplitude = config['test_signal']['amplitude']
    plot_update_interval = config['display']['plot_update_interval']
    
    # 运行仿真
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 45
        viewer.cam.elevation = -20
        viewer.cam.distance = 1.5
        viewer.cam.lookat = np.array([0.0, 0.0, 0.2])
        
        plot_update_counter = 0
        
        while viewer.is_running() and data.time < test_duration and plotter.running:
            step_start = time.time()
            
            # Chirp信号 (按readme要求: 0.1Hz-10Hz, ±10°, 20s)
            current_time = data.time
            desired_pos = chirp_signal(current_time, f_start, f_end, test_duration, amplitude)
            
            # 计算期望速度（数值微分）
            dt = model.opt.timestep
            if len(time_data) > 0:
                desired_vel = (desired_pos - pos_des_data[-1]) / dt
            else:
                desired_vel = 0.0
            
            # PD控制
            pos_error = desired_pos - data.qpos[0]
            vel_error = desired_vel - data.qvel[0]
            torque = kp * pos_error + kd * vel_error
            torque = np.clip(torque, -torque_limit, torque_limit)
            
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
        'inertia_theoretical': I_total_theoretical,
        'inertia_mujoco': I_total_mujoco
    }

def main():
    """主函数：测试三种配置"""
    
    # 加载配置
    config = load_config()
    
    # 从配置获取材料参数
    arm_radius = config['material']['arm_radius']
    density_aluminum = config['material']['density_aluminum']
    
    # 从配置获取测试配置
    test_configs = config['test_configs']
    
    # 计算每个配置的摆杆质量
    for test_config in test_configs:
        arm_length = test_config['arm_length']
        arm_volume = np.pi * arm_radius**2 * arm_length
        test_config['arm_mass'] = arm_volume * density_aluminum
    
    print("\n" + "="*60)
    print("MuJoCo仿真测试 - 不同惯量配置验证（带实时波形显示）")
    print("="*60)
    print(f"将依次测试{len(test_configs)}种配置，每次测试{config['test_signal']['duration']}秒")
    print(f"控制信号: Chirp {config['test_signal']['f_start']}~{config['test_signal']['f_end']}Hz, 幅值±{config['test_signal']['amplitude']}°")
    print(f"PD控制: Kp={config['control']['kp']}, Kd={config['control']['kd']}")
    print(f"力矩限制: ±{config['control']['torque_limit']} Nm")
    print("\n实时显示:")
    print("  - 位置跟踪曲线（期望 vs 实际）")
    print("  - 速度跟踪曲线（期望 vs 实际）")
    print("  - 控制力矩曲线")
    print("="*60)
    
    # 依次测试
    results = []
    for i, test_config in enumerate(test_configs, 1):
        print(f"\n[{i}/{len(test_configs)}] 准备测试: {test_config['name']}")
        input("按Enter开始...")
        result = run_test(
            test_config['arm_length'],
            test_config['arm_mass'],
            test_config['weight_mass'],
            test_config['weight_position'],
            test_config['name'],
            config
        )
        results.append(result)
    
    # 显示转动惯量汇总
    print(f"\n{'='*80}")
    print("转动惯量汇总对比")
    print(f"{'='*80}")
    print(f"{'配置':<12} {'理论计算':<18} {'MuJoCo计算':<18} {'误差%':<10}")
    print(f"{'-'*80}")
    
    for i, result in enumerate(results, 1):
        config_name = f"配置{i}"
        theoretical = result['inertia_theoretical']
        mujoco = result['inertia_mujoco']
        error = abs(mujoco - theoretical) / theoretical * 100
        
        print(f"{config_name:<12} {theoretical:<18.6f} {mujoco:<18.6f} {error:<10.2f}")
    
    print(f"{'='*80}")
    print("说明：")
    print("- 理论计算：使用平行轴定理手动计算")
    print("- MuJoCo计算：从质量矩阵M(q)提取")
    print("- 误差应该很小（<1%），证明计算正确")
    print(f"{'='*80}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n测试已中断")

