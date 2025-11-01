#!/usr/bin/env python3
"""
摆杆负载测试 - 验证电机驱动能力
核心目标：确认电机能否稳定驱动不同负载的摆杆完成位置跟踪任务

功能说明：
1. 动态生成不同负载配置的MuJoCo模型
2. 使用Chirp扫频信号测试位置跟踪性能
3. 实时显示位置、速度、力矩跟踪波形
4. 分析电机扭矩利用率，确定最大安全负载
5. 支持多电机配置，每个电机有独立的Kp、Kd和max_torque参数
6. 通过配置文件选择要测试的电机

使用方法：
1. 在config/test_params.yaml中配置电机参数（包含name, joint_name, kp, kd, max_torque）
2. 设置selected_motor为要测试的电机名称
3. 运行脚本，程序会使用选定电机的参数进行负载测试

"""

import mujoco
import mujoco.viewer
import numpy as np
import time
import matplotlib.pyplot as plt
import yaml
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_config():
    """
    加载配置文件
    
    Returns:
        dict: 包含所有仿真参数的配置字典
    """
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'test_params.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# 加载配置文件
config = load_config()

# ==================== 全局配置参数 ====================
# 摆杆物理参数
ARM_LENGTH = config['arm']['length']        # 摆杆长度 (m)
ARM_DIAMETER = config['arm']['diameter']   # 摆杆直径 (m)
ARM_DENSITY = config['arm']['density']      # 摆杆材质密度 (kg/m³)

# 负载测试范围（包含0kg空载情况）
LOAD_MASSES = [0] + config['load_masses']   # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] kg

# 获取选定的电机参数
selected_motor_name = config.get('selected_motor', 'PA100_18')
motors_list = config.get('motors', [])

# 查找选定的电机
selected_motor = None
for motor in motors_list:
    if motor['name'] == selected_motor_name:
        selected_motor = motor
        break

if selected_motor is None:
    print(f"错误：未找到电机 '{selected_motor_name}'，使用默认参数")
    MOTOR_NAME = "Unknown"
    MOTOR_JOINT = "unknown_joint"
    MOTOR_MAX_TORQUE = 100
    KP = 80.0
    KD = 10.0
else:
    MOTOR_NAME = selected_motor['name']
    MOTOR_JOINT = selected_motor['joint_name']
    MOTOR_MAX_TORQUE = selected_motor['max_torque']
    KP = selected_motor['kp']
    KD = selected_motor['kd']

# 测试信号参数 (按readme要求: Chirp 0.1Hz-10Hz, ±10°, 20s)
TEST_DURATION = config['test_signal']['duration']      # 测试持续时间 (s)
CHIRP_F_START = config['test_signal']['f_start']       # 起始频率 (Hz)
CHIRP_F_END = config['test_signal']['f_end']           # 结束频率 (Hz)
CHIRP_AMPLITUDE = config['test_signal']['amplitude']    # 信号幅值 (度)

# 实时绘图参数
PLOT_UPDATE_INTERVAL = config['display']['plot_update_interval']  # 绘图更新间隔

# 负载参数（负载半径将根据质量和密度动态计算）
LOAD_DENSITY = config['load'].get('density', ARM_DENSITY)  # 负载材质密度 (kg/m³)，默认与摆杆相同
# =================================================


def calculate_arm_mass(length, diameter, density):
    """
    计算摆杆质量
    
    Args:
        length (float): 摆杆长度 (m)
        diameter (float): 摆杆直径 (m) 
        density (float): 材质密度 (kg/m³)
        
    Returns:
        float: 摆杆质量 (kg)
    """
    radius = diameter / 2
    volume = np.pi * radius**2 * length  # 圆柱体体积公式
    mass = volume * density
    return mass


def create_model_xml(arm_length, arm_mass, load_mass):
    """
    创建MuJoCo模型XML字符串
    
    Args:
        arm_length (float): 摆杆长度 (m)
        arm_mass (float): 摆杆质量 (kg)
        load_mass (float): 负载质量 (kg)
        
    Returns:
        str: MuJoCo模型XML字符串
    """
    
    radius = ARM_DIAMETER / 2
    
    # 计算摆杆转动惯量（圆柱体，质心坐标系）
    # 注意：圆柱体通过quat="0.707107 0 0.707107 0"旋转了90度（绕y轴）
    # 旋转后：圆柱体轴线沿x轴方向
    # I_xx: 绕x轴（圆柱体轴线），仅与半径相关
    I_xx = 0.5 * arm_mass * radius**2
    # I_yy = I_zz: 绕垂直于轴线的轴，包含长度和半径项
    I_yy = (1/12) * arm_mass * arm_length**2 + (1/4) * arm_mass * radius**2
    I_zz = I_yy  # 圆柱体对称性
    
    # 负载部分（只有当load_mass > 0时才添加）
    load_body = ""
    if load_mass > 0:
        # 根据质量和密度计算球体半径（实心球体）
        # 球体体积: V = (4/3)πr³
        # 质量: m = ρV = ρ(4/3)πr³
        # 反推半径: r = ∛(3m/(4πρ))
        load_radius = (3 * load_mass / (4 * np.pi * LOAD_DENSITY)) ** (1/3)
        
        I_load = (2/5) * load_mass * load_radius**2  # 球体转动惯量
        load_body = f'''
          <body name="load" pos="{arm_length/2} 0 0">
            <geom type="sphere" size="{load_radius}" rgba="0.8 0.1 0.1 1"/>
            <inertial pos="0 0 0" mass="{load_mass}" 
                      diaginertia="{I_load:.8f} {I_load:.8f} {I_load:.8f}"/>
          </body>'''
    
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
          {load_body}
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
    """
    瞬时频率 f(t) = f_start + k*t(线性变化），
    相位 φ(t) = 2π ∫f(t)dt(积分结果），即 2π*(f_start*t + 0.5*k*t²)
    np.deg2rad 将振幅从 “度” 转换为 “弧度
    """
    k = (f_end - f_start) / duration
    phase = 2 * np.pi * (f_start * t + 0.5 * k * t**2)
    return np.deg2rad(amplitude_deg) * np.sin(phase)


class RealtimePlotter:
    """实时绘制跟踪曲线"""
    
    def __init__(self, config):
        """初始化实时绘图器"""
        self.window_size = config['display']['window_size']
        self.figure_size = config['display']['figure_size']
        self.keep_all_history = config['display'].get('keep_all_history', True)  # 默认保留所有历史
        
        # 数据缓冲区 - 根据配置选择是否限制长度
        if self.keep_all_history:
            # 保留所有历史数据
            self.time_buffer = []
            self.pos_des_buffer = []
            self.pos_act_buffer = []
            self.vel_des_buffer = []
            self.vel_act_buffer = []
            self.torque_cmd_buffer = []
            self.torque_frc_buffer = []
        else:
            # 只保留最近window_size个数据点（使用deque实现滑动窗口）
            from collections import deque
            self.time_buffer = deque(maxlen=self.window_size)
            self.pos_des_buffer = deque(maxlen=self.window_size)
            self.pos_act_buffer = deque(maxlen=self.window_size)
            self.vel_des_buffer = deque(maxlen=self.window_size)
            self.vel_act_buffer = deque(maxlen=self.window_size)
            self.torque_cmd_buffer = deque(maxlen=self.window_size)
            self.torque_frc_buffer = deque(maxlen=self.window_size)
        
        # 创建图形
        plt.ion()
        self.fig, self.axes = plt.subplots(3, 1, figsize=self.figure_size)
        
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
        self.line_torque_cmd, = self.ax_torque.plot([], [], 'r-', label='指令力矩', linewidth=2, alpha=0.8)
        self.line_torque_frc, = self.ax_torque.plot([], [], 'b-', label='反馈力矩', linewidth=1.5)
        self.ax_torque.set_xlabel('时间 (s)', fontsize=11)
        self.ax_torque.set_ylabel('力矩 (Nm)', fontsize=11)
        self.ax_torque.legend(loc='upper right')
        self.ax_torque.grid(True, alpha=0.3)
        self.ax_torque.set_title('力矩跟踪 (指令 vs 反馈)')
        
        plt.tight_layout()
        plt.show(block=False)
        
        self.running = True
    
    def update_data(self, time_val, pos_des, pos_act, vel_des, vel_act, torque_cmd, torque_frc):
        """
        更新数据缓冲区
        
        Args:
            time_val (float): 当前仿真时间 (s)
            pos_des (float): 期望位置 (rad)
            pos_act (float): 实际位置 (rad)
            vel_des (float): 期望速度 (rad/s)
            vel_act (float): 实际速度 (rad/s)
            torque_cmd (float): 控制力矩指令 (Nm)
            torque_frc (float): 执行器实际力矩反馈 (Nm)
        """
        self.time_buffer.append(time_val)
        self.pos_des_buffer.append(pos_des)
        self.pos_act_buffer.append(pos_act)
        self.vel_des_buffer.append(vel_des)
        self.vel_act_buffer.append(vel_act)
        self.torque_cmd_buffer.append(torque_cmd)
        self.torque_frc_buffer.append(torque_frc)
    
    def update_plot(self):
        """
        更新图形显示
        
        功能:
        1. 更新位置跟踪曲线 (期望位置 vs 实际位置)
        2. 更新速度跟踪曲线 (期望速度 vs 实际速度)  
        3. 更新控制力矩曲线
        4. 根据keep_all_history配置选择显示模式：
           - True: 固定Y轴范围，X轴显示完整时间范围（保留所有历史波形）
           - False: 自动调整坐标轴范围（滑动窗口，只显示最近数据）
        5. 刷新图形显示
        """
        if len(self.time_buffer) < 2:  # 数据点不足时不更新
            return
        
        time_array = np.array(self.time_buffer)
        pos_des_array = np.array(self.pos_des_buffer)
        pos_act_array = np.array(self.pos_act_buffer)
        vel_des_array = np.array(self.vel_des_buffer)
        vel_act_array = np.array(self.vel_act_buffer)
        torque_cmd_array = np.array(self.torque_cmd_buffer)
        torque_frc_array = np.array(self.torque_frc_buffer)
        
        # 更新位置跟踪曲线
        self.line_pos_des.set_data(time_array, pos_des_array)
        self.line_pos_act.set_data(time_array, pos_act_array)
        
        # 更新速度跟踪曲线
        self.line_vel_des.set_data(time_array, vel_des_array)
        self.line_vel_act.set_data(time_array, vel_act_array)
        
        # 更新力矩跟踪曲线 (指令 vs 反馈)
        self.line_torque_cmd.set_data(time_array, torque_cmd_array)
        self.line_torque_frc.set_data(time_array, torque_frc_array)
        
        # 根据配置选择坐标轴更新策略
        if self.keep_all_history:
            # 模式1: 保留所有历史数据，固定Y轴范围，X轴显示完整时间
            pos_max = max(np.max(np.abs(pos_des_array)), np.max(np.abs(pos_act_array)))
            self.ax_pos.set_xlim(0, max(time_array))
            self.ax_pos.set_ylim(-pos_max * 1.2, pos_max * 1.2)
            
            vel_max = max(np.max(np.abs(vel_des_array)), np.max(np.abs(vel_act_array)))
            if vel_max > 0:
                self.ax_vel.set_xlim(0, max(time_array))
                self.ax_vel.set_ylim(-vel_max * 1.2, vel_max * 1.2)
            
            torque_max = max(np.max(np.abs(torque_cmd_array)), np.max(np.abs(torque_frc_array)))
            if torque_max > 0:
                self.ax_torque.set_xlim(0, max(time_array))
                self.ax_torque.set_ylim(-torque_max * 1.2, torque_max * 1.2)
        else:
            # 模式2: 滑动窗口，自动调整坐标轴范围
            self.ax_pos.relim()
            self.ax_pos.autoscale_view()
            
            self.ax_vel.relim()
            self.ax_vel.autoscale_view()
            
            self.ax_torque.relim()
            self.ax_torque.autoscale_view()
        
        # 刷新图形显示
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def close(self):
        """关闭绘图窗口"""
        self.running = False
        plt.ioff()
        plt.close(self.fig)


def run_load_test(arm_mass, load_mass, show_plot=True):
    """运行单次负载测试（带实时波形显示）"""
    
    print(f"\n{'='*70}")
    if load_mass == 0:
        print(f"负载测试: 空载（仅摆杆）")
    else:
        print(f"负载测试: {load_mass} kg")
    print(f"{'='*70}")
    
    # 创建模型
    xml_string = create_model_xml(ARM_LENGTH, arm_mass, load_mass)
    
    # 保存XML文件到mjcf目录
    xml_filename = f'load_test_{load_mass}kg.xml' if load_mass > 0 else 'load_test_0kg_empty.xml'
    xml_path = os.path.join(os.path.dirname(__file__), '..', 'mjcf', xml_filename)
    with open(xml_path, 'w', encoding='utf-8') as f:
        f.write(xml_string)
    print(f"模型XML已保存至: mjcf/{xml_filename}")
    
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
    print(f"{'='*70}\n")
    
    # 创建实时绘图器
    plotter = None
    if show_plot:
        plotter = RealtimePlotter(config)
    
    # 记录数据
    time_data = []
    pos_des_data = []
    pos_act_data = []
    vel_des_data = []
    vel_act_data = []
    torque_cmd_data = []  # 指令力矩
    torque_frc_data = []  # 反馈力矩
    
    # 运行仿真
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 45
        viewer.cam.elevation = -20
        viewer.cam.distance = 0.8  # 从1.5调整到0.8，让模型更近更大
        viewer.cam.lookat = np.array([0.0, 0.0, 0.2])
        
        plot_update_counter = 0
        
        while viewer.is_running() and data.time < TEST_DURATION:
            if plotter and not plotter.running:
                break
                
            step_start = time.time()
            
            t = data.time
            
            # Chirp信号 (0.1Hz-10Hz, ±10°, 20s)
            pos_des = chirp_signal(t, CHIRP_F_START, CHIRP_F_END, TEST_DURATION, CHIRP_AMPLITUDE)
            
            # 期望速度指令为0
            vel_des = 0.0
            
            # PD控制：转换为期望力矩 (标准公式)
            default_joint_pos = 0.0  # MuJoCo中关节默认位置通常为0
            pos_error = pos_des - data.qpos[0] + default_joint_pos  # 位置误差 (包含默认位置)
            vel_error = vel_des - data.qvel[0]  # 速度误差 (期望速度为0)
            torque = KP * pos_error + KD * vel_error  # PD控制输出
            torque = np.clip(torque, -MOTOR_MAX_TORQUE, MOTOR_MAX_TORQUE)  # 限制在电机扭矩范围内
            
            data.ctrl[0] = torque
            
            # 执行仿真步进
            mujoco.mj_step(model, data)
            
            # 获取执行器实际力矩反馈 (actuator_force)
            # data.actuator_force[0] 包含执行器实际输出的力矩
            torque_frc = data.actuator_force[0] if model.nu > 0 else 0.0
            
            # 记录数据
            time_data.append(t)
            pos_des_data.append(pos_des)
            pos_act_data.append(data.qpos[0])
            vel_des_data.append(vel_des)
            vel_act_data.append(data.qvel[0])
            torque_cmd_data.append(torque)  # 指令力矩
            torque_frc_data.append(torque_frc)  # 反馈力矩
            
            # 更新实时绘图
            if plotter:
                plotter.update_data(t, pos_des, data.qpos[0], vel_des, data.qvel[0], torque, torque_frc)
                plot_update_counter += 1
                if plot_update_counter >= PLOT_UPDATE_INTERVAL:
                    plotter.update_plot()
                    plot_update_counter = 0
            
            viewer.sync()
            
            time_until_next = model.opt.timestep - (time.time() - step_start)
            if time_until_next > 0:
                time.sleep(time_until_next)
    
    # 最后更新一次图形
    if plotter:
        plotter.update_plot()
    
    # ========== 测试结果分析 ==========
    # 计算跟踪误差
    pos_error_array = np.array(pos_des_data) - np.array(pos_act_data)  # 位置跟踪误差
    vel_error_array = np.array(vel_des_data) - np.array(vel_act_data)  # 速度跟踪误差
    torque_cmd_array = np.array(torque_cmd_data)  # 指令力矩数组
    torque_frc_array = np.array(torque_frc_data)  # 反馈力矩数组
    torque_error_array = torque_cmd_array - torque_frc_array  # 力矩跟踪误差
    
    # 计算关键性能指标
    max_torque_cmd = np.max(np.abs(torque_cmd_array))  # 最大指令力矩
    max_torque_frc = np.max(np.abs(torque_frc_array))  # 最大反馈力矩
    rms_torque_cmd = np.sqrt(np.mean(torque_cmd_array**2))  # RMS指令力矩
    rms_torque_frc = np.sqrt(np.mean(torque_frc_array**2))  # RMS反馈力矩
    max_torque_error = np.max(np.abs(torque_error_array))  # 最大力矩误差
    rms_torque_error = np.sqrt(np.mean(torque_error_array**2))  # RMS力矩误差
    max_pos_error = np.max(np.abs(pos_error_array))  # 最大位置误差
    rms_pos_error = np.sqrt(np.mean(pos_error_array**2))  # RMS位置误差
    
    print(f"\n测试结果:")
    print(f"{'='*50}")
    print(f"位置跟踪:")
    print(f"  RMS误差: {np.sqrt(np.mean(pos_error_array**2)):.6f} rad ({np.rad2deg(np.sqrt(np.mean(pos_error_array**2))):.3f}°)")
    print(f"  最大误差: {max_pos_error:.6f} rad ({np.rad2deg(max_pos_error):.3f}°)")
    
    print(f"\n速度跟踪:")
    print(f"  RMS误差: {np.sqrt(np.mean(vel_error_array**2)):.6f} rad/s")
    print(f"  最大速度: {np.max(np.abs(vel_act_data)):.3f} rad/s")
    
    print(f"\n控制力矩:")
    print(f"  指令力矩 - 最大: {max_torque_cmd:.2f} Nm, RMS: {rms_torque_cmd:.2f} Nm")
    print(f"  反馈力矩 - 最大: {max_torque_frc:.2f} Nm, RMS: {rms_torque_frc:.2f} Nm")
    print(f"  力矩误差 - 最大: {max_torque_error:.2f} Nm, RMS: {rms_torque_error:.2f} Nm")
    print(f"  扭矩利用率: {max_torque_cmd/MOTOR_MAX_TORQUE*100:.1f}%")
    
    # ========== 电机驱动能力评估 ==========
    # 根据最大控制力矩判断电机是否能稳定驱动负载
    if max_torque_cmd < MOTOR_MAX_TORQUE * 0.9:  # 扭矩利用率 < 90%
        print(f"  ✓ 电机能稳定驱动")
        status = "OK"  # 安全状态
    elif max_torque_cmd < MOTOR_MAX_TORQUE:  # 90% ≤ 扭矩利用率 < 100%
        print(f"  ⚠️ 接近极限")
        status = "MARGINAL"  # 临界状态
    else:  # 扭矩利用率 ≥ 100%
        print(f"  ✗ 超出电机能力！")
        status = "FAIL"  # 失败状态
    
    print(f"{'='*50}\n")
    
    # 等待用户查看图形
    if plotter:
        input("按Enter键关闭波形图并继续...")
        plotter.close()
    
    return {
        'load_mass': load_mass,
        'inertia': inertia,
        'max_torque': max_torque_cmd,  # 使用指令力矩作为评估依据
        'max_torque_frc': max_torque_frc,  # 反馈力矩
        'rms_torque': rms_torque_cmd,
        'max_torque_error': max_torque_error,  # 力矩跟踪误差
        'rms_torque_error': rms_torque_error,
        'max_pos_error': max_pos_error,
        'rms_pos_error': rms_pos_error,
        'status': status
    }


def main():
    """
    主函数 - 摆杆负载测试程序入口
    
    功能流程:
    1. 显示测试参数配置
    2. 依次测试不同负载配置 (0~10kg)
    3. 实时显示跟踪波形
    4. 分析电机驱动能力
    5. 生成结构设计参考报告
    6. 保存测试结果到CSV文件
    """
    
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
    
    print(f"\n测试电机:")
    print(f"  电机型号: {MOTOR_NAME}")
    print(f"  对应关节: {MOTOR_JOINT}")
    print(f"  最大扭矩: {MOTOR_MAX_TORQUE} Nm")
    print(f"  控制参数: Kp={KP}, Kd={KD}")
    
    print(f"\n测试信号:")
    print(f"  Chirp {CHIRP_F_START}~{CHIRP_F_END} Hz, 幅值±{CHIRP_AMPLITUDE}°")
    print(f"  持续时间: {TEST_DURATION} 秒")
    print(f"  信号类型: 扫频信号 (0.1Hz→10Hz, 20秒)")
    
    print(f"\n负载范围: {LOAD_MASSES[0]}~{LOAD_MASSES[-1]} kg (包含0kg空载)")
    print("="*70)
    
    # 运行测试
    results = []
    for load_mass in LOAD_MASSES:
        if load_mass == 0:
            input(f"\n按Enter开始测试 空载（仅摆杆）...")
        else:
            input(f"\n按Enter开始测试 {load_mass}kg 负载...")
        result = run_load_test(arm_mass, load_mass, show_plot=True)
        results.append(result)
    
    # 生成汇总报告
    print(f"\n{'='*90}")
    print("测试汇总报告 - 给结构设计部门参考")
    print(f"{'='*90}")
    
    # 显示当前测试电机参数
    print(f"当前测试电机参数:")
    print(f"  电机型号: {MOTOR_NAME}")
    print(f"  对应关节: {MOTOR_JOINT}")
    print(f"  最大扭矩: {MOTOR_MAX_TORQUE} Nm")
    print(f"  控制参数: Kp={KP}, Kd={KD}")
    print(f"{'='*90}")
    
    print(f"{'负载(kg)':<10} {'转动惯量(kg·m²)':<18} {'最大扭矩(Nm)':<15} {'扭矩利用率':<12} {'状态':<10}")
    print(f"{'-'*90}")
    
    for result in results:
        status_symbol = {
            'OK': '✓ 正常',
            'MARGINAL': '⚠️ 接近极限',
            'FAIL': '✗ 超限'
        }[result['status']]
        
        load_str = "空载" if result['load_mass'] == 0 else str(result['load_mass'])
        
        # print(f"{load_str:<10} "
        #       f"{result['inertia']:<18.6f} "
        #       f"{result['max_torque']:<15.2f} "
        #       f"{result['max_torque']/MOTOR_MAX_TORQUE*100:<12.1f}% "
        #       f"{status_symbol:<10}")
        print(f"{load_str:<12} "  # 从10→12（增加2）
              f"{result['inertia']:<21.6f} "  # 从18→21（增加3）
              f"{result['max_torque']:<18.2f} "  # 从15→18（增加3）
              f"{result['max_torque']/MOTOR_MAX_TORQUE*100:<15.1f}% "  # 从12→15（增加3）
              f"{status_symbol:<12}")  # 从10→12（增加2）
    
    # 确定最大可用负载（包含OK和MARGINAL状态）
    acceptable_results = [r for r in results if r['status'] in ['OK', 'MARGINAL']]
    if acceptable_results:
        max_safe_load = max([r['load_mass'] for r in acceptable_results])
        
        # 计算所有可接受结果的转动惯量范围
        inertias_acceptable = [r['inertia'] for r in acceptable_results]
        min_inertia = min(inertias_acceptable)
        max_inertia = max(inertias_acceptable)
        
        print(f"\n{'='*90}")
        print(f"结论:")
        print(f"  电机最大稳定驱动负载: {max_safe_load} kg")
        print(f"  对应转动惯量: {[r['inertia'] for r in results if r['load_mass']==max_safe_load][0]:.6f} kg·m²")
        print(f"  对应最大扭矩: {[r['max_torque'] for r in results if r['load_mass']==max_safe_load][0]:.2f} Nm")
        print(f"\n  推荐工装设计负载范围: 0~{max_safe_load} kg (包含空载)")
        print(f"  对应转动惯量范围: {min_inertia:.6f} ~ {max_inertia:.6f} kg·m²")
        print(f"{'='*90}")
    else:
        print(f"\n⚠️ 警告: 所有负载都超出电机能力！")
    
    # 保存结果
    import csv
    with open('load_test_results.csv', 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['load_mass', 'inertia', 'max_torque', 'max_torque_frc',
                                               'rms_torque', 'max_torque_error', 'rms_torque_error',
                                               'max_pos_error', 'rms_pos_error', 'status'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n详细结果已保存至: load_test_results.csv")
    
    # # ========== 电机型号对比分析 ==========
    # if 'motors' in config:
    #     print(f"\n{'='*100}")
    #     print("不同电机型号驱动能力对比分析")
    #     print(f"{'='*100}")
        
    #     motor_comparison = []
        
    #     for motor_info in config['motors']:
    #         motor_name = motor_info['name']
    #         motor_max_torque = motor_info['max_torque']
            
    #         # 找到该电机能驱动的最大负载
    #         max_load = 0
    #         max_load_inertia = 0
    #         max_load_torque = 0
            
    #         for result in results:
    #             # 判断该负载是否在电机能力范围内（90%安全裕度）
    #             if result['max_torque'] < motor_max_torque * 0.9:
    #                 if result['load_mass'] > max_load:
    #                     max_load = result['load_mass']
    #                     max_load_inertia = result['inertia']
    #                     max_load_torque = result['max_torque']
            
    #         motor_comparison.append({
    #             'motor_name': motor_name,
    #             'max_torque': motor_max_torque,
    #             'max_safe_load': max_load,
    #             'max_inertia': max_load_inertia,
    #             'required_torque': max_load_torque
    #         })
        
    #     # 按最大负载排序
    #     motor_comparison.sort(key=lambda x: x['max_safe_load'], reverse=True)
        
    #     # 打印对比表格
    #     print(f"{'电机型号':<15} {'峰值扭矩(Nm)':<12} {'最大负载(kg)':<12} {'转动惯量(kg·m²)':<18} {'需求扭矩(Nm)':<14}")
    #     print(f"{'-'*90}")
        
    #     for mc in motor_comparison:
    #         if mc['max_safe_load'] > 0:
    #             print(f"{mc['motor_name']:<20} "  # 从15→20
    #                 f"{mc['max_torque']:<16.2f} "  # 从12→16
    #                 f"{mc['max_safe_load']:<16} "  # 从12→16
    #                 f"{mc['max_inertia']:<24.6f} "  # 从18→24
    #                 f"{mc['required_torque']:<20.2f}")  # 从14→20
    #         else:
    #             print(f"{mc['motor_name']:<20} "
    #                 f"{mc['max_torque']:<16.2f} "
    #                 f"{'无法驱动':<16} "
    #                 f"{'-':<24} "
    #                 f"{'-':<20}")
        
    #     # 保存电机对比结果
    #     with open('motor_comparison.csv', 'w', newline='', encoding='utf-8-sig') as f:
    #         writer = csv.DictWriter(f, fieldnames=['motor_name', 'max_torque', 'max_safe_load', 
    #                                                'max_inertia', 'required_torque'])
    #         writer.writeheader()
    #         writer.writerows(motor_comparison)
        
    #     print(f"\n电机对比结果已保存至: motor_comparison.csv")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n测试已中断")

