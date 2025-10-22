#!/usr/bin/env python3
"""
负载转动惯量测试仿真程序模板
基于100Nm峰值扭矩和0.3m摆杆长度限制设计
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
import matplotlib.pyplot as plt
from collections import deque

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class LoadInertiaTest:
    """负载转动惯量测试类"""
    
    def __init__(self):
        # 电机参数
        self.peak_torque = 100.0  # Nm 峰值扭矩
        self.rated_torque = 50.0   # Nm 额定扭矩
        
        # 摆杆参数
        self.max_arm_length = 0.3  # m 最大摆杆长度
        self.arm_radius = 0.015    # m 摆杆半径
        self.arm_density = 2700    # kg/m³ 铝材密度
        
        # 测试配置
        self.test_configs = self._generate_test_configs()
        
    def _generate_test_configs(self):
        """生成测试配置"""
        configs = []
        
        # 空载配置
        configs.append({
            'name': '空载测试',
            'arm_length': 0.2,
            'weight_mass': 0.0,
            'weight_position': 0.2,
            'expected_inertia': 0.001,  # kg·m²
            'max_torque_expected': 5.0   # Nm
        })
        
        # 中负载配置
        configs.append({
            'name': '中负载测试',
            'arm_length': 0.25,
            'weight_mass': 2.0,
            'weight_position': 0.25,
            'expected_inertia': 0.15,   # kg·m²
            'max_torque_expected': 30.0  # Nm
        })
        
        # 大负载配置
        configs.append({
            'name': '大负载测试',
            'arm_length': 0.3,
            'weight_mass': 4.0,
            'weight_position': 0.3,
            'expected_inertia': 0.4,    # kg·m²
            'max_torque_expected': 80.0  # Nm
        })
        
        return configs
    
    def create_model_xml(self, config):
        """根据配置创建MJCF模型"""
        arm_length = config['arm_length']
        weight_mass = config['weight_mass']
        weight_pos = config['weight_position']
        
        # 计算摆杆质量
        arm_volume = np.pi * self.arm_radius**2 * arm_length
        arm_mass = arm_volume * self.arm_density
        
        # 计算转动惯量
        arm_center = arm_length / 2
        I_arm = (1/12) * arm_mass * arm_length**2 + arm_mass * arm_center**2
        I_weight = weight_mass * weight_pos**2 if weight_mass > 0 else 0
        I_total = 0.0001 + I_arm + I_weight  # link1 + arm + weight
        
        xml = f"""
<mujoco model="load_inertia_test">
  <option timestep="0.001" gravity="0 0 -9.81"/>
  
  <asset>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1=".1 .2 .3" rgb2=".2 .3 .4"/>
    <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance=".2"/>
    <material name="base_mat" rgba="0.3 0.3 0.3 1"/>
    <material name="link1_mat" rgba="0.6 0.3 0.1 1"/>
    <material name="arm_mat" rgba="0.1 0.5 0.8 1"/>
    <material name="weight_mat" rgba="0.8 0.1 0.1 1"/>
  </asset>
  
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1" material="grid"/>
    <light directional="true" pos="0 0 3" dir="0 0 -1"/>
    
    <!-- 基座 -->
    <body name="base_link" pos="0 0 0.03">
      <geom name="base_geom" type="cylinder" size="0.075 0.03" material="base_mat"/>
      
      <!-- 电机连接杆 -->
      <body name="link1" pos="0 0 0.07">
        <joint name="joint1" type="hinge" axis="0 0 1" limited="false" damping="0.1"/>
        <geom name="link1_geom" type="cylinder" size="0.015 0.04" material="link1_mat"/>
        <inertial pos="0 0 0" mass="0.5" diaginertia="0.001 0.001 0.0001"/>
        
        <!-- 摆杆 -->
        <body name="arm" pos="{arm_center} 0 0.04">
          <geom name="arm_geom" type="cylinder" size="0.015 {arm_length/2}" 
                quat="0.707107 0 0.707107 0" material="arm_mat"/>
          <inertial pos="0 0 0" mass="{arm_mass:.3f}" diaginertia="{I_arm:.6f} {I_arm:.6f} 0.0001"/>
          
          <!-- 重量块 -->
          {"" if weight_mass <= 0 else f'''
          <body name="weight" pos="{weight_pos - arm_center} 0 0">
            <geom name="weight_geom" type="sphere" size="0.04" material="weight_mat"/>
            <inertial pos="0 0 0" mass="{weight_mass}" diaginertia="0.0005 0.0005 0.0005"/>
          </body>
          '''}
        </body>
      </body>
    </body>
  </worldbody>
  
  <actuator>
    <motor name="motor1" joint="joint1" gear="1" ctrllimited="true" ctrlrange="-{self.peak_torque} {self.peak_torque}"/>
  </actuator>
  
</mujoco>
"""
        return xml, I_total
    
    def chirp_signal(self, t, f_start, f_end, duration, amplitude):
        """Chirp扫频信号"""
        k = (f_end - f_start) / duration
        phase = 2 * np.pi * (f_start * t + 0.5 * k * t**2)
        return amplitude * np.sin(phase)
    
    def run_test(self, config, enable_plot=True):
        """运行单次测试"""
        print(f"\n{'='*70}")
        print(f"负载转动惯量测试: {config['name']}")
        print(f"{'='*70}")
        print(f"摆杆长度: {config['arm_length']:.3f} m")
        print(f"重量块质量: {config['weight_mass']:.1f} kg")
        print(f"重量块位置: {config['weight_position']:.3f} m")
        
        # 创建模型
        xml_string, actual_inertia = self.create_model_xml(config)
        model = mujoco.MjModel.from_xml_string(xml_string)
        data = mujoco.MjData(model)
        
        print(f"实际转动惯量: {actual_inertia:.6f} kg·m²")
        print(f"期望转动惯量: {config['expected_inertia']:.6f} kg·m²")
        print(f"误差: {abs(actual_inertia - config['expected_inertia'])/config['expected_inertia']*100:.1f}%")
        print(f"{'='*70}\n")
        
        # 创建绘图器
        plotter = None
        if enable_plot:
            plotter = self._create_plotter()
        
        # 测试参数
        test_duration = 15.0  # 秒
        f_start, f_end = 0.1, 5.0  # Hz
        amplitude = np.deg2rad(10)  # 10度
        
        # 控制参数
        kp = 100.0  # 根据100Nm峰值扭矩调整
        kd = 10.0
        
        # 数据记录
        time_data = []
        pos_des_data = []
        pos_act_data = []
        vel_des_data = []
        vel_act_data = []
        torque_data = []
        
        # 运行仿真
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.azimuth = 45
            viewer.cam.elevation = -20
            viewer.cam.distance = 1.5
            viewer.cam.lookat = np.array([0.0, 0.0, 0.2])
            
            update_counter = 0
            prev_pos_des = 0.0
            
            while viewer.is_running() and data.time < test_duration:
                step_start = time.time()
                
                t = data.time
                
                # 期望位置
                pos_des = self.chirp_signal(t, f_start, f_end, test_duration, amplitude)
                
                # 期望速度
                vel_des = (pos_des - prev_pos_des) / model.opt.timestep
                prev_pos_des = pos_des
                
                # PD控制
                pos_err = pos_des - data.qpos[0]
                vel_err = vel_des - data.qvel[0]
                torque = kp * pos_err + kd * vel_err
                torque = np.clip(torque, -self.peak_torque, self.peak_torque)
                
                data.ctrl[0] = torque
                
                # 记录数据
                time_data.append(t)
                pos_des_data.append(pos_des)
                pos_act_data.append(data.qpos[0])
                vel_des_data.append(vel_des)
                vel_act_data.append(data.qvel[0])
                torque_data.append(torque)
                
                # 更新绘图
                if plotter:
                    plotter.update(t, pos_des, data.qpos[0], vel_des, data.qvel[0], torque)
                    update_counter += 1
                    if update_counter >= 50:  # 每50步更新一次
                        plotter.refresh()
                        update_counter = 0
                
                # 步进仿真
                mujoco.mj_step(model, data)
                viewer.sync()
                
                # 实时控制
                time_until_next = model.opt.timestep - (time.time() - step_start)
                if time_until_next > 0:
                    time.sleep(time_until_next)
        
        # 最后更新绘图
        if plotter:
            plotter.refresh()
        
        # 分析结果
        self._analyze_results(time_data, pos_des_data, pos_act_data, 
                            vel_des_data, vel_act_data, torque_data, config)
        
        if plotter:
            input("按Enter关闭波形图...")
            plotter.close()
        
        return {
            'config': config,
            'actual_inertia': actual_inertia,
            'time': np.array(time_data),
            'position': np.array(pos_act_data),
            'velocity': np.array(vel_act_data),
            'torque': np.array(torque_data)
        }
    
    def _create_plotter(self):
        """创建绘图器"""
        class SimplePlotter:
            def __init__(self):
                self.data = {
                    'time': deque(maxlen=1000),
                    'pos_des': deque(maxlen=1000),
                    'pos_act': deque(maxlen=1000),
                    'vel_des': deque(maxlen=1000),
                    'vel_act': deque(maxlen=1000),
                    'torque': deque(maxlen=1000)
                }
                
                plt.ion()
                self.fig, self.axes = plt.subplots(3, 1, figsize=(12, 8))
                self.fig.suptitle('负载转动惯量测试 - 实时波形', fontsize=14, fontweight='bold')
                
                # 位置跟踪
                self.ax1 = self.axes[0]
                self.line_pos_des, = self.ax1.plot([], [], 'r--', label='期望位置', linewidth=2)
                self.line_pos_act, = self.ax1.plot([], [], 'b-', label='实际位置', linewidth=1.5)
                self.ax1.set_ylabel('位置 (rad)', fontsize=11)
                self.ax1.legend()
                self.ax1.grid(True, alpha=0.3)
                self.ax1.set_title('位置跟踪')
                
                # 速度跟踪
                self.ax2 = self.axes[1]
                self.line_vel_des, = self.ax2.plot([], [], 'r--', label='期望速度', linewidth=2)
                self.line_vel_act, = self.ax2.plot([], [], 'g-', label='实际速度', linewidth=1.5)
                self.ax2.set_ylabel('速度 (rad/s)', fontsize=11)
                self.ax2.legend()
                self.ax2.grid(True, alpha=0.3)
                self.ax2.set_title('速度跟踪')
                
                # 控制力矩
                self.ax3 = self.axes[2]
                self.line_torque, = self.ax3.plot([], [], 'm-', label='控制力矩', linewidth=1.5)
                self.ax3.axhline(y=100, color='r', linestyle=':', alpha=0.5, label='峰值扭矩')
                self.ax3.axhline(y=-100, color='r', linestyle=':', alpha=0.5)
                self.ax3.set_xlabel('时间 (s)', fontsize=11)
                self.ax3.set_ylabel('力矩 (Nm)', fontsize=11)
                self.ax3.legend()
                self.ax3.grid(True, alpha=0.3)
                self.ax3.set_title('控制力矩')
                
                plt.tight_layout()
                plt.show(block=False)
            
            def update(self, t, pos_des, pos_act, vel_des, vel_act, torque):
                self.data['time'].append(t)
                self.data['pos_des'].append(pos_des)
                self.data['pos_act'].append(pos_act)
                self.data['vel_des'].append(vel_des)
                self.data['vel_act'].append(vel_act)
                self.data['torque'].append(torque)
            
            def refresh(self):
                if len(self.data['time']) < 2:
                    return
                
                t_array = np.array(self.data['time'])
                
                self.line_pos_des.set_data(t_array, self.data['pos_des'])
                self.line_pos_act.set_data(t_array, self.data['pos_act'])
                self.ax1.relim()
                self.ax1.autoscale_view()
                
                self.line_vel_des.set_data(t_array, self.data['vel_des'])
                self.line_vel_act.set_data(t_array, self.data['vel_act'])
                self.ax2.relim()
                self.ax2.autoscale_view()
                
                self.line_torque.set_data(t_array, self.data['torque'])
                self.ax3.relim()
                self.ax3.autoscale_view()
                
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
            
            def close(self):
                plt.close(self.fig)
        
        return SimplePlotter()
    
    def _analyze_results(self, time_data, pos_des, pos_act, vel_des, vel_act, torque, config):
        """分析测试结果"""
        pos_errors = np.array(pos_des) - np.array(pos_act)
        vel_errors = np.array(vel_des) - np.array(vel_act)
        torques = np.array(torque)
        
        print(f"\n测试结果分析 - {config['name']}")
        print(f"{'='*50}")
        print(f"位置跟踪:")
        print(f"  RMS误差: {np.sqrt(np.mean(pos_errors**2)):.6f} rad ({np.rad2deg(np.sqrt(np.mean(pos_errors**2))):.3f}°)")
        print(f"  最大误差: {np.max(np.abs(pos_errors)):.6f} rad ({np.rad2deg(np.max(np.abs(pos_errors))):.3f}°)")
        
        print(f"\n速度跟踪:")
        print(f"  RMS误差: {np.sqrt(np.mean(vel_errors**2)):.6f} rad/s")
        print(f"  最大速度: {np.max(np.abs(vel_act)):.3f} rad/s")
        
        print(f"\n控制力矩:")
        print(f"  最大力矩: {np.max(np.abs(torques)):.3f} Nm")
        print(f"  平均力矩: {np.mean(np.abs(torques)):.3f} Nm")
        print(f"  RMS力矩: {np.sqrt(np.mean(torques**2)):.3f} Nm")
        print(f"  期望最大: {config['max_torque_expected']:.1f} Nm")
        
        # 判断是否达到期望性能
        max_torque_used = np.max(np.abs(torques))
        if max_torque_used > config['max_torque_expected'] * 0.8:
            print(f"  ⚠️  力矩使用率: {max_torque_used/config['max_torque_expected']*100:.1f}% (接近极限)")
        else:
            print(f"  ✓ 力矩使用率: {max_torque_used/config['max_torque_expected']*100:.1f}% (正常)")
        
        print(f"{'='*50}\n")
    
    def run_all_tests(self, enable_plot=True):
        """运行所有测试"""
        print("负载转动惯量测试仿真程序")
        print("="*70)
        print("电机参数: 峰值扭矩100Nm, 额定扭矩50Nm")
        print("摆杆限制: 最大长度0.3m")
        print("测试信号: Chirp 0.1~5Hz, 幅值±10°")
        print("="*70)
        
        results = []
        for i, config in enumerate(self.test_configs, 1):
            print(f"\n[{i}/{len(self.test_configs)}] 准备测试: {config['name']}")
            input("按Enter开始...")
            
            result = self.run_test(config, enable_plot)
            results.append(result)
        
        # 汇总结果
        self._print_summary(results)
        return results
    
    def _print_summary(self, results):
        """打印汇总结果"""
        print("\n" + "="*70)
        print("测试汇总")
        print("="*70)
        
        for result in results:
            config = result['config']
            actual_inertia = result['actual_inertia']
            max_torque = np.max(np.abs(result['torque']))
            
            print(f"\n{config['name']}:")
            print(f"  转动惯量: {actual_inertia:.6f} kg·m²")
            print(f"  最大力矩: {max_torque:.1f} Nm")
            print(f"  力矩利用率: {max_torque/self.peak_torque*100:.1f}%")
        
        print(f"\n{'='*70}")


def main():
    """主函数"""
    test_suite = LoadInertiaTest()
    
    try:
        results = test_suite.run_all_tests(enable_plot=True)
        print("\n所有测试完成！")
        
    except KeyboardInterrupt:
        print("\n\n测试已中断")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
