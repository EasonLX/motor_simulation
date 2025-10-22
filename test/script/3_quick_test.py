import mujoco
import mujoco.viewer
import numpy as np
import time
import matplotlib.pyplot as plt
from collections import deque

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

"""
快速测试脚本 - 单配置实时波形显示
用于快速验证和调试
"""

def create_simple_model():
    """创建简单测试模型"""
    xml = """
<mujoco model="simple_arm">
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
        
        <body name="arm" pos="0.125 0 0.04">
          <geom type="cylinder" size="0.015 0.125" quat="0.707107 0 0.707107 0" rgba="0.1 0.5 0.8 1"/>
          <inertial pos="0 0 0" mass="0.2" diaginertia="0.002 0.002 0.0001"/>
          
          <body name="weight" pos="0.125 0 0">
            <geom type="sphere" size="0.04" rgba="0.8 0.1 0.1 1"/>
            <inertial pos="0 0 0" mass="2.0" diaginertia="0.001 0.001 0.001"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  
  <actuator>
    <motor name="motor1" joint="joint1" gear="1" ctrllimited="true" ctrlrange="-100 100"/>
  </actuator>
</mujoco>
"""
    return mujoco.MjModel.from_xml_string(xml)


def chirp_signal(t, f_start, f_end, duration, amplitude):
    """Chirp扫频信号"""
    k = (f_end - f_start) / duration
    phase = 2 * np.pi * (f_start * t + 0.5 * k * t**2)
    return amplitude * np.sin(phase)


class LivePlotter:
    """实时绘图器"""
    
    def __init__(self, window_size=1000):
        self.window_size = window_size
        
        # 数据缓冲
        self.data = {
            'time': deque(maxlen=window_size),
            'pos_des': deque(maxlen=window_size),
            'pos_act': deque(maxlen=window_size),
            'vel_des': deque(maxlen=window_size),
            'vel_act': deque(maxlen=window_size),
            'torque': deque(maxlen=window_size)
        }
        
        # 创建图形
        plt.ion()
        self.fig, self.axes = plt.subplots(3, 1, figsize=(14, 10))
        self.fig.canvas.manager.set_window_title('实时跟踪曲线')
        
        # 位置跟踪
        self.ax1 = self.axes[0]
        self.line_pos_des, = self.ax1.plot([], [], 'r--', label='期望位置', linewidth=2.5, alpha=0.8)
        self.line_pos_act, = self.ax1.plot([], [], 'b-', label='实际位置', linewidth=1.8)
        self.ax1.set_ylabel('位置 (rad)', fontsize=12, fontweight='bold')
        self.ax1.legend(loc='upper right', fontsize=10)
        self.ax1.grid(True, alpha=0.4, linestyle='--')
        self.ax1.set_title('位置跟踪', fontsize=13, fontweight='bold', pad=10)
        
        # 速度跟踪
        self.ax2 = self.axes[1]
        self.line_vel_des, = self.ax2.plot([], [], 'r--', label='期望速度', linewidth=2.5, alpha=0.8)
        self.line_vel_act, = self.ax2.plot([], [], 'g-', label='实际速度', linewidth=1.8)
        self.ax2.set_ylabel('速度 (rad/s)', fontsize=12, fontweight='bold')
        self.ax2.legend(loc='upper right', fontsize=10)
        self.ax2.grid(True, alpha=0.4, linestyle='--')
        self.ax2.set_title('速度跟踪', fontsize=13, fontweight='bold', pad=10)
        
        # 控制力矩
        self.ax3 = self.axes[2]
        self.line_torque, = self.ax3.plot([], [], 'm-', label='控制力矩', linewidth=1.8)
        self.ax3.axhline(y=10, color='r', linestyle=':', alpha=0.5, label='力矩上限')
        self.ax3.axhline(y=-10, color='r', linestyle=':', alpha=0.5)
        self.ax3.set_xlabel('时间 (s)', fontsize=12, fontweight='bold')
        self.ax3.set_ylabel('力矩 (Nm)', fontsize=12, fontweight='bold')
        self.ax3.legend(loc='upper right', fontsize=10)
        self.ax3.grid(True, alpha=0.4, linestyle='--')
        self.ax3.set_title('控制力矩', fontsize=13, fontweight='bold', pad=10)
        
        plt.tight_layout()
        plt.show(block=False)
        
        self.running = True
    
    def update(self, t, pos_des, pos_act, vel_des, vel_act, torque):
        """更新数据"""
        self.data['time'].append(t)
        self.data['pos_des'].append(pos_des)
        self.data['pos_act'].append(pos_act)
        self.data['vel_des'].append(vel_des)
        self.data['vel_act'].append(vel_act)
        self.data['torque'].append(torque)
    
    def refresh(self):
        """刷新显示"""
        if len(self.data['time']) < 2:
            return
        
        t_array = np.array(self.data['time'])
        
        # 更新位置
        self.line_pos_des.set_data(t_array, self.data['pos_des'])
        self.line_pos_act.set_data(t_array, self.data['pos_act'])
        self.ax1.relim()
        self.ax1.autoscale_view()
        
        # 更新速度
        self.line_vel_des.set_data(t_array, self.data['vel_des'])
        self.line_vel_act.set_data(t_array, self.data['vel_act'])
        self.ax2.relim()
        self.ax2.autoscale_view()
        
        # 更新力矩
        self.line_torque.set_data(t_array, self.data['torque'])
        self.ax3.relim()
        self.ax3.autoscale_view()
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def close(self):
        """关闭"""
        self.running = False
        plt.close(self.fig)


def main():
    """主函数"""
    print("="*70)
    print("快速测试 - 实时波形显示")
    print("="*70)
    print("配置: 摆杆0.25m + 末端重量块2kg")
    print("信号: Chirp 0.1~5Hz, 幅值±10°")
    print("控制: PD (Kp=50, Kd=5)")
    print("持续: 15秒")
    print("="*70)
    print("\n启动MuJoCo仿真和matplotlib实时绘图...")
    print("提示: 关闭任意窗口可停止仿真\n")
    
    # 创建模型
    model = create_simple_model()
    data = mujoco.MjData(model)
    
    # 创建绘图器
    plotter = LivePlotter(window_size=1500)
    
    # 控制参数
    kp = 50.0
    kd = 5.0
    test_duration = 15.0
    
    # 记录统计数据
    pos_errors = []
    vel_errors = []
    torques = []
    
    # 运行仿真
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 45
        viewer.cam.elevation = -20
        viewer.cam.distance = 1.5
        viewer.cam.lookat = np.array([0.0, 0.0, 0.2])
        
        update_counter = 0
        prev_pos_des = 0.0
        
        while viewer.is_running() and data.time < test_duration and plotter.running:
            step_start = time.time()
            
            t = data.time
            
            # 期望位置
            pos_des = chirp_signal(t, 0.1, 5.0, test_duration, np.deg2rad(10))
            
            # 期望速度（数值微分）
            vel_des = (pos_des - prev_pos_des) / model.opt.timestep
            prev_pos_des = pos_des
            
            # PD控制
            pos_err = pos_des - data.qpos[0]
            vel_err = vel_des - data.qvel[0]
            torque = kp * pos_err + kd * vel_err
            torque = np.clip(torque, -100, 100)
            
            data.ctrl[0] = torque
            
            # 记录统计
            pos_errors.append(pos_err)
            vel_errors.append(vel_err)
            torques.append(torque)
            
            # 更新绘图
            plotter.update(t, pos_des, data.qpos[0], vel_des, data.qvel[0], torque)
            
            update_counter += 1
            if update_counter >= 30:  # 每30步刷新一次（减少绘图频率）
                plotter.refresh()
                update_counter = 0
            
            # 步进仿真
            mujoco.mj_step(model, data)
            viewer.sync()
            
            # 实时控制
            time_until_next = model.opt.timestep - (time.time() - step_start)
            if time_until_next > 0:
                time.sleep(time_until_next)
    
    # 最后刷新
    plotter.refresh()
    
    # 统计结果
    pos_errors = np.array(pos_errors)
    vel_errors = np.array(vel_errors)
    torques = np.array(torques)
    
    print("\n" + "="*70)
    print("测试完成 - 统计结果")
    print("="*70)
    print(f"位置跟踪:")
    print(f"  RMS误差: {np.sqrt(np.mean(pos_errors**2)):.6f} rad = {np.rad2deg(np.sqrt(np.mean(pos_errors**2))):.3f}°")
    print(f"  最大误差: {np.max(np.abs(pos_errors)):.6f} rad = {np.rad2deg(np.max(np.abs(pos_errors))):.3f}°")
    print(f"\n速度跟踪:")
    print(f"  RMS误差: {np.sqrt(np.mean(vel_errors**2)):.6f} rad/s")
    print(f"  最大速度: {np.max(np.abs(plotter.data['vel_act'])):.3f} rad/s")
    print(f"\n控制力矩:")
    print(f"  最大力矩: {np.max(np.abs(torques)):.3f} Nm")
    print(f"  平均力矩: {np.mean(np.abs(torques)):.3f} Nm")
    print(f"  RMS力矩: {np.sqrt(np.mean(torques**2)):.3f} Nm")
    print("="*70)
    
    input("\n按Enter关闭波形图...")
    plotter.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n仿真已中断")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()

