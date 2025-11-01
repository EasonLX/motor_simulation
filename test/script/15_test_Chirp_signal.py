#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'Heiti TC', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# --------------------------
# Chirp信号核心参数（确保采样合规）
# --------------------------
f_start = 1.0          # 起始频率 (Hz)
f_end = 1250.0         # 结束频率 (Hz)
duration = 25.0        # 持续时间 (s)
amplitude = 10.0       # 振幅 (A)
fs = 1000.0            # 采样频率 (Hz)，满足 fs > 2*f_end（3000 > 2500）
dt = 1.0 / fs          # 时间步长 (s)
num_points = int(duration * fs)  # 总采样点数

# --------------------------
# 生成Chirp信号
# --------------------------
time = np.linspace(0, duration, num_points, endpoint=False)  # 时间序列
k = (f_end - f_start) / duration  # 频率变化率 (Hz/s)
phase = 2 * np.pi * (f_start * time + 0.5 * k * time**2)  # 线性扫频相位
signal = amplitude * np.sin(phase)
inst_freq = f_start + k * time  # 瞬时频率（理论值）

# --------------------------
# 频谱分析（验证1kHz附近信号）
# --------------------------
freq_target = 1000.0
window_width = 1.0  # 分析窗口宽度 (s)，避免与时间序列变量冲突
time_center = (freq_target - f_start) / k  # 1kHz对应的理论时间点（标量）

# 截取1kHz附近的信号片段
mask_fft = (time >= time_center - window_width/2) & (time <= time_center + window_width/2)
signal_window = signal[mask_fft]
time_in_window = time[mask_fft]  # 窗口内的时间序列（数组）

# FFT计算
n = len(signal_window)
yf = fft(signal_window)
xf = fftfreq(n, dt)[:n//2]
yf_abs = 2.0/n * np.abs(yf[:n//2])  # 幅度谱（归一化）

# --------------------------
# 可视化展示
# --------------------------
fig = plt.figure(figsize=(15, 12))
gs = plt.GridSpec(4, 1, hspace=0.4)

# 1. 完整信号时域图
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(time, signal, 'b-', linewidth=0.5)
ax1.axvline(x=time_center, color='r', linestyle='--', 
            label=f'1kHz对应时间点 ({time_center:.2f}s)')
ax1.set_xlabel('时间 (s)')
ax1.set_ylabel('电流 (A)')
ax1.set_title(f'Chirp信号（1-1250Hz，采样率{fs}Hz）')
ax1.grid(True, alpha=0.3)
ax1.legend()

# 2. 1kHz附近时域放大图（修复格式化错误）
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(time_in_window, signal_window, 'r-', linewidth=1)
ax2.set_xlabel('时间 (s)')
ax2.set_ylabel('电流 (A)')
# 使用标量计算时间范围，避免数组格式化
start_time = time_center - window_width/2
end_time = time_center + window_width/2
ax2.set_title(f'1kHz附近信号放大（{start_time:.2f}s至{end_time:.2f}s）')
ax2.grid(True, alpha=0.3)

# 3. 瞬时频率曲线
ax3 = fig.add_subplot(gs[2, 0])
ax3.plot(time, inst_freq, 'g-', linewidth=1)
ax3.axhline(y=1000, color='r', linestyle='--', label='1kHz')
ax3.axhline(y=fs/2, color='k', linestyle=':', label=f'奈奎斯特频率 ({fs/2}Hz)')
ax3.set_xlabel('时间 (s)')
ax3.set_ylabel('频率 (Hz)')
ax3.set_title('瞬时频率变化')
ax3.grid(True, alpha=0.3)
ax3.legend()

# 4. 1kHz附近频谱图
ax4 = fig.add_subplot(gs[3, 0])
ax4.plot(xf, yf_abs, 'purple', linewidth=1)
ax4.axvline(x=1000, color='r', linestyle='--', label='1kHz理论值')
ax4.set_xlabel('频率 (Hz)')
ax4.set_ylabel('幅度')
ax4.set_title(f'1kHz附近频谱（实际峰值：{xf[np.argmax(yf_abs)]:.1f}Hz）')
ax4.set_xlim(900, 1100)  # 聚焦1kHz附近
ax4.grid(True, alpha=0.3)
ax4.legend()

# 保存与显示
plt.savefig('chirp_complete_analysis.png', dpi=300, bbox_inches='tight')
print(f"图像已保存为 'chirp_complete_analysis.png'")
print(f"关键参数验证：")
print(f"  最高瞬时频率：{inst_freq[-1]:.1f}Hz")
print(f"  奈奎斯特频率：{fs/2:.1f}Hz（采样率{fs}Hz）")
print(f"  1kHz附近频谱峰值：{xf[np.argmax(yf_abs)]:.1f}Hz")

plt.show()