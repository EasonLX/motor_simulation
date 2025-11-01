# motor_simulation 电机单摆

1. 安装依赖：
```bash
pip install -r requirements.txt
```
2. 测试程序

```bash
config/
├── test_params.yaml                # Chirp 电机单摆仿真交付终版代码配置文件
└── config.yaml                     # 初步测试配置

md/                                 # 一些可参考md文档

mjcf/                               # 动态/默认生成xml

script/
├── 1_inertia_range.py              # 转动惯量计算
├── 2_simulate_configurations.py    # 仿真验证不同摆杆配置(load_test初版代码)
├── 3_quick_test.py                 # 快速测试脚本 - 单配置实时波形显示
├── 4_font_check.py                 # 字体检查
├── 6_load_inertia_test.py          # 专业测试模板
├── 7_verify_inertia.py             # 验证手算的系统总转动惯量与MuJoCo仿真的一致性
├── 8_mass_matrix_explanation.py    # 解释为什么可以直接从M(q)提取转动惯量
├── 9_visualize_mass_matrix.py      # 可视化质量矩阵M(q)的物理意义
├── 10_get_mujoco_inertia.py        # 获取MuJoCo计算的转动惯量 - 简单演示
├── 11_load_test0.py                # 摆杆负载测试 - 验证电机驱动能力(load_test初版代码)
├── 12_load_no_dia_test.py          # 不计算diaginertia，让MuJoCo自动计算(失败，必须传入)
├── 13_test_config.py               # config yaml配置文件导入测试
├── 14_test_inertia_variation.py    # 测试质量矩阵M(q)是否随关节位置变化
├── 15_test_Chirp_signal.py         # Chirp信号测试，确认奈奎斯特频率
├── 16_load_test.py                 # Chirp 电机单摆仿真交付终版代码
├── requirements.txt                # 安装依赖
└── readme.md                       # 本说明文档
```
3. test
```bash
python3 xxxx.py
```
