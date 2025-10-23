# 电机仿真测试 - 配置说明

## 配置文件

所有参数现在都通过 `config.yaml` 文件进行配置，包括：

### 控制参数
- `kp`: 位置增益
- `kd`: 速度增益  
- `torque_limit`: 力矩限制

### 测试信号参数
- `f_start`: 起始频率
- `f_end`: 结束频率
- `amplitude`: 幅值（度）
- `duration`: 测试时长

### 材料参数
- `arm_radius`: 摆杆半径
- `density_aluminum`: 铝密度

### 测试配置
- 可以添加/修改不同的测试配置
- 每个配置包含：名称、摆杆长度、重量块质量、重量块位置

## 使用方法

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 修改 `config.yaml` 中的参数

3. 运行测试：
```bash
python script/2_simulate_configurations.py
```

## 主要改进

- ✅ 所有硬编码参数移至YAML配置
- ✅ PD控制参数实时显示
- ✅ 支持动态配置测试参数
- ✅ 代码结构更清晰，易于维护
