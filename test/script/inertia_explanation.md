# 转动惯量概念详解

## 两种转动惯量的区别

### 1. **系统总转动惯量** (我们计算的)
```
I_total = I_link1 + I_arm + I_weight (绕电机轴z的转动惯量)
```

**定义**：整个系统绕**电机旋转轴**（z轴）的转动惯量  
**用途**：
- 估算电机所需扭矩：`τ = I_total × α` (角加速度)
- 动力学仿真性能预测
- 给结构设计人员提供负载参考

**计算方法**：
- 使用平行轴定理：`I = I_com + m × d²`
- `I_com`：刚体绕自身质心的转动惯量
- `m × d²`：由于质心偏离旋转轴产生的附加惯量

**例子**：
```python
# 摆杆（单侧杆）
arm_length = 0.3  # m
arm_mass = 0.3    # kg
arm_center = arm_length / 2  # 质心距轴0.15m

I_com = (1/12) * arm_mass * arm_length**2  # 绕质心
I_total = I_com + arm_mass * arm_center**2  # 绕旋转轴
```

---

### 2. **Body inertial 参数** (MJCF中设置的)

**定义**：每个刚体在**其自身质心坐标系**下的转动惯量张量  
**格式**：`diaginertia="Ixx Iyy Izz"` (主转动惯量对角元素)

**用途**：
- MuJoCo物理引擎需要的刚体属性
- 计算每个body自身的转动惯性

**坐标系**：
- 以该body的质心为原点
- 沿body的主惯性轴方向

---

## 具体示例对比

### 水平摆杆 (圆柱体)

**物理参数**：
- 长度 L = 0.3 m
- 半径 r = 0.015 m  
- 质量 m = 0.3 kg
- 质心位置：距离旋转轴 d = 0.15 m

#### 方法1：系统总转动惯量（绕电机轴z）
```python
# 用于估算电机负载
I_com_zz = (1/12) * m * L**2          # 绕质心的垂直轴
I_total_z = I_com_zz + m * d**2       # 平行轴定理
I_total_z ≈ 0.0023 + 0.0068 = 0.0091 kg·m²
```

#### 方法2：MJCF body inertial（绕自身质心）
```xml
<body name="arm" pos="0.15 0 0.04">
  <geom type="cylinder" size="0.015 0.15" quat="0.707 0 0.707 0"/>
  <inertial pos="0 0 0" mass="0.3" 
            diaginertia="0.0023 0.0023 0.000034"/>
  <!--
    Ixx = Iyy = (1/12)*m*L² + (1/4)*m*r² = 0.0023 kg·m²
    Izz = (1/2)*m*r² = 0.000034 kg·m²
  -->
</body>
```

---

## 哪种正确？

**答案：两种都正确，但用途不同！**

| 用途 | 使用哪种 | 说明 |
|------|---------|------|
| 给结构设计负载范围 | **系统总转动惯量** | 告诉他们电机要驱动多大的惯量 |
| 设置MJCF模型 | **Body inertial** | 每个body在其质心系的转动惯量 |
| 估算所需扭矩 | **系统总转动惯量** | τ = I_total × α |
| 物理仿真计算 | **Body inertial** | MuJoCo自动计算系统动力学 |

---

## 正确的工作流程

### 步骤1：计算系统总转动惯量（给结构）
```python
# script/1_inertia_range.py 计算的就是这个
I_total = I_link1_zz + I_arm_zz + I_weight_zz
print(f"系统转动惯量范围: {I_min} ~ {I_max} kg·m²")
```
**输出给结构**：转动惯量范围供工装设计参考

### 步骤2：设置MJCF的inertial（建模）
```xml
<!-- 每个body设置其在质心系的转动惯量 -->
<body name="arm">
  <inertial pos="0 0 0" mass="m" diaginertia="Ixx Iyy Izz"/>
</body>
```

### 步骤3：MuJoCo仿真验证
- MuJoCo根据各body的inertial自动计算系统动力学
- 验证电机扭矩是否足够

---

## 实际例子

### 配置：摆杆0.3m + 重量块4kg

#### 给结构的信息（系统总转动惯量）：
```python
arm_length = 0.3
arm_mass = 0.27      # 铝材圆柱
weight_mass = 4.0
weight_pos = 0.3

# 绕电机轴z的转动惯量
I_arm = (1/12)*0.27*0.3² + 0.27*(0.15)²     = 0.0084 kg·m²
I_weight = 4.0 * 0.3²                        = 0.36 kg·m²
I_total = 0.0001 + 0.0084 + 0.36             = 0.3685 kg·m²

所需扭矩 ≈ 0.3685 × 角加速度
```
**给结构**：系统转动惯量约 **0.37 kg·m²**

#### MJCF模型设置（body inertial）：
```xml
<!-- 摆杆arm：水平圆柱，质心在自身中点 -->
<body name="arm" pos="0.15 0 0.04">
  <inertial pos="0 0 0" mass="0.27" 
            diaginertia="0.0023 0.0023 0.000034"/>
  <!-- Ixx, Iyy: 绕arm自身质心的垂直轴 -->
  <!-- Izz: 绕arm自身轴线 -->
  
  <!-- 重量块：球体 -->
  <body name="weight" pos="0.15 0 0">
    <inertial pos="0 0 0" mass="4.0" 
              diaginertia="0.0026 0.0026 0.0026"/>
    <!-- 球体: I = (2/5)*m*r² -->
  </body>
</body>
```

---

## 总结

1. **脚本计算的"总转动惯量"** = 系统绕电机轴的转动惯量
   - ✅ **正确用于**：给结构设计负载参考
   - ✅ **正确用于**：估算所需扭矩
   
2. **MJCF中的diaginertia** = 每个body在其质心系的转动惯量
   - ✅ **正确用于**：MuJoCo模型建立
   - ✅ **正确用于**：物理仿真计算

3. **两者关系**：
   - MuJoCo根据各body的inertial，通过平行轴定理自动计算系统总转动惯量
   - 仿真结果应该与我们手算的总转动惯量一致

---

## 验证方法

可以在MuJoCo中验证计算是否正确：

```python
import mujoco

model = mujoco.MjModel.from_xml_path('motor.xml')
data = mujoco.MjData(model)

# 设置角加速度
data.qacc[0] = 1.0  # 1 rad/s²

# 步进一次
mujoco.mj_forward(model, data)

# 所需扭矩 = 系统转动惯量 × 角加速度
torque_needed = data.qfrc_bias[0]  # 考虑重力等
print(f"所需扭矩: {torque_needed} Nm")
print(f"系统转动惯量约: {torque_needed / 1.0} kg·m²")
```

