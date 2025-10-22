import numpy as np
import pandas as pd
import mujoco
import matplotlib.pyplot as plt

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

"""
转动惯量范围计算脚本
目的：给出不同摆杆配置下的转动惯量范围，供结构设计参考
"""

# 物理参数
DENSITY_STEEL = 7850  # kg/m³ 钢材密度
DENSITY_ALUMINUM = 2700  # kg/m³ 铝材密度
ARM_RADIUS = 0.015  # m, 摆杆半径(直径30mm)

def calculate_rod_inertia(length, mass, distance_to_axis):
    """
    计算单侧杆件绕旋转轴的转动惯量
    
    Parameters:
    - length: 摆杆长度 (m)
    - mass: 摆杆质量 (kg)
    - distance_to_axis: 质心到旋转轴距离 (m)，单侧杆为 length/2
    
    Returns:
    - inertia: 转动惯量 (kg·m²)
    """
    # 细杆绕端点的转动惯量: I = (1/3) * m * L^2
    # 或使用平行轴定理: I = I_com + m * d^2
    # 其中 I_com = (1/12) * m * L^2 (绕质心)
    I_com = (1/12) * mass * length**2
    I_total = I_com + mass * distance_to_axis**2
    return I_total

def calculate_point_mass_inertia(mass, distance):
    """计算质点绕轴的转动惯量"""
    return mass * distance**2

def scan_configurations():
    """扫描不同配置，计算转动惯量范围"""
    
    results = []
    
    # 摆杆长度范围 (m): 0.1m ~ 0.3m
    arm_lengths = np.linspace(0.1, 0.3, 5)
    
    # 重量块质量 (kg): 0 ~ 5kg (根据100Nm峰值扭矩调整)
    weight_masses = [0, 1.0, 2.0, 3.0, 4.0, 5.0]
    
    # link1固有参数
    link1_mass = 0.5  # kg
    link1_inertia_zz = 0.0001  # kg·m²（绕z轴）
    
    print("=" * 80)
    print("转动惯量计算结果")
    print("=" * 80)
    print(f"{'摆杆长度(m)':<12} {'摆杆质量(kg)':<12} {'重量块(kg)':<12} "
          f"{'重量块位置(m)':<14} {'总转动惯量(kg·m²)':<18} {'负载等级':<10}")
    print("-" * 80)
    
    for arm_length in arm_lengths:
        # 摆杆质量 = 体积 × 密度
        arm_volume = np.pi * ARM_RADIUS**2 * arm_length
        arm_mass_steel = arm_volume * DENSITY_STEEL
        arm_mass_aluminum = arm_volume * DENSITY_ALUMINUM
        
        # 选择铝材（较轻）
        arm_mass = arm_mass_aluminum
        
        # 摆杆质心到轴距离（单侧）
        arm_center_distance = arm_length / 2
        
        # 摆杆转动惯量
        arm_inertia = calculate_rod_inertia(arm_length, arm_mass, arm_center_distance)
        
        for weight_mass in weight_masses:
            # 重量块放在摆杆末端
            weight_distance = arm_length
            weight_inertia = calculate_point_mass_inertia(weight_mass, weight_distance)
            
            # 总转动惯量
            total_inertia = link1_inertia_zz + arm_inertia + weight_inertia
            
            # 负载分级（根据100Nm峰值扭矩重新定义）
            if weight_mass == 0:
                load_level = "空载"
            elif weight_mass <= 2.0:
                load_level = "中负载"
            else:
                load_level = "大负载"
            
            results.append({
                '摆杆长度_m': arm_length,
                '摆杆质量_kg': arm_mass,
                '重量块质量_kg': weight_mass,
                '重量块位置_m': weight_distance,
                '总转动惯量_kg_m2': total_inertia,
                '负载等级': load_level
            })
            
            print(f"{arm_length:<12.3f} {arm_mass:<12.3f} {weight_mass:<12.1f} "
                  f"{weight_distance:<14.3f} {total_inertia:<18.6f} {load_level:<10}")
    
    return results

def generate_summary(results):
    """生成汇总表格"""
    df = pd.DataFrame(results)
    
    print("\n" + "=" * 80)
    print("转动惯量范围汇总（供结构设计参考）")
    print("=" * 80)
    
    # 按负载等级分组统计
    for load_level in ["空载", "中负载", "大负载"]:
        subset = df[df['负载等级'] == load_level]
        if len(subset) > 0:
            min_inertia = subset['总转动惯量_kg_m2'].min()
            max_inertia = subset['总转动惯量_kg_m2'].max()
            mean_inertia = subset['总转动惯量_kg_m2'].mean()
            print(f"\n{load_level}:")
            print(f"  转动惯量范围: {min_inertia:.6f} ~ {max_inertia:.6f} kg·m²")
            print(f"  平均值: {mean_inertia:.6f} kg·m²")
    
    # 整体范围
    print(f"\n整体范围:")
    print(f"  最小转动惯量: {df['总转动惯量_kg_m2'].min():.6f} kg·m²")
    print(f"  最大转动惯量: {df['总转动惯量_kg_m2'].max():.6f} kg·m²")
    
    # 推荐配置
    print("\n" + "=" * 80)
    print("推荐配置方案")
    print("=" * 80)
    print("\n方案1 - 空载测试:")
    empty = df[df['负载等级'] == "空载"].iloc[0]
    print(f"  摆杆长度: {empty['摆杆长度_m']:.3f} m")
    print(f"  重量块: 无")
    print(f"  转动惯量: {empty['总转动惯量_kg_m2']:.6f} kg·m²")
    
    print("\n方案2 - 中负载测试:")
    mid = df[(df['负载等级'] == "中负载") & (df['重量块质量_kg'] == 1.0)].iloc[len(df)//2 % sum((df['负载等级'] == "中负载") & (df['重量块质量_kg'] == 1.0))]
    print(f"  摆杆长度: {mid['摆杆长度_m']:.3f} m")
    print(f"  重量块: {mid['重量块质量_kg']:.1f} kg (末端)")
    print(f"  转动惯量: {mid['总转动惯量_kg_m2']:.6f} kg·m²")
    
    print("\n方案3 - 大负载测试:")
    heavy = df[df['负载等级'] == "大负载"].iloc[-1]
    print(f"  摆杆长度: {heavy['摆杆长度_m']:.3f} m")
    print(f"  重量块: {heavy['重量块质量_kg']:.1f} kg (末端)")
    print(f"  转动惯量: {heavy['总转动惯量_kg_m2']:.6f} kg·m²")
    
    # 保存到CSV
    output_file = 'inertia_range_results.csv'
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n详细数据已保存至: {output_file}")

if __name__ == "__main__":
    results = scan_configurations()
    generate_summary(results)
    
    print("\n" + "=" * 80)
    print("说明:")
    print("- 摆杆材料假设为铝材(密度2700 kg/m³)")
    print("- 重量块放置在摆杆末端以最大化转动惯量")
    print("- 实际工装设计时需考虑电机扭矩限制和结构强度")
    print("- 建议先用仿真验证电机能否驱动该惯量负载")
    print("=" * 80)

