#!/usr/bin/env python3
"""
字体检查脚本
检查系统可用的中文字体，解决matplotlib中文显示问题
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

def check_fonts():
    """检查系统可用字体"""
    print("="*60)
    print("系统字体检查")
    print("="*60)
    
    # 获取所有字体
    fonts = [f.name for f in fm.fontManager.ttflist]
    unique_fonts = sorted(set(fonts))
    
    print(f"系统共有 {len(unique_fonts)} 种字体")
    print("\n包含中文字符的字体:")
    
    chinese_fonts = []
    for font in unique_fonts:
        try:
            # 检查字体是否支持中文
            font_path = fm.findfont(fm.FontProperties(family=font))
            font_obj = fm.FontProperties(fname=font_path)
            # 简单测试：如果字体名包含中文相关关键词
            if any(keyword in font.lower() for keyword in ['chinese', 'cjk', 'han', 'sim', 'hei', 'kai', 'song']):
                chinese_fonts.append(font)
                print(f"  ✓ {font}")
        except:
            continue
    
    if not chinese_fonts:
        print("  未找到明确的中文字体")
        print("\n建议安装中文字体:")
        print("  sudo apt-get install fonts-wqy-zenhei fonts-wqy-microhei")
        print("  或")
        print("  sudo apt-get install fonts-noto-cjk")
    
    return chinese_fonts

def test_chinese_display():
    """测试中文显示效果"""
    print("\n" + "="*60)
    print("中文显示测试")
    print("="*60)
    
    # 设置字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建测试图形
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.linspace(0, 2*np.pi, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    
    ax.plot(x, y1, 'r-', label='正弦波', linewidth=2)
    ax.plot(x, y2, 'b--', label='余弦波', linewidth=2)
    
    ax.set_xlabel('时间 (s)', fontsize=12)
    ax.set_ylabel('幅值', fontsize=12)
    ax.set_title('中文字体显示测试', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 添加中文文本
    ax.text(1.5, 0.5, '位置跟踪曲线', fontsize=12, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    plt.tight_layout()
    
    # 保存图片
    output_file = 'font_test.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"测试图片已保存: {output_file}")
    
    # 显示图片
    plt.show()
    
    return True

def fix_font_issue():
    """修复字体问题"""
    print("\n" + "="*60)
    print("字体问题修复建议")
    print("="*60)
    
    print("1. 安装中文字体包:")
    print("   sudo apt-get update")
    print("   sudo apt-get install fonts-wqy-zenhei fonts-wqy-microhei")
    print("   sudo apt-get install fonts-noto-cjk")
    
    print("\n2. 清除matplotlib字体缓存:")
    print("   rm -rf ~/.cache/matplotlib")
    print("   python -c \"import matplotlib.font_manager; matplotlib.font_manager._rebuild()\"")
    
    print("\n3. 在代码中设置字体:")
    print("   import matplotlib.pyplot as plt")
    print("   plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']")
    print("   plt.rcParams['axes.unicode_minus'] = False")
    
    print("\n4. 如果仍有问题，使用英文标签:")
    print("   - 将中文标签改为英文")
    print("   - 或使用matplotlib的英文默认字体")

if __name__ == "__main__":
    try:
        # 检查字体
        chinese_fonts = check_fonts()
        
        # 测试显示
        test_chinese_display()
        
        if not chinese_fonts:
            fix_font_issue()
        else:
            print(f"\n✓ 找到 {len(chinese_fonts)} 个中文字体，应该可以正常显示中文")
            
    except Exception as e:
        print(f"\n错误: {e}")
        fix_font_issue()
