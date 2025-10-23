#!/usr/bin/env python3
import os
import yaml

def test_config_path():
    """测试配置文件路径"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.yaml")
    
    print(f"脚本目录: {script_dir}")
    print(f"配置文件路径: {config_path}")
    print(f"配置文件存在: {os.path.exists(config_path)}")
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print("配置文件加载成功！")
            print(f"控制参数: Kp={config['control']['kp']}, Kd={config['control']['kd']}")
        except Exception as e:
            print(f"配置文件加载失败: {e}")
    else:
        print("配置文件不存在！")

if __name__ == "__main__":
    test_config_path()
