import mujoco
import mujoco.viewer
import numpy as np
import time

# 加载模型
model = mujoco.MjModel.from_xml_path('../mjcf/motor.xml')
data = mujoco.MjData(model)

print("=" * 50)
print("单侧摆臂模型信息")
print("=" * 50)
print(f"关节数量: {model.njnt}")
print(f"关节名称: {[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]}")
print(f"执行器数量: {model.nu}")
print(f"刚体数量: {model.nbody}")
print(f"刚体名称: {[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)]}")
print("=" * 50)

# 使用被动查看器
def run_simulation():
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 设置相机视角
        viewer.cam.azimuth = 45
        viewer.cam.elevation = -20
        viewer.cam.distance = 1.0
        viewer.cam.lookat = np.array([0.0, 0.0, 0.2])
        
        start_time = time.time()
        
        while viewer.is_running():
            step_start = time.time()
            
            # 施加正弦控制信号让link1旋转
            current_time = data.time
            data.ctrl[0] = 2.0 * np.sin(2 * np.pi * 0.5 * current_time)
            
            # 步进仿真
            mujoco.mj_step(model, data)
            
            # 同步查看器
            viewer.sync()
            
            # 控制仿真速度
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    print("\n启动仿真...")
    print("link1 将以正弦模式旋转")
    print("按 Ctrl+C 或关闭窗口退出\n")
    
    try:
        run_simulation()
    except KeyboardInterrupt:
        print("\n仿真已停止")

