from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import numpy as np
import omni
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, DynamicCylinder
from isaacsim.sensors.physx import _range_sensor

# World 및 타임라인 설정
my_world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()
timeline = omni.timeline.get_timeline_interface()

lidarInterface = _range_sensor.acquire_lidar_sensor_interface()

# LiDAR가 달릴 obj1
obj1 = my_world.scene.add(
    DynamicCylinder(
        prim_path="/World/new_obj_1",
        name="obj_1",
        position=np.array([0.0, 0.0, 0.3]),
        scale=np.array([0.1, 0.1, 0.1]),
    )
)

# 앞에 박스(obj2)
obj2 = my_world.scene.add(
    DynamicCuboid(
        prim_path="/World/new_obj_2",
        name="obj_2",
        position=np.array([2.0, 0.0, 0.1]),
        scale=np.array([0.5, 0.5, 0.5]),
    )
)

# Physics Scene 추가
omni.kit.commands.execute(
    "AddPhysicsSceneCommand",
    stage=stage,
    path="/World/PhysicsScene"
)

# LiDAR 생성
lidarPath = "/LidarName"
result, prim = omni.kit.commands.execute(
    "RangeSensorCreateLidar",
    path=lidarPath,
    parent="/World/new_obj_1",
    min_range=0.5,
    max_range=1000.0,
    draw_points=False,
    draw_lines=True,
    horizontal_fov=1.0,
    vertical_fov=1.0,
    horizontal_resolution=0.4,
    vertical_resolution=0.4,
    rotation_rate=0.0,
    high_lod=True,
    yaw_offset=0.0,
    enable_semantics=False,
)

# 실제 LiDAR prim path (엔진이 자동으로 바꿨을 수도 있어서 안전하게 가져오기)
lidar_prim_path = prim.GetPath().pathString
print("LiDAR prim path:", lidar_prim_path)

lidarInterface = _range_sensor.acquire_lidar_sensor_interface()

my_world.scene.add_default_ground_plane()

timeline.play()
my_world.reset()

i = 1

while simulation_app.is_running():
    my_world.step(render=True)

    if i % 100 == 0:

        if not lidarInterface.is_lidar_sensor(lidar_prim_path):
            print("이 경로에는 LiDAR 센서가 없습니다:", lidar_prim_path)
        else:
            depth_array = lidarInterface.get_linear_depth_data(lidar_prim_path)

            if depth_array is None or len(depth_array) == 0:
                print(f"[{i}] LiDAR depth 데이터가 없습니다.")
            else:
                depth_array = np.array(depth_array, dtype=np.float32)
                valid = depth_array[np.isfinite(depth_array) & (depth_array > 0)]

                if valid.size > 0:
                    min_depth = float(valid.min())
                    print(f"[{i}] depth: {min_depth:.4f} m")
                else:
                    print(f"[{i}] 유효한 depth가 없습니다.")

    i += 1

simulation_app.close()
