from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import numpy as np
import omni
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, DynamicCylinder
from isaacsim.sensors.physx import _range_sensor
from isaacsim.sensors.camera import Camera
import isaacsim.core.utils.numpy.rotations as rot_utils

# World 및 타임라인 설정
my_world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()
timeline = omni.timeline.get_timeline_interface()

lidarInterface = _range_sensor.acquire_lidar_sensor_interface()

cam_body = my_world.scene.add(
    DynamicCylinder(
        prim_path="/World/cam_body",
        name="cam_body",
        position=np.array([0.0, 0.0, 0.3]),
        scale=np.array([0.1, 0.1, 0.1]),
    )
)

# 앞에 박스(obj2)
obj1 = my_world.scene.add(
    DynamicCuboid(
        prim_path="/World/new_obj_1",
        name="obj_1",
        position=np.array([2.0, 0.0, 0.1]),
        scale=np.array([0.5, 0.5, 0.5]),
        color=np.array([1.0, 0.0, 0.0])
    )
)

camera = Camera(
    prim_path="/World/cam_body/camera",
    frequency=30,
    resolution=(100, 100),
    orientation=rot_utils.euler_angles_to_quats(np.array([0, 0, 180]), degrees=True),
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
    parent="/World/cam_body",
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
camera.initialize()
camera.add_motion_vectors_to_frame()
reset_needed = False

i = 1

while simulation_app.is_running():
    my_world.step(render=True)

    if i % 100 == 0:
        cam_pixel = camera.get_rgb()
        cam_rgb = cam_pixel[49:52,49:52]
        total_rgb = cam_rgb.sum(axis=1)
        _total_rgb = total_rgb.sum(axis=0)

        if _total_rgb[0] > _total_rgb[1] and _total_rgb[0] > _total_rgb[2]:
            cam_color = '빨간색'
        elif _total_rgb[1] > _total_rgb[0] and _total_rgb[1] > _total_rgb[2]:
            cam_color = '초록색'
        elif _total_rgb[2] > _total_rgb[1] and _total_rgb[2] > _total_rgb[0]:
            cam_color = '파란색'
        
        print(f"cube의 색상은 {cam_color}")
        depth_array = lidarInterface.get_linear_depth_data(lidar_prim_path)

        depth_array = np.array(depth_array, dtype=np.float32)
        valid = depth_array[np.isfinite(depth_array) & (depth_array > 0)]

        if valid.size > 0:
            min_depth = float(valid.min())
            print(f"[{i}] depth: {min_depth:.4f} m")
        else:
            print(f"[{i}] 유효한 depth가 없습니다.")

    i += 1

simulation_app.close()
