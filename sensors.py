from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import numpy as np
import omni
from isaacsim.core.api import World
from isaacsim.sensors.physx import _range_sensor
from isaacsim.sensors.camera import Camera
from isaacsim.core.api.objects import DynamicCuboid, DynamicCylinder
import isaacsim.core.utils.numpy.rotations as rot_utils


class SensorSystem:
    def __init__(self, cam_body_pos=[0.0, 0.0, 0.3]):
        # Simulation 및 월드 초기화
        self.sim = simulation_app
        self.world = World(stage_units_in_meters=1.0)

        self.stage = omni.usd.get_context().get_stage()
        self.timeline = omni.timeline.get_timeline_interface()

        self.lidar_interface = _range_sensor.acquire_lidar_sensor_interface()

        # cam_body 위치 저장
        self.cam_body_pos = np.array(cam_body_pos, dtype=float)

        # 생성 함수 분리
        self._create_environment()
        self._create_cam_body()
        self._create_camera()
        self._create_lidar()

        self.timeline.play()
        self.world.reset()
        self.camera.initialize()
        self.camera.add_motion_vectors_to_frame()

        print("SensorSystem initialized successfully.")

    # -------------------------------------------------------------
    # cam_body 생성 (위치 파라미터 적용)
    # -------------------------------------------------------------
    def _create_cam_body(self):
        """카메라 몸체 생성 (사용자가 전달한 위치 적용)"""
        self.cam_body = self.world.scene.add(
            DynamicCylinder(
                prim_path="/World/cam_body",
                name="cam_body",
                position=self.cam_body_pos,   # ← 여기에 적용됨!
                scale=np.array([0.1, 0.1, 0.1]),
            )
        )

    # -------------------------------------------------------------
    # 기본 환경 생성 (바닥, 박스 등)
    # -------------------------------------------------------------
    def _create_environment(self):
        # 전방 빨간 박스
        self.obj1 = self.world.scene.add(
            DynamicCuboid(
                prim_path="/World/obj_1",
                name="obj_1",
                position=np.array([2.0, 0.0, 0.1]),
                scale=np.array([0.5, 0.5, 0.5]),
                color=np.array([1.0, 0.0, 0.0])
            )
        )

        # 물리 엔진 추가
        omni.kit.commands.execute(
            "AddPhysicsSceneCommand",
            stage=self.stage,
            path="/World/PhysicsScene",
        )

        # 바닥 생성
        self.world.scene.add_default_ground_plane()

    # -------------------------------------------------------------
    def _create_camera(self):
        """카메라 생성"""
        self.camera = Camera(
            prim_path="/World/cam_body/camera",
            frequency=30,
            resolution=(100, 100),
            orientation=rot_utils.euler_angles_to_quats(
                np.array([0, 0, 180]),
                degrees=True
            ),
        )

    # -------------------------------------------------------------
    def _create_lidar(self):
        """LiDAR 생성"""
        result, prim = omni.kit.commands.execute(
            "RangeSensorCreateLidar",
            path="/Lidar",
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

        self.lidar_path = prim.GetPath().pathString
        print("LiDAR prim path:", self.lidar_path)

    # -------------------------------------------------------------
    # Sensor Data
    # -------------------------------------------------------------
    def get_camera_color(self):
        """카메라 중앙 픽셀의 색상 반환"""
        cam_pixel = self.camera.get_rgb()
        cam_rgb = cam_pixel[49:52, 49:52]
        total_rgb = cam_rgb.sum(axis=1)
        _total_rgb = total_rgb.sum(axis=0)

        if _total_rgb[0] > _total_rgb[1] and _total_rgb[0] > _total_rgb[2]:
            return "빨간색"
        elif _total_rgb[1] > _total_rgb[0] and _total_rgb[1] > _total_rgb[2]:
            return "초록색"
        else:
            return "파란색"

    # -------------------------------------------------------------
    def get_lidar_depth(self):
        """LiDAR로부터 최소 깊이값 추출"""
        depth = self.lidar_interface.get_linear_depth_data(self.lidar_path)
        depth = np.array(depth, dtype=np.float32)
        valid = depth[np.isfinite(depth) & (depth > 0)]

        if valid.size > 0:
            return float(valid.min())
        return None

    # -------------------------------------------------------------
    # Main Loop
    # -------------------------------------------------------------
    def step(self):
        """한 스텝 실행"""
        self.world.step(render=True)

    # -------------------------------------------------------------
    def close(self):
        print("Closing simulation...")
        self.sim.close()

    # -------------------------------------------------------------
    # Example Loop
    # -------------------------------------------------------------
    def run_demo(self):
        """100 스텝마다 카메라 색상 + lidar depth를 출력하는 데모 루프"""
        i = 1
        while self.sim.is_running():
            self.step()

            if i % 100 == 0:
                color = self.get_camera_color()
                depth = self.get_lidar_depth()

                print(f"[{i}] color: {color}")

                if depth:
                    print(f"[{i}] depth: {depth:.4f} m")
                else:
                    print(f"[{i}] 유효한 depth 없음")

            i += 1


# -------------------------------------------------------------
# 실행
# -------------------------------------------------------------
if __name__ == "__main__":
    system = SensorSystem()
    system.run_demo()
    system.close()
