import numpy as np
import sys
import random
import omni
import omni.ext
import omni.usd #에러나면 삭제
import omni.ui as ui
import omni.timeline
import omni.kit.app
import carb
import os
from typing import Dict
from isaacsim.examples.interactive.base_sample import BaseSample
import omni.isaac.core.utils.prims as prim_utils
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path
from isaacsim.robot.manipulators.grippers import SurfaceGripper
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.core.api.objects import DynamicCuboid, DynamicCylinder
import isaacsim.core.utils.numpy.rotations as rot_utils
import isaacsim.robot_motion.motion_generation as mg
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.prims import SingleArticulation
from isaacsim.sensors.physx import _range_sensor
from isaacsim.sensors.camera import Camera
from isaacsim.core.api import World
from pxr import Gf, UsdGeom
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.wheeled_robots.robots import WheeledRobot

#데이터 저장용 로그경로
LOG_PATH = "/home/rokey/isaacsim/exts/my_custom_info_display/logs/rox_data.txt" 

class RMPFlowController(mg.MotionPolicyController):

    def __init__(
        self,
        name: str,
        robot_articulation: SingleArticulation,
        physics_dt: float = 1.0 / 60.0,
        attach_gripper: bool = False,
    ) -> None:

        if attach_gripper:
            self.rmp_flow_config = mg.interface_config_loader.load_supported_motion_policy_config(
                "UR10", "RMPflowSuction"
            )
        else:
            self.rmp_flow_config = mg.interface_config_loader.load_supported_motion_policy_config("UR10", "RMPflow")
        self.rmp_flow = mg.lula.motion_policies.RmpFlow(**self.rmp_flow_config)

        self.articulation_rmp = mg.ArticulationMotionPolicy(
            robot_articulation, self.rmp_flow, physics_dt
        )

        mg.MotionPolicyController.__init__(self, name=name, articulation_motion_policy=self.articulation_rmp)
        (
            self._default_position,
            self._default_orientation,
        ) = self._articulation_motion_policy._robot_articulation.get_world_pose()
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position,
            robot_orientation=self._default_orientation,
        )
        return

    def reset(self):
        mg.MotionPolicyController.reset(self)
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position,
            robot_orientation=self._default_orientation,
        )

class RoxProject(BaseSample):
    def __init__(self) -> None:
        super().__init__()

        self.colors = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        ]
        
        # 각 색이 몇 번 사용됐는지 카운트 (인덱스 기준)
        self._max_per_color = 2
        self._color_counts = {0: 0, 1: 0, 2: 0}

        # 랜덤 큐브 생성
        self._random_cube_spawn_position = np.array([-1.63, 0.0, 2])
        self._random_cube_position = self._random_cube_spawn_position.copy()
        self._cube_height = 0.05
        self._stack_level = 0  

        self._cube_spawn_interval = 3.5
        self._cube_spawn_timer = 0.0

        # 첫 번째 로봇 위치 (기존)
        self.robot_position = np.array([7.25, 0.0, 0.3])

        # 추가 세 로봇 위치
        self.robot_positions_extra = [
            np.array([10.5, -3.2, 0.0]),
            np.array([10.5, -1.0, 0.0]),
            np.array([10.5,  1.2, 0.0]),
        ]

        # 도착 컨베이어 좌표
        self.place_pos_left = np.array([7.2, 0.7, 0.4])
        self.place_pos_center = np.array([8.0, 0.0, 0.4])
        self.place_pos_right = np.array([7.2, -0.7, 0.4])
        self.place_pos_default = np.array(
            [-np.pi / 2, -np.pi / 2, -np.pi / 2, -np.pi / 2, np.pi / 2, 0])

        # 카트에 놓을 좌표
        self.place_cart_2= np.array([11.3, -2.2, 0.7])
        self.place_cart_3 = np.array([11.3, 0.0, 0.7])
        self.place_cart_4 = np.array([11.3, 2.2, 0.7])

        # 2번 공정 초기화 로봇 pose
        self.place_init_2 = np.array([10.7, -2.6, 0.5])
        self.place_init_3 = np.array([10.7, -0.4, 0.5])
        self.place_init_4 = np.array([10.7, 1.8, 0.5])

        self.task_phase = 0
        self._wait_counter = 0
        self._cube_index = 0
        self.cube = None
        self.cube_name = ""
        self.cube_spawn = np.array([-1.63, 0.0, 2])
        self.cube_prim = "/World/Trash_Random"

        # sensor init 부분
        self.timeline = omni.timeline.get_timeline_interface()
        self.lidar_interface = _range_sensor.acquire_lidar_sensor_interface()

        # sensor spawn 위치
        self.cam_body_pos = np.array([6.3, 1.55, 0.55])
        self.lidar_body_pos = np.array([6.8, 0.0, 0.175])
        self.lidar_sensor_pos = np.array([6.8, 0.0, 0.365])

        self.val = 0 # 1(R, 윗쪽) 2(G, 가운데) 3(B,아래쪽)
        self.depth = 0
        self.cam_color = None

        # UR10_2,3,4 전용 FSM 상태 저장용 ---
        self.extra_robot_tasks = {}   # 나중에 setup_post_load에서 채움
        
        self.extra_robot_lane_ys = {
            "my_ur10_2": -2.2,
            "my_ur10_3":  0.0,
            "my_ur10_4":  2.2,
        }
        # x 픽업 기준 (컨베이어 따라오는 방향 기준)
        self.pickup_x_threshold = 10.5

        # y 허용 폭 (레인 폭의 절반) – 레인 중심 ± 이 값 안에 있는 큐브만 본다
        self.lane_y_half_width = 0.4

        # 어떤 큐브가 이미 어느 로봇에 할당 되었는지 추적
        self.assigned_cube_names = set()

        # 우리가 관리하는 큐브 리스트 (setup_scene에서 채움)
        self.cubes = []

        # 로봇 이름 -> 담당 카트 매핑 (UR10_2,3,4)
        self.robot_to_cart = {
            "my_ur10_2": "/World/Cart_1",
            "my_ur10_3": "/World/Cart_2",
            "my_ur10_4": "/World/Cart_3",
        }

        # 각 카트에 쌓인 큐브 개수
        self.cart_cube_count = {
            "/World/Cart_1": 0,
            "/World/Cart_2": 0,
            "/World/Cart_3": 0,
        }

        # 카트 이동 여부 플래그
        self.cart_moving = {
            "/World/Cart_1": False,
            "/World/Cart_2": False,
            "/World/Cart_3": False,
        }

        # 카트 바퀴 이름(UR10_Cart 예제에서 쓰던 것 그대로)
        self.wheel_dof_names_left = ["wheel_left_front_joint", "wheel_left_back_joint"]
        self.wheel_dof_names_right = ["wheel_right_front_joint", "wheel_right_back_joint"]
        self.wheel_dof_names = self.wheel_dof_names_left + self.wheel_dof_names_right

        # 카트 바퀴 반경 / 속도
        self.cart_wheel_radius = 0.15
        self.cart_speed = 0.5


        # cart 모델의 바퀴 joint 이름 (너의 UR10_Cart 예제 그대로)
        self.wheel_dof_names_left = ["wheel_left_front_joint", "wheel_left_back_joint"]
        self.wheel_dof_names_right = ["wheel_right_front_joint", "wheel_right_back_joint"]
        self.wheel_dof_names = self.wheel_dof_names_left + self.wheel_dof_names_right

        

        return

    def write_log(self,key, value):
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        with open(LOG_PATH, "w") as f:
            f.write(f"{key}:{value}")

    def setup_scene(self):
        world = self.get_world()

        # sensor 구현
        self.cam_body = world.scene.add(
            DynamicCylinder(
                prim_path="/World/cam_body",
                name="cam_body",
                position=self.cam_body_pos,
                scale=np.array([0.1, 0.1, 0.1]),
            )
        )
        self.camera = Camera(
            prim_path="/World/cam_body/camera",
            frequency=30,
            resolution=(100, 100),
            orientation=rot_utils.euler_angles_to_quats(
                np.array([0.0, -5.0, 270.0]),
                degrees=True
            ),
        )
        self.lidar_body = world.scene.add(
            DynamicCylinder(
                prim_path="/World/lidar_body",
                name="lidar_body",
                position=self.lidar_body_pos,
                scale=np.array([0.05, 0.05, 0.35]),
            )
        )

        self.lidar = world.scene.add(
            DynamicCylinder(
                prim_path="/World/lidar_sensor",
                name="lidar_sensor",
                position=self.lidar_sensor_pos,
                scale=np.array([0.03, 0.03, 0.03]),
            )
        )

        result, prim = omni.kit.commands.execute(
            "RangeSensorCreateLidar",
            path="/Lidar",
            parent="/World/lidar_sensor",
            min_range=0.1,
            max_range=0.8,
            draw_points=False,
            draw_lines=True,
            horizontal_fov=1.0,
            vertical_fov=1.0,
            horizontal_resolution=0.5,
            vertical_resolution=0.5,
            rotation_rate=100,
            high_lod=True,
            yaw_offset=0.0,
            enable_semantics=False,
        )
        self.lidar_path = prim.GetPath().pathString

        # 배경 / 그라운드
        ### ur10 로봇 부분 ###
        self.background_usd = "/home/rokey/Downloads/background.usd"
        add_reference_to_stage(
            usd_path=self.background_usd, prim_path="/World/Background"
            )

        world.scene.add_default_ground_plane()

        # === Cart 3대 생성 ===
        cart_usd = "/home/rokey/Downloads/cart.usd"
        cart_prim_paths = [
            "/World/Cart_1",
            "/World/Cart_2",
            "/World/Cart_3",
        ]
        cart_positions = [
            (11.45, -2.2, 0.0),
            (11.45,  0.0, 0.0),
            (11.45,  2.2, 0.0),
        ]

        for prim_path, pos in zip(cart_prim_paths, cart_positions):
            add_reference_to_stage(
                usd_path=cart_usd,
                prim_path=prim_path,
            )
            cart_prim = prim_utils.get_prim_at_path(prim_path)
            # cart.usd 안에 xformOp:translate 이 있다고 가정하고 위치 세팅
            cart_prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*pos))

        # UR10 로봇 4대 생성
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
            sys.exit()

        asset_path = assets_root_path + "/Isaac/Robots/UniversalRobots/ur10/ur10.usd"

        # 각 로봇의 prim path / name 정의
        self.robot_prim_paths = [
            "/World/UR10",      # 기존 첫 번째
            "/World/UR10_2",    # 추가 1
            "/World/UR10_3",    # 추가 2
            "/World/UR10_4",    # 추가 3
        ]
        self.robot_names = [
            "my_ur10",
            "my_ur10_2",
            "my_ur10_3",
            "my_ur10_4",
        ]

        for prim_path, name in zip(self.robot_prim_paths, self.robot_names):
            robot_prim = add_reference_to_stage(
                usd_path=asset_path, prim_path=prim_path
            )
            robot_prim.GetVariantSet("Gripper").SetVariantSelection("Short_Suction")

            gripper = SurfaceGripper(
                end_effector_prim_path=f"{prim_path}/ee_link",
                surface_gripper_path=f"{prim_path}/ee_link/SurfaceGripper",
            )

            ur10 = world.scene.add(
                SingleManipulator(
                    prim_path=prim_path,
                    name=name,
                    end_effector_prim_path=f"{prim_path}/ee_link",
                    gripper=gripper,
                )
            )
            ur10.set_joints_default_state(
                positions=np.array(
                    [
                        -np.pi / 2,
                        -np.pi / 2,
                        -np.pi / 2,
                        -np.pi / 2,
                        np.pi / 2,
                        0,
                    ]
                )
            )

        # # 랜덤 큐브
        # first_color = self._get_next_cube_color()
        # if first_color is None:
        #     first_color = np.array([0.5, 0.5, 0.5])  # 혹시 모를 fallback (회색)

        # cube0 = world.scene.add(
        #     DynamicCuboid(
        #         prim_path=self.cube_prim,
        #         name="random_cube_0",
        #         position=self.cube_spawn,
        #         scale=np.array([0.15, 0.15, 0.15]),
        #         color=first_color,
        #     )
        # )
        # self.cubes.append(cube0)


        return

    async def setup_post_load(self):
        # 씬 로드 시 자동 실행, 오류시 삭제
        self.reset_simulation_state()

        self._world = self.get_world()
        self.camera.initialize()
        self.camera.add_motion_vectors_to_frame()

        # 첫 번째 큐브 레퍼런스
        self.cube_name = f"random_cube_{self._cube_index}"
        self.cube = self._world.scene.get_object(self.cube_name)

        # 첫 번째 로봇 (기존 제어 대상)
        self.robots = self._world.scene.get_object("my_ur10")
        self.cspace_controller = RMPFlowController(
            name="my_ur10_cspace_controller",
            robot_articulation=self.robots,
            attach_gripper=True,
        )
        self.robots.set_world_pose(position=self.robot_position)
        
        
        # 추가 세 로봇 위치 세팅
        extra_names = ["my_ur10_2", "my_ur10_3", "my_ur10_4"]
        for name, base_pos in zip(extra_names, self.robot_positions_extra):
            r = self._world.scene.get_object(name)
            if r is None:
                print(f"[WARN] extra robot {name} not found")
                continue

            r.set_world_pose(position=base_pos)

            controller = RMPFlowController(
                name=f"{name}_controller",
                robot_articulation=r,
                attach_gripper=True,
            )


        # UR10_2,3,4용 컨트롤러 + FSM 상태 초기화 ---
        self.extra_robot_tasks = {}

        extra_configs = [
            {
                "name": "my_ur10_2",
                "base_pos": self.robot_positions_extra[0],
                # 이 로봇이 담당하는 월드 좌표 영역 (x_min, x_max), (y_min, y_max)
                "x_range": (-0.5, 0.5),
                "y_range": (-3.0, -1.5),
                "place_pos": self.place_cart_2,
            },
            {
                "name": "my_ur10_3",
                "base_pos": self.robot_positions_extra[1],
                "x_range": (-0.5, 0.5),
                "y_range": (-0.75, 0.75),
                "place_pos": self.place_cart_3,
            },
            {
                "name": "my_ur10_4",
                "base_pos": self.robot_positions_extra[2],
                "x_range": (-0.5, 0.5),
                "y_range": (1.5, 3.0),
                "place_pos": self.place_cart_4, 
            },
        ]

        for cfg in extra_configs:
            name = cfg["name"]
            base_pos = cfg["base_pos"]
            x_range = cfg["x_range"]
            y_range = cfg["y_range"]

            robot = self._world.scene.get_object(name)
            if robot is None:
                print(f"[WARN] extra robot {name} not found")
                continue

            robot.set_world_pose(position=base_pos)

            controller = RMPFlowController(
                name=f"{name}_controller",
                robot_articulation=robot,
                attach_gripper=True,
            )

            self.extra_robot_tasks[name] = {
                "robot": robot,
                "controller": controller,
                "base_pos": base_pos,
                "x_range": x_range,
                "y_range": y_range,
                "place_pos": cfg["place_pos"],
                "phase": 1,
                "wait_counter": 0,
                "random_cube_pos": None,
            }

        # ----- Cart 1/2/3 를 WheeledRobot 으로 등록 -----
        self.carts = {}  # prim_path -> WheeledRobot 객체

        cart_paths = ["/World/Cart_1", "/World/Cart_2", "/World/Cart_3"]
        cart_names = ["cart_1", "cart_2", "cart_3"]

        for prim_path, name in zip(cart_paths, cart_names):
            cart = self._world.scene.add(
                WheeledRobot(
                    prim_path=prim_path,
                    name=name,
                    wheel_dof_names=self.wheel_dof_names,  # __init__에서 정의한 4개 조인트 이름
                )
            )
            self.carts[prim_path] = cart
            print(f"[CartSetup] WheeledRobot 등록 완료: {name} ({prim_path})")
            cart.initialize()

        # 물리 콜백 등록
        self._world.add_physics_callback("sim_step", callback_fn=self.physics_step)
        await self._world.play_async()
        self.task_phase = 0

        # 첫번째 큐브 생성 없애고 여기에 추가
        self._spawn_new_cube() 

        return

    # x: 11/28 = x:100 x=40 // y: 11.5/15.5 = y :100 y=74
    def get_camera_color(self):
        """카메라 중앙 픽셀의 색상 반환"""
        cam_pixel = self.camera.get_rgb()
        cam_rgb = cam_pixel[57:93, 8:48] #[H:W]
        total_rgb = cam_rgb.sum(axis=1)
        _total_rgb = total_rgb.sum(axis=0)

        print(_total_rgb)

        if _total_rgb[0] > _total_rgb[1] and _total_rgb[0] > _total_rgb[2]:
            return "red"
        elif _total_rgb[1] > _total_rgb[0] and _total_rgb[1] > _total_rgb[2]:
            return "green"
        else:
            return "blue"

    def _step_extra_robot(self, task: dict, step_size: float):
        """UR10_2/3/4 한 대에 대해 pick&place FSM 한 스텝 수행"""

        robot = task["robot"]
        controller = task["controller"]
        base_pos = task["base_pos"]
        phase = task["phase"]

        cube = self.cube

        # ----- phase 1: 큐브가 집기 위치까지 올 때까지 기다림 -----
        if phase == 1:
            cube_position, cube_orientation = cube.get_world_pose()

            current_x_position = cube_position[0]
            if current_x_position >= 10.5 and current_x_position < 11:
                print(f"[{robot.name}] cube reached pickup range: {current_x_position}")
                task["phase"] = 2

        # ----- phase 2: 살짝 대기 후 큐브 위치 저장 -----
        elif phase == 2:
            if task["wait_counter"] < 10:
                task["wait_counter"] += 1
            else:
                task["random_cube_pos"], _ = cube.get_world_pose()
                task["wait_counter"] = 0
                task["phase"] = 3

        # ----- phase 3: 큐브 위로 이동 -----
        elif phase == 3:
            _target_position = task["random_cube_pos"].copy()
            _target_position[2] = 0.4

            end_effector_orientation = euler_angles_to_quat(
                np.array([0.0, np.pi / 2, 0.0])
            )
            action = controller.forward(
                target_end_effector_position=_target_position,
                target_end_effector_orientation=end_effector_orientation,
            )
            robot.apply_action(action)
            current_joint_positions = robot.get_joint_positions()

            if np.all(
                np.abs(current_joint_positions[:6] - action.joint_positions) < 0.001
            ):
                controller.reset()
                task["phase"] = 4

        # ----- phase 4: 큐브까지 내려가기 -----
        elif phase == 4:
            _target_position = task["random_cube_pos"].copy()
            _target_position[2] = 0.3

            end_effector_orientation = euler_angles_to_quat(
                np.array([0.0, np.pi / 2, 0.0])
            )
            action = controller.forward(
                target_end_effector_position=_target_position,
                target_end_effector_orientation=end_effector_orientation,
            )
            robot.apply_action(action)
            current_joint_positions = robot.get_joint_positions()

            if np.all(
                np.abs(current_joint_positions[:6] - action.joint_positions) < 0.001
            ):
                controller.reset()
                task["phase"] = 5

        # ----- phase 5: 그리퍼 닫기 -----
        elif phase == 5:
            robot.gripper.close()
            task["phase"] = 6

        # ----- phase 6: 다시 위로 -----
        elif phase == 6:
            _target_position = task["random_cube_pos"].copy()
            _target_position[2] = 0.6

            end_effector_orientation = euler_angles_to_quat(
                np.array([0.0, np.pi / 2, 0.0])
            )
            action = controller.forward(
                target_end_effector_position=_target_position,
                target_end_effector_orientation=end_effector_orientation,
            )
            robot.apply_action(action)
            current_joint_positions = robot.get_joint_positions()

            if np.all(
                np.abs(current_joint_positions[:6] - action.joint_positions) < 0.001
            ):
                controller.reset()
                task["phase"] = 7

        # ----- phase 7: 공용 place 위치(예: center)로 이동 -----
        elif phase == 7:
            place_pos = task["place_pos"]
            _target_position = place_pos

            end_effector_orientation = euler_angles_to_quat(
                np.array([0.0, np.pi / 2, 0.0])
            )
            action = controller.forward(
                target_end_effector_position=_target_position,
                target_end_effector_orientation=end_effector_orientation,
            )
            robot.apply_action(action)
            current_joint_positions = robot.get_joint_positions()

            if np.all(
                np.abs(current_joint_positions[:6] - action.joint_positions) < 0.001
            ):
                controller.reset()
                task["phase"] = 8

        # ----- phase 8: 내려놓기 -----
        elif phase == 8:
            robot.gripper.open()

            # 어떤 카트인지 찾기 (UR10_2 → Cart_1, UR10_3 → Cart_2, UR10_4 → Cart_3)
            cart_path = self.robot_to_cart.get(robot.name, None)
            if cart_path is not None:
                self.cart_cube_count[cart_path] += 1
                print(f"[{robot.name}] cube placed on {cart_path}. count = {self.cart_cube_count[cart_path]}")

                # 🔥카트 위 큐브가 2개 이상이면 이동 플래그 ON
                if self.cart_cube_count[cart_path] >= 2:
                    self.cart_moving[cart_path] = True
                    print(f"[{cart_path}] now moving because cube >= 2!")

            task["phase"] = 9


        # ----- phase 9: 살짝 위로 빼고 종료 -----
        elif phase == 9:
            if self.val == 1 : 
                _target_position = self.place_init_2
            elif self.val == 2 :
                _target_position = self.place_init_3
            elif self.val == 3 :
                _target_position = self.place_init_4

            end_effector_orientation = euler_angles_to_quat(
                np.array([0.0, np.pi / 2, 0.0])
            )
            action = controller.forward(
                target_end_effector_position=_target_position,
                target_end_effector_orientation=end_effector_orientation,
            )
            robot.apply_action(action)
            current_joint_positions = robot.get_joint_positions()

            if np.all(
                np.abs(current_joint_positions[:6] - action.joint_positions) < 0.001
            ):
                controller.reset()
                task["phase"] = 1  # 여기서는 한 번만 수행하고 멈춤

    # get_world_pos 관련 문제점 - 큐브관련 변수가 초기화 되지않아서 처음만 정상작동하고 다시 로드시 기존의 갯수를 연속으로 셈, 초기화하는 과정필요
    def reset_simulation_state(self):
        self._cube_index = 0
        self.cube = None
        self.cubes = []
        self.assigned_cube_names.clear()

        self._color_counts = {0: 0, 1: 0, 2: 0}
        self.task_phase = 0
        self._wait_counter = 0
        self._random_cube_position = np.array([0,0,0])

        self.cart_cube_count = {
            "/World/Cart_1": 0,
            "/World/Cart_2": 0,
            "/World/Cart_3": 0,
        }
        self.cart_moving = {
            "/World/Cart_1": False,
            "/World/Cart_2": False,
            "/World/Cart_3": False,
        }
        print("[RoxProject] Simulation state fully reset.")

    # 큐브 색별 갯수 측정
    def _get_next_cube_color(self):
        """아직 2개 미만인 색들 중에서 랜덤 선택. 다 찼으면 None 리턴."""
        # 아직 제한 미만인 색 인덱스만 후보에 넣기
        available_indices = [idx for idx, cnt in self._color_counts.items() if cnt < self._max_per_color]

        if not available_indices:
            print("[Cube] 모든 색이 이미 최대 개수에 도달했습니다. 더 이상 색을 가진 큐브를 생성하지 않습니다.")
            return None

        # 후보 중에서 랜덤 선택
        idx = random.choice(available_indices)
        self._color_counts[idx] += 1
        return self.colors[idx]
    

    def _spawn_new_cube(self):
        # 위치변경
        cube_color = self._get_next_cube_color()
        if cube_color is None:
            return
        
        self._cube_index += 1
        self.cube_name = f"random_cube_{self._cube_index}"
        prim_path = f"/World/RandomCube_{self._cube_index}"

        # cube_color = self._get_next_cube_color()
        # if cube_color is None:
        #     return

        self.cube = self._world.scene.add(
            DynamicCuboid(
                prim_path=prim_path,
                name=self.cube_name,
                position=self._random_cube_spawn_position,
                scale=np.array([0.15, 0.15, 0.15]),
                color=cube_color,
            )
        )
        self._random_cube_position = self._random_cube_spawn_position.copy()
        print(f"Spawned new cube: {self.cube_name} color={cube_color} at {self._random_cube_spawn_position}")



    def get_lidar_depth(self):
        """LiDAR로부터 최소 깊이값 추출"""
        depth = self.lidar_interface.get_linear_depth_data(self.lidar_path)
        depth = np.array(depth, dtype=np.float32)
        self.valid = depth[np.isfinite(depth) & (depth > 0)]

        if self.valid.size > 0:
            return float(self.valid.min())
        return None

    def physics_step(self, step_size):
        # cube가 아직 없으면 로직 실행 금지
        if self.cube is None:
            return

        # 먼저 UR10_2,3,4에 대한 FSM 한 스텝씩 실행 ---
        if self.val == 1:
            active_name = "my_ur10_2"
        elif self.val == 2:
            active_name = "my_ur10_3"
        elif self.val == 3:
            active_name = "my_ur10_4"
        else:
            active_name = None

        # 선택된 로봇만 FSM 실행
        if active_name is not None :
            task = self.extra_robot_tasks[active_name]
            self._step_extra_robot(task, step_size)

        # phase 0: 초음파 센서 판자 위 객체 확인
        if self.task_phase == 0 : 
            value = self.get_lidar_depth()
            # print(value)
            if value < 0.363 :
                #로그기록추가-depth
                # self.write_log("depth", value) 
                self.task_phase = 1

            # if current_x_position >= 5.0:
                # print(f"Cube X ({current_x_position}) reached target range (>= -0.09).")
        
        # 이하 로직은 기존 그대로
        # phase 2: 카메라 RGB 확인 후 분류 라벨정하기
        elif self.task_phase == 1:
            if self._wait_counter < 50:
                self._wait_counter += 1
            
            else :
                cube_position, cube_orientation = self.cube.get_world_pose()
                current_x_position = cube_position[0]
                self.cam_color = self.get_camera_color()
                
                if self.cam_color == "red":
                    self.val = 1
                elif self.cam_color == "green":
                    self.val = 2
                elif self.cam_color == "blue":
                    self.val = 3
                
                print(self.cam_color)
                print(self.val)
                self.write_log(self.cam_color, self._color_counts[self.val-1])
                self._wait_counter = 0
                self.task_phase = 2

            #로그기록추가-val(color)
            # self.write_log("color",self.cam_color)
            # self.write_log(self.cam_color, self._color_counts[self.val-1])
            # self.task_phase = 2

        elif self.task_phase == 2:
            if self._wait_counter < 100:
                self._wait_counter += 1
            else:
                self._random_cube_position, _ = self.cube.get_world_pose()
                self._wait_counter = 0
                self.task_phase = 3

        # phase 3: 큐브 위로 이동 (상승)
        elif self.task_phase == 3:
            _target_position = self._random_cube_position.copy() - self.robot_position
            _target_position[2] = 0.4  # 위에서 접근

            end_effector_orientation = euler_angles_to_quat(np.array([0.0, np.pi / 2, 0.0]))
            action = self.cspace_controller.forward(
                target_end_effector_position=_target_position,
                target_end_effector_orientation=end_effector_orientation,
            )
            self.robots.apply_action(action)
            current_joint_positions = self.robots.get_joint_positions()

            if np.all(np.abs(current_joint_positions[:6] - action.joint_positions) < 0.001):
                self.cspace_controller.reset()
                self.task_phase = 4

        # phase 4: 큐브까지 내려가기
        elif self.task_phase == 4:
            _target_position = self._random_cube_position.copy() - self.robot_position
            _target_position[2] = 0.15  # 큐브 가까이

            end_effector_orientation = euler_angles_to_quat(np.array([0.0, np.pi / 2, 0.0]))
            action = self.cspace_controller.forward(
                target_end_effector_position=_target_position,
                target_end_effector_orientation=end_effector_orientation,
            )
            self.robots.apply_action(action)
            current_joint_positions = self.robots.get_joint_positions()

            if np.all(np.abs(current_joint_positions[:6] - action.joint_positions) < 0.001):
                self.cspace_controller.reset()
                self.task_phase = 5

        # phase 5: 그리퍼 닫아서 큐브 집기
        elif self.task_phase == 5:
            self.robots.gripper.close()
            self.task_phase = 6

        # phase 6: 큐브 들고 위로 다시 올리기
        elif self.task_phase == 6:
            _target_position = self._random_cube_position.copy() - self.robot_position
            _target_position[2] = 0.4

            end_effector_orientation = euler_angles_to_quat(np.array([0.0, np.pi / 2, 0.0]))
            action = self.cspace_controller.forward(
                target_end_effector_position=_target_position,
                target_end_effector_orientation=end_effector_orientation,
            )
            self.robots.apply_action(action)

            current_joint_positions = self.robots.get_joint_positions()

            if np.all(np.abs(current_joint_positions[:6] - action.joint_positions) < 0.001):
                self.cspace_controller.reset()
                self.task_phase = 7

        # phase 7: 판(plate) 위 목표 위치로 이동 (현재 스택 레벨에 맞게)
        elif self.task_phase == 7:
            if self.val == 1:  # right
                _target_position = self.place_pos_right - self.robot_position
                end_effector_orientation = euler_angles_to_quat(np.array([-np.pi / 2, np.pi / 2, 0.0]))

            elif self.val == 3:  # left
                _target_position = self.place_pos_left - self.robot_position
                end_effector_orientation = euler_angles_to_quat(np.array([0.0, np.pi / 2, 0.0]))

            elif self.val == 2:  # center
                _target_position = self.place_pos_center - self.robot_position
                end_effector_orientation = euler_angles_to_quat(np.array([np.pi / 2, np.pi / 2, 0]))

            action = self.cspace_controller.forward(
                target_end_effector_position=_target_position,
                target_end_effector_orientation=end_effector_orientation,
            )
            self.robots.apply_action(action)

            current_joint_positions = self.robots.get_joint_positions()

            if np.all(np.abs(current_joint_positions[:6] - action.joint_positions) < 0.001):
                self.cspace_controller.reset()
                self.task_phase = 8

        # phase 8: 그리퍼 열어서 판 위에 큐브 내려놓기
        elif self.task_phase == 8:
            self.robots.gripper.open()
            self.task_phase = 9

        # phase 9: 로봇 팔을 위로 살짝 빼주고 (리셋 포즈로) + 스택 레벨 증가
        elif self.task_phase == 9:
            _target_position = np.array([0.18, 0.7, 0.5])
            end_effector_orientation = euler_angles_to_quat(np.array([0.0, np.pi / 2, 0.0]))
            action = self.cspace_controller.forward(
                target_end_effector_position=_target_position,
                target_end_effector_orientation=end_effector_orientation,
            )
            self.robots.apply_action(action)

            current_joint_positions = self.robots.get_joint_positions()

            if np.all(np.abs(current_joint_positions[:6] - action.joint_positions) < 0.001):
                self.cspace_controller.reset()
                self._wait_counter = 0
                self.task_phase = 10
        
        # phase 10: 일정 시간 기다렸다가 "새로운" 큐브를 컨베이어 시작 위치에 생성
        elif self.task_phase == 10:
            self._cube_spawn_timer += step_size

            if self._cube_spawn_timer >= self._cube_spawn_interval:
                self._spawn_new_cube()

                self._cube_spawn_timer = 0.0
                self.task_phase = 0

        # --------------------- 카트 이동 ---------------------
        for cart_path, moving in self.cart_moving.items():
            if not moving:
                continue

            # setup_post_load 에서 등록한 WheeledRobot 객체 사용
            cart = self.carts.get(cart_path, None)
            if cart is None:
                continue

            num_dof = cart.num_dof
            joint_velocities = np.zeros(num_dof, dtype=np.float32)

            v_linear = self.cart_speed
            wheel_omega = v_linear / self.cart_wheel_radius

            # 바퀴 조인트 인덱스 (매 step 계산해도 되고, 나중에 캐시해도 됨)
            wheel_idx_left = [cart.get_dof_index(n) for n in self.wheel_dof_names_left]
            wheel_idx_right = [cart.get_dof_index(n) for n in self.wheel_dof_names_right]

            for idx in wheel_idx_left:
                joint_velocities[idx] = wheel_omega
            for idx in wheel_idx_right:
                joint_velocities[idx] = wheel_omega

            action = ArticulationAction(joint_velocities=joint_velocities)
            cart.apply_action(action)


        return
