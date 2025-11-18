import numpy as np
import sys
import random
import omni
import carb

from isaacsim.examples.interactive.base_sample import BaseSample
import omni.isaac.core.utils.prims as prim_utils
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path
from isaacsim.robot.manipulators.grippers import SurfaceGripper
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.core.api.objects import DynamicCuboid

import isaacsim.robot_motion.motion_generation as mg
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.prims import SingleArticulation

# 센서 시스템 (카메라 + LiDAR)
from .sensors import ArmSensorSystem


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

        self.articulation_rmp = mg.ArticulationMotionPolicy(robot_articulation, self.rmp_flow, physics_dt)

        mg.MotionPolicyController.__init__(self, name=name, articulation_motion_policy=self.articulation_rmp)
        (
            self._default_position,
            self._default_orientation,
        ) = self._articulation_motion_policy._robot_articulation.get_world_pose()
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position, robot_orientation=self._default_orientation
        )

    def reset(self):
        mg.MotionPolicyController.reset(self)
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position, robot_orientation=self._default_orientation
        )


class RoxProject(BaseSample):
    def __init__(self) -> None:
        super().__init__()

        self.colors = [
            np.array([1.0, 0.0, 0.0]),  # red
            np.array([0.0, 1.0, 0.0]),  # green
            np.array([0.0, 0.0, 1.0]),  # blue
        ]

        # UR10 base pose (world)
        self.robot_position = np.array([7.25, 0.0, 0.3])

        # === Conveyor place positions (world) ===
        self.place_pos_left = np.array([7.2, 0.7, 0.4])
        self.place_pos_center = np.array([8.0, 0.0, 0.4])
        self.place_pos_right = np.array([7.2, -0.7, 0.4])

        # which place position to use: 0=black, 1=right, 2=left, 3=center
        self.val = 0

        # task state
        self.task_phase = 1
        self._wait_counter = 0

        # cube info
        self._cube_index = 0
        self.cube = None
        self.cube_name = ""
        self.cube_spawn = np.array([-1.63, 0.0, 2.0])
        self.cube_prim = "/World/Trash_Random"

        # for robot motion
        self.cspace_controller = None
        self._random_cube_position = None

        # sensor system (camera + lidar)
        self.sensors: ArmSensorSystem | None = None
        self._sensor_frame_count = 0

    def setup_scene(self):
        world = self.get_world()

        # background
        self.background_usd = "/home/dell/Downloads/background.usd"
        add_reference_to_stage(usd_path=self.background_usd, prim_path="/World/Background")

        world.scene.add_default_ground_plane()

        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
            sys.exit(1)

        asset_path = assets_root_path + "/Isaac/Robots/UniversalRobots/ur10/ur10.usd"
        robot_prim = add_reference_to_stage(usd_path=asset_path, prim_path="/World/UR10")
        robot_prim.GetVariantSet("Gripper").SetVariantSelection("Short_Suction")

        gripper = SurfaceGripper(
            end_effector_prim_path="/World/UR10/ee_link",
            surface_gripper_path="/World/UR10/ee_link/SurfaceGripper",
        )

        ur10 = world.scene.add(
            SingleManipulator(
                prim_path="/World/UR10",
                name="my_ur10",
                end_effector_prim_path="/World/UR10/ee_link",
                gripper=gripper,
            )
        )
        ur10.set_joints_default_state(
            positions=np.array([-np.pi / 2, -np.pi / 2, -np.pi / 2, -np.pi / 2, np.pi / 2, 0.0])
        )

        # first cube
        world.scene.add(
            DynamicCuboid(
                prim_path=self.cube_prim,
                name="random_cube_0",
                position=self.cube_spawn,
                scale=np.array([0.15, 0.15, 0.15]),
                color=random.choice(self.colors),
            )
        )

    async def setup_post_load(self):
        self._world = self.get_world()

        # cube handle
        self.cube_name = f"random_cube_{self._cube_index}"
        self.cube = self._world.scene.get_object(self.cube_name)

        # robot & controller
        self.robots = self._world.scene.get_object("my_ur10")
        self.cspace_controller = RMPFlowController(
            name="my_ur10_cspace_controller",
            robot_articulation=self.robots,
            attach_gripper=True,
        )
        self.robots.set_world_pose(position=self.robot_position)

        # === attach ArmSensorSystem ===
        # 카메라/라이다가 달린 실린더를 로봇 베이스 근처에 설치
        self.sensors = ArmSensorSystem(
            world=self._world,
            cam_body_pos=(7.25, 0.0, 0.9),  # 로봇 위쪽에 약간 띄워서 설치
        )
        carb.log_info("[RoxProject] ArmSensorSystem attached to world")

        # physics callback
        self._world.add_physics_callback("sim_step", callback_fn=self.physics_step)

        self.task_phase = 1

    def _debug_sensors(self):
        """주기적으로 센서 값 출력 (디버깅용)."""
        if self.sensors is None:
            return

        self._sensor_frame_count += 1
        if self._sensor_frame_count % 30 != 0:
            return

        color = self.sensors.get_camera_color()
        depth = self.sensors.get_lidar_depth()
        carb.log_info(f"[SENSOR] color={color}, depth={depth}")
        #color_save[] = color
        # if color_save != color[255,0,0] and x 5

    def physics_step(self, step_size: float):
        # 센서 디버그 출력
        self._debug_sensors()

        # phase 1: 큐브가 특정 X 위치에 도달할 때까지 대기
        if self.task_phase == 1:
            cube_position, cube_orientation = self.cube.get_world_pose()
            current_x_position = cube_position[0]

        # color 정보 받아오고 판단, 센서 1프레임에 한 번만 읽어서 분류
            color = None
            color = self.sensors.get_camera_color()
            if self.val == 0 and self.sensors is not None:
                if color == "빨간색":
                    self.val = 1  # right
                elif color == "초록색":
                    self.val = 2  # left
                elif color == "파란색":
                    self.val = 3  # center
                else:
                    self.val = 0  # black or unknown

            carb.log_info(f"[SORT] detected color={color}, val={self.val}")
            
            if current_x_position >= 5.0:
                print(f"Cube X ({current_x_position}) reached target range (>= 5.0).")
                # 현재 큐브 위치를 잠깐 저장해 두어도 됨
                self._random_cube_position = cube_position.copy()
                self.task_phase = 2

        # phase 2: 잠깐 대기 후, 큐브 위치 다시 스냅샷
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

        # phase 4: 큐브까지 내려가기 (그리퍼 바로 위까지)
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

        # phase 7: 도착 컨베이어 쪽으로 이동 (left/center/right)
        elif self.task_phase == 7:
            if self.val == 1:  # right
                _target_position = self.place_pos_right - self.robot_position
                end_effector_orientation = euler_angles_to_quat(np.array([-np.pi / 2, np.pi / 2, 0.0]))
            elif self.val == 2:  # left
                _target_position = self.place_pos_left - self.robot_position
                end_effector_orientation = euler_angles_to_quat(np.array([0.0, np.pi / 2, 0.0]))
            else:  # center (3)
                _target_position = self.place_pos_center - self.robot_position
                end_effector_orientation = euler_angles_to_quat(np.array([np.pi / 2, np.pi / 2, 0.0]))

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

        # phase 9: 로봇 팔을 살짝 빼기 (간단한 리트랙션 포즈)
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

        # phase 10: (옵션) 이후 동작을 추가하거나 시퀀스 종료
        elif self.task_phase == 10:
            # 여기서 다시 새로운 큐브를 스폰하거나, 로봇을 원위치로 보내는 로직을 추가할 수 있음
            pass

        return
