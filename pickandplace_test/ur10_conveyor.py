import numpy as np
import sys
import carb
import random 

from isaacsim.examples.interactive.base_sample import BaseSample

from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path
from isaacsim.robot.manipulators.grippers import SurfaceGripper
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.core.api.objects import DynamicCuboid

import isaacsim.robot_motion.motion_generation as mg
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.prims import SingleArticulation


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
        return

    def reset(self):
        mg.MotionPolicyController.reset(self)
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position, robot_orientation=self._default_orientation
        )


class UR10_Conveyor(BaseSample):
    def __init__(self) -> None:
        super().__init__()

        self.colors = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),  
            np.array([0.0, 0.0, 1.0]),  
        ]

        self._random_cube_spawn_position = np.array([-1.5, 0.0, 0.5])
        self._random_cube_position = self._random_cube_spawn_position.copy()

        self.robot_position = np.array([0.3, 0.0, 0.0])

        self.plate_position = np.array([0.7, 0.6, 0.18])
        self.PLATE_COLOR = np.array([0.3, 0.3, 0.3])

        self.place_base_position = np.array([0.7, 0.6, 0.25])

        self._cube_height = 0.05
        self._stack_level = 0  

        self.task_phase = 1
        self._wait_counter = 0
        
        self._cube_spawn_interval = 3.0
        self._cube_spawn_timer = 0.0

        self._cube_index = 0
        self.cube = None
        self.cube_name = ""

        return

    def setup_scene(self):
        world = self.get_world()
        self.background_usd = "/home/rokey/Downloads/back.usd"
        add_reference_to_stage(usd_path=self.background_usd, prim_path="/World/Background")

        world.scene.add_default_ground_plane()

        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
            simulation_app.close()
            sys.exit()

        asset_path = assets_root_path + "/Isaac/Robots/UniversalRobots/ur10/ur10.usd"
        robot = add_reference_to_stage(usd_path=asset_path, prim_path="/World/UR10")
        robot.GetVariantSet("Gripper").SetVariantSelection("Short_Suction")

        gripper = SurfaceGripper(
            end_effector_prim_path="/World/UR10/ee_link", surface_gripper_path="/World/UR10/ee_link/SurfaceGripper"
        )

        ur10 = world.scene.add(
            SingleManipulator(
                prim_path="/World/UR10", name="my_ur10", end_effector_prim_path="/World/UR10/ee_link", gripper=gripper
            )
        )
        ur10.set_joints_default_state(
            positions=np.array([-np.pi / 2, -np.pi / 2, -np.pi / 2, -np.pi / 2, np.pi / 2, 0])
        )

        world.scene.add(
            DynamicCuboid(
                prim_path="/World/Plate",
                name="plate",
                position=self.plate_position,
                scale=np.array([0.6, 0.6, 0.02]),
                color=self.PLATE_COLOR,
            )
        )

        first_color = random.choice(self.colors)

        first_cube_prim = "/World/RandomCube_0"
        world.scene.add(
            DynamicCuboid(
                prim_path=first_cube_prim,
                name="random_cube_0",
                position=self._random_cube_spawn_position,
                scale=np.array([0.05, 0.05, 0.05]),
                color=first_color,
            )
        )

        return

    async def setup_post_load(self):
        self._world = self.get_world()

        # 첫 번째 큐브 레퍼런스
        self.cube_name = f"random_cube_{self._cube_index}"
        self.cube = self._world.scene.get_object(self.cube_name)

        self.robots = self._world.scene.get_object("my_ur10")
        self.cspace_controller = RMPFlowController(
            name="my_ur10_cspace_controller", robot_articulation=self.robots, attach_gripper=True
        )

        # UR10 베이스 위치 세팅
        self.robots.set_world_pose(
            position=self.robot_position,
        )

        # 물리 콜백 등록
        self._world.add_physics_callback("sim_step", callback_fn=self.physics_step)

        await self._world.play_async()
        self.task_phase = 1
        return

    def _get_current_place_position(self):
        place_pos = self.place_base_position.copy()
        place_pos[2] = self.place_base_position[2] + self._stack_level * self._cube_height
        return place_pos

    def _spawn_new_cube(self):
        self._cube_index += 1
        self.cube_name = f"random_cube_{self._cube_index}"
        prim_path = f"/World/RandomCube_{self._cube_index}"

        cube_color = random.choice(self.colors)

        self.cube = self._world.scene.add(
            DynamicCuboid(
                prim_path=prim_path,
                name=self.cube_name,
                position=self._random_cube_spawn_position,
                scale=np.array([0.05, 0.05, 0.05]),
                color=cube_color,
            )
        )
        self._random_cube_position = self._random_cube_spawn_position.copy()
        print(f"Spawned new cube: {self.cube_name} color={cube_color} at {self._random_cube_spawn_position}")

    def physics_step(self, step_size):
        # phase 1: 큐브가 집기 위치까지 컨베이어 타고 올 때까지 기다림
        if self.task_phase == 1:
            cube_position, cube_orientation = self.cube.get_world_pose()
            current_x_position = cube_position[0]
            if current_x_position >= -0.09:
                print(f"Cube X ({current_x_position}) reached target range (>= -0.09).")
                self.task_phase = 2

        # phase 2: 살짝 대기한 뒤 큐브 현재 위치 저장
        elif self.task_phase == 2:
            if self._wait_counter < 10:
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
            _target_position[2] = 0.2  # 큐브 가까이

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
            place_pos = self._get_current_place_position()
            _target_position = place_pos - self.robot_position

            end_effector_orientation = euler_angles_to_quat(np.array([0.0, np.pi / 2, 0.0]))
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
            place_pos = self._get_current_place_position()
            _target_position = place_pos - self.robot_position
            _target_position[2] = place_pos[2] + 0.25  # 해당 스택에서 조금 더 위로

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
                self._cube_spawn_timer = 0.0

                self._stack_level += 1

                self.task_phase = 10

        # phase 10: 일정 시간 기다렸다가 "새로운" 큐브를 컨베이어 시작 위치에 생성
        elif self.task_phase == 10:
            self._cube_spawn_timer += step_size

            if self._cube_spawn_timer >= self._cube_spawn_interval:
                self._spawn_new_cube()

                self._cube_spawn_timer = 0.0
                self.task_phase = 1

        return
