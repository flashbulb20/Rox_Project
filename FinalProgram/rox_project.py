import numpy as np
import sys
import random 
import omni

from isaacsim.examples.interactive.base_sample import BaseSample
import omni.isaac.core.utils.prims as prim_uils
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
from pxr import Gf,UsdGeom
# from sensors import SensorSystem


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

class RoxProject(BaseSample):
    def __init__(self) -> None:
        super().__init__()

        self.colors = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),  
            np.array([0.0, 0.0, 1.0]),  
        ]

        self.robot_position = np.array([7.25, 0.0, 0.3])

        ### 도착 컨베이어 좌표 ###
        
        self.place_pos_left = np.array([7.2, 0.7, 0.4])
        self.place_pos_center = np.array([8.0, 0.0, 0.4])
        self.place_pos_right = np.array([7.2, -0.7, 0.4])
        self.place_pos_default = np.array([-np.pi / 2, -np.pi / 2, -np.pi / 2, -np.pi / 2, np.pi / 2, 0])

        self.task_phase = 1
        self._wait_counter = 0
        self._cube_index = 0
        self.cube = None
        self.cube_name = ""
        self.cube_spawn = np.array([-1.63, 0.0, 2])
        self.cube_prim = "/World/Trash_Random"


        ### sensor init 부분 ###
        self.timeline = omni.timeline.get_timeline_interface()
        self.lidar_interface = _range_sensor.acquire_lidar_sensor_interface()

        ### sensor spawn 위치 ###
        self.cam_body_pos = np.array([6.3, 1.55, 0.55])
        self.lidar_body_pos = np.array([6.8, 0.0, 0.175])

        self.val = 2 # 1(R, 윗쪽) 2(G, 가운데) 3(B,아래쪽)
        return



    def setup_scene(self):
        world = self.get_world()
        # self.sensor = SensorSystem()

        #### sensor 구현 ###
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
            frequency=30 ,
            resolution=(100, 100),
            orientation=rot_utils.euler_angles_to_quats(
                np.array([0.0, -5.0, 270.0]),
                degrees=True
            )
        )
        self.lidar_body = world.scene.add(
            DynamicCylinder(
                prim_path = "/World/lidar_sensor",
                name = "lidar_sensor",
                position = self.lidar_body_pos,
                scale = np.array([0.05, 0.05, 0.35])

            )
        )

        result, prim = omni.kit.commands.execute(
            "RangeSensorCreateLidar",
            path="/Lidar",
            parent="/World/lidar_sensor",
            min_range=0.1,
            max_range=0.5,
            draw_points=False,
            draw_lines=True,
            horizontal_fov=1.0,
            vertical_fov=1.0,
            horizontal_resolution=0.5,
            vertical_resolution=0.5,
            rotation_rate=100,
            high_lod=True,
            yaw_offset= 0.0,
            enable_semantics=False,
        )
        stage = omni.usd.get_context().get_stage()
        prim_lidar = stage.GetPrimAtPath("/World/lidar_sensor/Lidar")
        xform = UsdGeom.XformCommonAPI(prim_lidar)
        xform.SetTranslate(Gf.Vec3f(1, 2, 3))


        ### ur10 로봇 부분 ###
        self.background_usd = "/home/rokey/Downloads/background.usd"
        add_reference_to_stage(usd_path=self.background_usd, prim_path="/World/Background")

        world.scene.add_default_ground_plane()

        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
            sys.exit()

        asset_path = assets_root_path + "/Isaac/Robots/UniversalRobots/ur10/ur10.usd"
        robot = add_reference_to_stage(usd_path=asset_path, prim_path="/World/UR10")
        robot.GetVariantSet("Gripper").SetVariantSelection("Short_Suction")

        # 로봇 스케일 변화    
        # ur10 = XFormPrim("/World/UR10")
        # ur10.set_local_scale(np.array([x, y, z]))

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
                prim_path=self.cube_prim,
                name="random_cube_0",
                position=self.cube_spawn,
                scale=np.array([0.15, 0.15, 0.15]),
                color=random.choice(self.colors),
            )
        )
        return
    async def setup_post_load(self):
        self._world = self.get_world()
        self.camera.initialize()
        self.camera.add_motion_vectors_to_frame()
        # 첫 번째 큐브 레퍼런스
        self.cube_name = f"random_cube_{self._cube_index}"
        self.cube = self._world.scene.get_object(self.cube_name)

        self.robots = self._world.scene.get_object("my_ur10")
        self.cspace_controller = RMPFlowController(
            name="my_ur10_cspace_controller", robot_articulation=self.robots, attach_gripper=True
        )
        self.robots.set_world_pose(
            position = self.robot_position
        )
        # 물리 콜백 등록
        self._world.add_physics_callback("sim_step", callback_fn=self.physics_step)
        await self._world.play_async()
        self.task_phase = 1
        return
    

    def physics_step(self, step_size):
        # phase 1: 
        if self.task_phase == 1:
            cube_position, cube_orientation = self.cube.get_world_pose()
            current_x_position = cube_position[0]
            if current_x_position >= 5.0:
                print(f"Cube X ({current_x_position}) reached target range (>= -0.09).")
                self.task_phase = 2

        # phase 2: 살짝 대기한 뒤 큐브 현재 위치 저장
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

########################################################################################################

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
            if self.val == 1 : # right
                _target_position = self.place_pos_right - self.robot_position 
                end_effector_orientation = euler_angles_to_quat(np.array([ - np.pi / 2 , np.pi/2, 0.0]))

            elif self.val == 3: # left
                _target_position = self.place_pos_left - self.robot_position
                end_effector_orientation = euler_angles_to_quat(np.array([0.0, np.pi / 2, 0.0]))

            elif self.val == 2 : # center
                _target_position = self.place_pos_center - self.robot_position
                end_effector_orientation = euler_angles_to_quat(np.array([ np.pi / 2, np.pi / 2, 0]))

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
        

            end_effector_orientation = euler_angles_to_quat(np.array([0.0, np.pi / 2 ,0.0]))
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

        return
