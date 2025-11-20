import numpy as np
from isaacsim.examples.interactive.base_sample import BaseSample
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.robot.wheeled_robots.robots import WheeledRobot
from isaacsim.core.utils.types import ArticulationAction


class CartTest(BaseSample):
    def __init__(self):
        super().__init__()

        # 바퀴 joint 이름 (cart.usd 모델과 동일해야 함)
        self.wheel_dof_names_left = [
            "wheel_left_front_joint",
            "wheel_left_back_joint",
        ]
        self.wheel_dof_names_right = [
            "wheel_right_front_joint",
            "wheel_right_back_joint",
        ]
        self.wheel_dof_names = self.wheel_dof_names_left + self.wheel_dof_names_right

        # 바퀴 반경
        self.wheel_radius = 0.15

        # m/s
        self.cart_speed = 1.0

    def setup_scene(self):
        world = self.get_world()
        world.scene.add_default_ground_plane()

        # cart.usd 로드
        add_reference_to_stage(
            usd_path="/home/rokey/Downloads/cart.usd",
            prim_path="/World/CartTest",
        )

    async def setup_post_load(self):
        world = self.get_world()

        # 카트를 WheeledRobot 으로 등록
        self.cart = world.scene.add(
            WheeledRobot(
                prim_path="/World/CartTest",
                name="test_cart",
                wheel_dof_names=self.wheel_dof_names,
            )
        )

        # 리셋 필수 (바퀴 joint index를 읽기 위해)
        await world.reset_async()

        # joint index 찾기
        self.wheel_idx_left = [self.cart.get_dof_index(n) for n in self.wheel_dof_names_left]
        self.wheel_idx_right = [self.cart.get_dof_index(n) for n in self.wheel_dof_names_right]

        # physics loop 등록
        world.add_physics_callback("cart_test", self.physics_step)

        await world.play_async()

    def physics_step(self, step_size: float):

        # 선속도(m/s) → 각속도(rad/s)
        wheel_omega = self.cart_speed / self.wheel_radius

        # 전체 DOF 속도 배열
        num_dof = self.cart.num_dof
        joint_vel = np.zeros(num_dof, dtype=np.float32)

        # 바퀴 회전 속성 입력
        for idx in self.wheel_idx_left:
            joint_vel[idx] = wheel_omega
        for idx in self.wheel_idx_right:
            joint_vel[idx] = wheel_omega

        # Isaac Sim articulation action
        action = ArticulationAction(joint_velocities=joint_vel)
        self.cart.apply_action(action)


# 실행
if __name__ == "__main__":
    CartTest().run()
