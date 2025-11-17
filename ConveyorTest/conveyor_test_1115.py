"""

구현 단계

1. 컨베이어 벨트 환경을 USD환경으로 만들기 (완)
2. USD환경 standalone 방식에서 불러오기 (완)
3. 불러온 USD환경에서 컨베이어 프림 이용하기 (이름 출력으로 완)
4. 큐브 컨베이어 위에 생성하기. (완)

"""



from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})


import numpy as np
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.sensors.camera import Camera
import omni.usd

# 1) 컨베이어 환경 USD 환경 주소
usd_path = "/home/rokey/RoxProject/ConveyorTest/conveyor_test_ver1.usd" 

# 2) USD stage 열기
omni.usd.get_context().open_stage(usd_path)



world = World(stage_units_in_meters=1.0)
######################## 객체 생성 구역 ####################################
cube_2 = world.scene.add(
    DynamicCuboid(
        prim_path="/spawn_cube_1",
        name="cube_1",
        position=np.array([-1.5, 0.0, 2.5]),
        scale=np.array([0.3, 0.3, 0.3]),
        color=np.array([255, 0, 0]),
    )
)



#########################################################################
world.reset()

# 3) 컨베이어 프림 -> 변수로 지정 후 이름 출력
conveyor_path = "/World/Conveyor/ConveyorTrack_start"
stage = omni.usd.get_context().get_stage()
# conveyor_start = stage.GetPrimAtPath(conveyor_path)

while simulation_app.is_running():

    world.step(render=True)

simulation_app.close()