# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

# import argparse

# parser = argparse.ArgumentParser() # CLI 명령어 입력 할 때 --version 같은 옵션 추가하는 기능
# parser.add_argument("--disable_output", action="store_true", help="Disable debug output.")
# parser.add_argument("--test", action="store_true", help="Enable test mode (fixed frame count).")
# args, _ = parser.parse_known_args()

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.sensors.camera import Camera


############################# 객체 형성 구역 및 초기화 구역 ###################################################
my_world = World(stage_units_in_meters=1.0)

cube_2 = my_world.scene.add(
    DynamicCuboid(
        prim_path="/new_cube_2",
        name="cube_1",
        position=np.array([0.0, 0.0, 0.1]),
        scale=np.array([0.5, 0.5, 0.5]),
        color=np.array([255, 0, 0]),
    )
)

camera = Camera(
    prim_path="/World/camera",
    position=np.array([-3, 0.0, 0.25]),
    frequency=30,
    resolution=(100, 100) #(width,height)
)

my_world.scene.add_default_ground_plane()
my_world.reset()
camera.initialize()

i = 1
camera.add_motion_vectors_to_frame()
reset_needed = False

##############################################################################################################



######################## Isaac sim 시뮬레이션 동작 구역 ###########################################################
while simulation_app.is_running():
    my_world.step(render=True)
    
    if i % 100 == 0:
        cam_pixel = camera.get_rgb()
        cam_rgb = cam_pixel[49:52,49:52]
        total_rgb = cam_rgb.sum(axis=1)
        _total_rgb = total_rgb.sum(axis=0)

        print(f"cube2의 RGB 값은 {_total_rgb}")
        if _total_rgb[0] > _total_rgb[1] and _total_rgb[0] > _total_rgb[2]:
            cam_color = '빨간색 입니다.'
        else :
            cam_color = '빨간색이 아닙니다.' 
        print(f"cube의 색상은 {cam_color}")
    if my_world.is_stopped() and not reset_needed:
        reset_needed = True
    if my_world.is_playing():
        if reset_needed:
            my_world.reset()
            reset_needed = False
    i += 1


simulation_app.close()

################################################################################################################
