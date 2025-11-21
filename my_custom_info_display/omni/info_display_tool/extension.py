import omni.ext
import omni.ui as ui
import omni.timeline
import omni.kit.app
import omni.usd
import carb
import os
import numpy as np
from typing import Dict

# ----------------------------------------------
# 이전 단계에서 작성한 DetailedInfoPanel 클래스를 여기에 전체 붙여넣습니다.
# (on_update, simulate_detection_logic, update_ui_elements, destroy 메소드 포함)
# ----------------------------------------------

#로그경로
LOG_PATH = "/home/rokey/isaacsim/exts/my_custom_info_display/logs/rox_data.txt" 

def reset_log_file_init():
    log_dir = os.path.dirname(LOG_PATH)
    os.makedirs(log_dir, exist_ok=True)  # 폴더 없으면 생성
    # w 모드로 열면 기존 내용은 전부 삭제되고 새 파일처럼 초기화됨
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("")  # 필요하면 헤더 같은 거 쓰기
    
class DetailedInfoPanel:
    # ... (이전에 만든 DetailedInfoPanel 클래스 내용 전체) ...
    def __init__(self):
        self.data: Dict[str, any] = {
            "is_playing": False,
            "current_time": 0.0,
            "last_detected_color": "None",
            "classification_id": "00",
            "total_count": 0,
            "red_count": 0,
            "blue_count": 0,
            "green_count": 0,
        }
        
        # NOTE: 창의 크기와 Docking 설정은 필요에 따라 조정하세요.
        self.window = ui.Window(title="Info Display", width=400, height=200,allow_docking=False)
        self.last_position = 0

        # 이전 상태 기억용 플래그 (재생/정지 전환 체크용)
        # self._was_playing = False

        with self.window.frame:
            with ui.VStack(spacing=5, height=0):
                self.status_label = ui.Label(
                    "Simulation Inactive", 
                    height=20, 
                    alignment=ui.Alignment.CENTER,
                    style={"font_size": 18, "color": 0xFF888888}
                )
                ui.Separator()
                
                # Grid를 대체했던 안정적인 HStack 레이아웃 사용
                # separator 당 한 데이터만 출력 가능
                with ui.Frame(height=0): 
                    with ui.HStack(spacing=5): 
                        ui.Label("Time:", width=150, alignment=ui.Alignment.RIGHT)
                        self.time_label = ui.Label("--", width=150, alignment=ui.Alignment.LEFT)

                    # with ui.HStack(spacing=5):
                    #     ui.Label("Color:", width=150, alignment=ui.Alignment.RIGHT)
                    #     self.color_label = ui.Label("--", width=150, alignment=ui.Alignment.LEFT)
                        
                    # with ui.HStack(spacing=5):
                    #     ui.Label("Class ID:", width=150, alignment=ui.Alignment.RIGHT)
                    #     self.id_label = ui.Label("--", width=150, alignment=ui.Alignment.LEFT)
                
                ui.Separator()
                with ui.Frame(height=0): 
                    with ui.HStack(spacing=5):
                        ui.Label("Color:", width=150, alignment=ui.Alignment.RIGHT)
                        self.color_label = ui.Label("--", width=150, alignment=ui.Alignment.LEFT,style={"color":0xFF888888})

                ui.Separator() 
                with ui.Frame(height=0):
                    with ui.HStack(spacing=5):
                        ui.Label("Total Processed:", width=150, alignment=ui.Alignment.RIGHT)
                        self.total_count_label = ui.Label("0", width=150, alignment=ui.Alignment.LEFT)
                
                ui.Separator() 
                with ui.HStack(spacing=5):
                        ui.Label("R/G/B Count:", width=150, alignment=ui.Alignment.RIGHT)
                        self.rgb_count_label = ui.Label("R:0 / G:0 / B:0", width=150, alignment=ui.Alignment.LEFT)

        self.timeline = omni.timeline.get_timeline_interface()
        self.subscription = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
            self.on_update, name="DetailedInfoPanel Update"
        )

        # 1프레임 뒤에 실행하여 위치 강제 설정
        self._position_sub = (
            omni.kit.app.get_app()
            .get_update_event_stream()
            .create_subscription_to_pop(self._set_initial_position_once)
        )
        
        self.update_ui_elements()
        carb.log_info("Detailed Info Panel Initialized.")
   
    # 🔹 로그 파일 + 카운터 모두 초기화
    def reset_log_file_load(self):
        # 1) 로그 파일 비우기
        log_dir = os.path.dirname(LOG_PATH)
        os.makedirs(log_dir, exist_ok=True)
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            f.write("")

        # 2) 내부 카운터 값 초기화
        self.data["total_count"] = 0
        self.data["red_count"] = 0
        self.data["green_count"] = 0
        self.data["blue_count"] = 0

        # 3) UI 라벨도 같이 초기화
        self.total_count_label.text = "0"
        self.rgb_count_label.text = "R:0 / G:0 / B:0"
        self.color_label.text = '--'
        self.color_label.set_style({"color": 0xFF888888})

    def on_update(self, event):
        # if self.timeline.is_playing():
            # self.data["current_time"] = self.timeline.get_current_time()
            # self.simulate_detection_logic() 

        # # 현재 타임라인 상태 확인
        # is_playing = self.timeline.is_playing()
        # # 이전에는 멈춰 있었는데 지금 처음으로 재생되면 → 로그 파일 초기화
        # if is_playing and not self._was_playing:
        #     self.reset_log_file_load()
        # # 다음 프레임을 위해 상태 저장
        # self._was_playing = is_playing

        self.data["current_time"] = self.timeline.get_current_time()
        self.simulate_detection_logic()  
        self.update_ui_elements()

    # 4. (예시) 더미 감지 로직 - 실제 시뮬레이션 데이터로 대체 필요
    def simulate_detection_logic(self):
        try:
            if os.path.exists(LOG_PATH):
                with open(LOG_PATH, "r") as f:
                    lines = f.readlines()

                for line in lines:
                    # if "color:" in line:
                    #     self.color_label.text = f"{line.split(':')[1].strip()}"
                    #     if self.color_label.text == "red":
                    #         self.color_label.set_style({"color": 0xFF0000FF})

                    #     elif self.color_label.text =="green":
                    #         self.color_label.set_style({"color": 0xFF58FA2A})

                    #     elif self.color_label.text == "blue":
                    #         self.color_label.set_style({"color": 0xFFFA2A2A})
                    if ":" in line:  # [red , 0]
                        self.color_label.text, self.color_count = line.split(':')
                        if self.color_label.text == "red":
                            self.color_label.set_style({"color": 0xFF0000FF})
                            self.data["red_count"] = self.color_count
                        elif self.color_label.text =="green":
                            self.color_label.set_style({"color": 0xFF58FA2A})
                            self.data["green_count"] = self.color_count
                        elif self.color_label.text == "blue":
                            self.color_label.set_style({"color": 0xFFFA2A2A})
                            self.data["blue_count"] = self.color_count
                        self.data["total_count"] = int(self.data["red_count"]) + int(self.data["green_count"]) + int(self.data["blue_count"])
                        if self.data["total_count"] == 6:
                            self.data["total_count"] = f"6 last!"

                    # elif "depth=" in line:
                    #     self.depth_label.text = f"Depth: {line.split('=')[1].strip()}"
                    #
        except:
            pass


        # frame = int(self.data["current_time"] * 60)


    # 2. UI 위젯 업데이트 함수
    def update_ui_elements(self):
        is_playing = self.timeline.is_playing()
        
        if is_playing:
            self.status_label.text = "Simulation Running"
            self.status_label.set_style({"color": 0xFF58FA2A})
        else:
            self.status_label.text = "Simulation Paused/Stopped"
            self.status_label.set_style({"color": 0xFFFA2A2A})
            
        self.time_label.text = f"{self.data['current_time']:.2f} s"

        # self.color = self.color_label
        # class_id = self.data['classification_id']
        
        #현재작동안함
        # color_style = {"color": 0xFFFFFFFF}
        # if self.color == "red":
        #     self.color_label.set_style({"color": 0xFFFF5555})
        # elif self.color == "green":
        #     self.color_label.set_style({"color": 0xFF58FA2A})
        # elif self.color == "blue":
        #     self.color_label.set_style({"color": 0xFFFA2A2A})

        # self.color_label.set_style(self.color_style)
        # self.id_label.text = class_id

        self.total_count_label.text = str(self.data['total_count'])
        self.rgb_count_label.text = (
            f"R:{self.data['red_count']} / G:{self.data['green_count']} / B:{self.data['blue_count']}"
        )
    
    def _set_initial_position_once(self, e):
        # 왼쪽 하단 고정 위치 적용
        self.window.position_x = 940     # 화면 왼쪽에서 10px
        self.window.position_y = 535     # 화면 아래에서 10px

        # 이 subscription은 한 번만 실행되면 되므로 제거
        self._position_sub = None


    # 5. 창 종료 및 구독 해지
    def destroy(self):
        self.subscription = None
        self.window.destroy()
        
# ----------------------------------------------


# 확장 기능의 시작점
class MyInfoDisplayToolExtension(omni.ext.IExt):
    # 확장 기능이 활성화될 때 (Isaac Sim 시작 시)
    def on_startup(self, ext_id):
        carb.log_info(f"Extension {ext_id} startup")

        # DetailedInfoPanel 인스턴스를 생성하여 창을 띄웁니다.
        self._panel = DetailedInfoPanel()
    
        # 실행할 때마다 로그 파일 새로 만들기 / 초기화
        ctx = omni.usd.get_context()
        self._stage_event_sub = ctx.get_stage_event_stream().create_subscription_to_pop(
            self.on_stage_event
        )
        #reset_log_file_init()

    # 아깐 안되던데 왜 작동하지?
    def on_stage_event(self, event):
        if event.type == int(omni.usd.StageEventType.OPENED):
            print("Scene Loaded → Reset Log")
            self._panel.reset_log_file_load()

    # 확장 기능이 비활성화될 때 (Isaac Sim 종료 시 또는 사용자가 확장 기능을 끌 때)
    def on_shutdown(self):
        carb.log_info("Extension shutdown")
        # 생성된 패널 객체를 정리하고 창을 닫습니다.
        if self._panel:
            self._panel.destroy()
            self._panel = None
